"""
run_webcam.py — Pipeline HRC sem câmera OAK-D.
(Versão de diagnóstico — inclui ferramentas para testar hipóteses de falha)

Substitui o run.py original usando:
  - OpenCV para captura de vídeo (webcam do notebook)
  - MediaPipe Pose (CPU) para estimativa de esqueleto
  - IntentionPredictor (DLinear) para predição de intenção

Modos de diagnóstico disponíveis:
  --diag        Exibe entropia, probabilidades brutas e estatísticas de Z
  --no_qrot     Desativa a rotação câmera→mundo (quaternion identidade)
  --proc_fps N  Limita o processamento a N fps (padrão: 8, igual ao treino)
  --replay PKL  Reproduz um arquivo .pkl gravado com a OAK-D

Uso básico:
    python run_webcam.py --show --task webcam001

Diagnóstico completo:
    python run_webcam.py --show --task webcam001 --diag --proc_fps 8

Testar sem rotação de câmera:
    python run_webcam.py --show --task webcam001 --diag --no_qrot

Replay de pkl original:
    python run_webcam.py --diag --replay human_traj/abc/abc001.pkl
"""

import os
import sys
import cv2
import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from pathlib import Path

FILE_DIR = Path(__file__).parent
sys.path.append(str(FILE_DIR / 'depthai_blazepose'))
sys.path.append(str(FILE_DIR / 'traj_intention'))

from mediapipe_fallback import MediaPipePoseModule
from predict import IntentionPredictor
from Dataset import INTENTION_LIST

# [ROS] import rospy
# [ROS] from std_msgs.msg import String

# ── Quaternion de rotação câmera→mundo (mesmo do run.py original) ────────────
# Calibrado para a posição da OAK-D no experimento original.
# Com webcam em posição diferente, recalibrar este valor ou usar --no_qrot.
_CAM_TO_WORLD_Q = np.array(
    [0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32
)
_IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _qrot(q, v):
    """Rotação de vetor v pelo quaternion q. Idêntico ao run.py e Dataset.py."""
    qvec = q[..., 1:]
    uv   = np.cross(qvec, v, len(q.shape) - 1)
    uuv  = np.cross(qvec, uv, len(q.shape) - 1)
    return v + 2 * (q[..., :1] * uv + uuv)


def camera_to_world(X, quat=None):
    """
    Aplica a transformação câmera→mundo.
    quat: quaternion a usar (None = usar _CAM_TO_WORLD_Q calibrado).
          Passe _IDENTITY_Q para desativar a rotação (hipótese H2).
    """
    if quat is None:
        quat = _CAM_TO_WORLD_Q
    return _qrot(np.tile(quat, (*X.shape[:-1], 1)), X)


# ── Limiar de movimento: abaixo disso a pose é considerada estática ──────────
# Valor em coordenadas normalizadas (0-2) após min-max.
STILLNESS_THRESHOLD = 0.015

# ── Nomes das classes (ordem do INTENTION_LIST) ───────────────────────────────
CLASS_NAMES = [k for k, v in sorted(INTENTION_LIST.items(), key=lambda x: x[1])]


def get_intention_name(index):
    for key, value in INTENTION_LIST.items():
        if value == index:
            return key
    return 'no_action'


def send_intention(intention_name):
    """Publica a intenção. Atualmente apenas imprime; descomentar para ROS."""
    print(f'[INTENTION] {intention_name}')
    # [ROS] pub.publish(intention_name)


def parse_camera_source(camera_arg):
    """
    Converte --camera para uma fonte aceita pelo OpenCV.
    Exemplos: "0" -> 0, "/dev/video2" -> "/dev/video2", "auto" -> auto-detectar.
    """
    camera_arg = str(camera_arg).strip()
    if camera_arg.lower() == 'auto':
        return 'auto'
    try:
        return int(camera_arg)
    except ValueError:
        return camera_arg


def list_video_devices():
    """Lista dispositivos Linux /dev/video* visíveis no sistema."""
    return sorted(str(path) for path in Path('/dev').glob('video*'))


def camera_backend_id(name):
    """Mapeia nome de backend para constante OpenCV."""
    name = str(name).lower()
    if name == 'v4l2':
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def open_video_capture(source, backend):
    backend_id = camera_backend_id(backend)
    if backend_id == cv2.CAP_ANY:
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source, backend_id)


def configure_camera(cap, width, height, fps, fourcc, buffer_size):
    """Solicita formato/resolução ao driver antes da primeira leitura real."""
    fourcc = str(fourcc).strip().upper()
    if len(fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if buffer_size > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)


def camera_fourcc_string(cap):
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    chars = [chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]
    text = ''.join(chars)
    return text if text.strip() else '----'


def ensure_bgr_frame(frame):
    """Garante frame BGR de 3 canais para MediaPipe, desenho e HUD."""
    if frame is None:
        return None
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def overlay_scale(frame):
    """Escala HUD/painéis para resoluções maiores sem exagerar em 480p."""
    h = frame.shape[0]
    return float(np.clip(h / 900.0, 1.0, 1.8))


def _camera_candidate_is_valid(cap):
    """Confirma que a câmera abriu e realmente entrega frames."""
    if not cap.isOpened():
        return False
    for _ in range(3):
        ret, frame = cap.read()
        frame = ensure_bgr_frame(frame)
        if ret and frame is not None and frame.size > 0:
            return True
    return False


def open_camera(args):
    """
    Abre câmera por índice, caminho ou auto-detecção.
    Retorna (cap, source_usada). Se falhar, cap vem fechado.
    """
    source = parse_camera_source(args.camera)

    if source == 'auto':
        candidates = list_video_devices() + list(range(10))
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            cap = open_video_capture(candidate, args.capture_backend)
            configure_camera(
                cap, args.cam_width, args.cam_height, args.cam_fps,
                args.cam_fourcc, args.camera_buffer
            )
            if _camera_candidate_is_valid(cap):
                return cap, candidate
            cap.release()
        return open_video_capture(-1, args.capture_backend), 'auto'

    cap = open_video_capture(source, args.capture_backend)
    configure_camera(
        cap, args.cam_width, args.cam_height, args.cam_fps,
        args.cam_fourcc, args.camera_buffer
    )
    return cap, source


def print_camera_error(camera_source):
    devices = list_video_devices()
    print(f'Erro: não foi possível abrir a câmera {camera_source}')
    if devices:
        print('Dispositivos de vídeo encontrados:')
        for device in devices:
            print(f'  - {device}')
        print('Tente executar com, por exemplo: --camera /dev/video0 ou --camera auto')
    else:
        print('Nenhum dispositivo /dev/video* foi encontrado.')
        print('Verifique se a webcam está conectada, liberada para este ambiente e se o usuário tem permissão de vídeo.')


# ─────────────────────────────────────────────────────────────────────────────
# Bloco de diagnóstico
# ─────────────────────────────────────────────────────────────────────────────

def compute_diag(predictor, inputs, poses_raw):
    """
    Calcula métricas de diagnóstico sem aplicar nenhuma restrição.

    Retorna dict com:
      probs      — probabilidades softmax para cada classe (4 valores)
      entropy    — entropia de Shannon das probabilidades
      z_raw_min  — mínimo do eixo Z antes da normalização
      z_raw_max  — máximo do eixo Z antes da normalização
      z_raw_std  — desvio-padrão do eixo Z antes da normalização
      z_norm_std — desvio-padrão do eixo Z após normalização min-max
    """
    with torch.no_grad():
        _, pred_logits = predictor.model(inputs)
    probs_t = F.softmax(pred_logits, dim=1)[0].detach()
    entropy_val = float(Categorical(probs=probs_t).entropy())
    probs_np = probs_t.numpy()

    z = poses_raw[:, :, 2]
    return {
        'probs'     : probs_np,
        'entropy'   : entropy_val,
        'z_raw_min' : float(z.min()),
        'z_raw_max' : float(z.max()),
        'z_raw_std' : float(z.std()),
        'z_norm_std': float(((2 * (z - z.min()) / (z.max() - z.min() + 1e-8))).std()),
    }


def draw_diag_panel(frame, diag, motion_disp, is_still, qrot_active, proc_fps_target):
    """
    Sobrepõe painel de diagnóstico no canto superior direito do frame.
    Desenhado sobre o display_frame (já ampliado) para fontes nítidas.
    """
    h, w = frame.shape[:2]
    scale    = overlay_scale(frame)
    margin   = int(8 * scale)
    pad      = int(8 * scale)
    line_h   = int(28 * scale)
    fs       = 0.58 * scale
    thick    = max(1, int(round(scale)))
    n_rows   = len(CLASS_NAMES) + 8
    panel_w  = int(420 * scale)
    panel_x  = w - panel_w - margin
    panel_y  = int(10 * scale)
    bar_x    = panel_x + int(250 * scale)
    bar_h    = max(8, int(10 * scale))
    bar_w    = int(130 * scale)

    # Fundo opaco — sem addWeighted para evitar piscar
    cv2.rectangle(frame,
                  (panel_x - pad, panel_y - pad),
                  (w - margin // 2, panel_y + line_h * n_rows + pad),
                  (20, 20, 20), -1)

    def put(text, row, color=(220, 220, 220)):
        cv2.putText(frame, text, (panel_x, panel_y + row * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, thick, cv2.LINE_AA)

    put('--- DIAGNOSTICO ---', 0, (100, 255, 100))
    put(f'qrot: {"ON" if qrot_active else "OFF (identidade)"}', 1,
        (100, 200, 255) if qrot_active else (255, 150, 50))
    put(f'proc_fps_alvo: {proc_fps_target}', 2)

    entropy = diag['entropy']
    ent_color = (0, 200, 0) if entropy < 0.4 else (0, 165, 255) if entropy < 0.8 else (0, 0, 255)
    put(f'entropia: {entropy:.3f}', 3, ent_color)
    put(f'motion : {motion_disp:.4f} {"(PARADO)" if is_still else ""}', 4)
    put('classe          prob  barra', 5, (180, 180, 180))

    max_prob = float(max(diag['probs']))
    for i, name in enumerate(CLASS_NAMES):
        prob = float(diag['probs'][i])
        bar_len = int(prob * bar_w)
        bar_color = (0, 200, 0) if prob == max_prob else (100, 100, 200)
        put(f'{name[:14]:<14} {prob:.2f}', 6 + i)
        bar_y = panel_y + (6 + i) * line_h
        cv2.rectangle(frame,
                      (bar_x, bar_y - bar_h - int(4 * scale)),
                      (bar_x + bar_len, bar_y - int(4 * scale)),
                      bar_color, -1)

    row = 6 + len(CLASS_NAMES)
    put(f'Z raw  std={diag["z_raw_std"]:.4f}', row, (180, 180, 100))
    put(f'       [{diag["z_raw_min"]:.3f}, {diag["z_raw_max"]:.3f}]', row + 1, (180, 180, 100))


# ─────────────────────────────────────────────────────────────────────────────
# Modo replay: reproduz um .pkl gravado com OAK-D
# ─────────────────────────────────────────────────────────────────────────────

def run_replay(args):
    """
    Reproduz um arquivo .pkl gravado com a OAK-D e passa os landmarks
    pelo mesmo pipeline de predição. Não usa câmera.

    Serve para H5: verificar se o modelo funciona com dados de treino reais.
    """
    pkl_path = Path(args.replay)
    if not pkl_path.exists():
        print(f'[ERRO] Arquivo não encontrado: {pkl_path}')
        return

    with open(pkl_path, 'rb') as f:
        bodies = pickle.load(f)
    print(f'[REPLAY] {len(bodies)} frames carregados de {pkl_path}')

    predictor = IntentionPredictor(model_type=args.model_type)
    quat = _IDENTITY_Q if args.no_qrot else None

    seq_len    = args.seq_len
    traj_queue = []
    #smoothed_probs = None
    old_intention  = None
    intention_queue = []

    for frame_idx, body in enumerate(bodies):
        lms = body.landmarks
        # Normaliza para 15 joints se veio com 33 (arquivos do OAK-D)
        if lms.shape[0] == 33:
            upperbody = np.concatenate((lms[11:25, :], lms[0:1, :]), axis=0)
        else:
            upperbody = lms  # já são 15

        if len(traj_queue) >= seq_len:
            traj_queue.pop(0)
        traj_queue.append(upperbody)

        if len(traj_queue) < seq_len:
            continue

        poses = np.array(traj_queue)
        motion_disp = float(np.abs(np.diff(poses, axis=0)).mean())

        # Pré-processamento idêntico ao Dataset.py e run.py
        poses_norm  = 2 * (poses - poses.min()) / (poses.max() - poses.min() + 1e-8)
        poses_world = camera_to_world(poses_norm, quat)
        poses_world[:, :, 2] -= poses_world[:, :, 2].min()

        inputs = torch.tensor(poses_world.reshape(1, seq_len, -1)).float()

        diag = None
        if args.diag:
            diag = compute_diag(predictor, inputs, poses)

        _, pred_intention = predictor.predict(inputs, restrict=args.restrict)
        intention = get_intention_name(pred_intention[0].item())

        # Saída no terminal
        if args.diag and diag:
            probs_str = '  '.join(
                f'{CLASS_NAMES[i]}={diag["probs"][i]:.2f}' for i in range(len(CLASS_NAMES))
            )
            print(
                f'[{frame_idx:04d}] intenção={intention:<18} '
                f'entropia={diag["entropy"]:.3f}  motion={motion_disp:.4f}  |  {probs_str}'
            )
        else:
            print(f'[{frame_idx:04d}] intenção={intention}  motion={motion_disp:.4f}')

    print('[REPLAY] Concluído.')


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal: webcam ao vivo
# ─────────────────────────────────────────────────────────────────────────────

def run_live(args):
    show       = args.show
    task       = args.task
    seq_len    = args.seq_len
    send_win   = args.send_window
    restrict   = args.restrict
    save_video = args.video
    save_frames = args.save_frames
    diag_mode  = args.diag
    quat       = _IDENTITY_Q if args.no_qrot else None
    proc_fps   = args.proc_fps   # alvo de FPS de processamento (0 = sem limite)

    ROOT_DIR = FILE_DIR / 'human_traj' / task[:-3]
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = ROOT_DIR / f'images{task[-3:]}'
    if save_frames and img_dir.exists():
        import shutil
        shutil.rmtree(img_dir)
    if save_frames:
        img_dir.mkdir()

    # ── Câmera ──────────────────────────────────────────────────────────────────
    cap, camera_source = open_camera(args)
    if not cap.isOpened():
        print_camera_error(camera_source)
        return
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    cam_fourcc = camera_fourcc_string(cap)
    print(
        f'Câmera aberta ({camera_source}): {img_w}x{img_h} '
        f'@ {cam_fps:.1f} fps  fourcc={cam_fourcc}  backend={args.capture_backend}'
    )

    # ── Janela de exibição (criada uma vez para evitar piscar) ───────────────────
    DISPLAY_SCALE = 2  # fator de ampliação para facilitar leitura
    disp_w = int(img_w * DISPLAY_SCALE)
    disp_h = int(img_h * DISPLAY_SCALE)
    if show:
        cv2.namedWindow('HRC Webcam', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('HRC Webcam', disp_w, disp_h)

    # ── Módulos ─────────────────────────────────────────────────────────────────
    pose_module = MediaPipePoseModule(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        smoothing=True,
    )
    predictor = IntentionPredictor(model_type=args.model_type)

    # ── Vídeo de saída ───────────────────────────────────────────────────────────
    video_out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(
            str(ROOT_DIR / f'{task}_camera_out.mp4'), fourcc, 8, (img_w, img_h)
        )

    # ── Estado ───────────────────────────────────────────────────────────────────
    traj_queue      = []
    intention_queue = []
    old_intention   = None
    frame_count     = 0
    traj_save       = []
    #smoothed_probs  = None

    # Estado persistente para o HUD (evita piscar entre predições)
    last_diag        = None
    last_motion_disp = 0.0
    last_is_still    = False
    last_intention   = 'aguardando...'
    last_score       = 0.0

    # Controle de FPS de processamento (hipótese H3)
    last_proc_time  = 0.0
    proc_interval   = (1.0 / proc_fps) if proc_fps > 0 else 0.0

    fps_counter = 0
    fps_start   = time.monotonic()
    fps         = 0.0
    proc_fps_real = 0.0
    proc_count  = 0
    proc_fps_start = time.monotonic()

    # [ROS] rospy.init_node('intention_webcam', anonymous=True)
    # [ROS] pub = rospy.Publisher('chatter', String, queue_size=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Fim do stream de vídeo.')
            break
        frame = ensure_bgr_frame(frame)
        if frame is None:
            print('Frame vazio recebido da câmera.')
            continue

        # ── Estimativa de pose ───────────────────────────────────────────────────
        body = pose_module.inference(frame)

        if body and body.score > 0.5:
            now = time.monotonic()

            # H3 — Limitador de FPS de processamento
            # Acumula landmarks mas só roda o modelo a cada proc_interval segundos.
            upperbody = body.landmarks  # (15, 3) coords normalizadas

            if len(traj_queue) >= seq_len:
                traj_queue.pop(0)
            traj_queue.append(upperbody)

            traj_save.append(body)
            frame_count += 1

            intention   = None
            is_still    = False
            motion_disp = 0.0
            diag        = None

            # Só roda o modelo se passou o intervalo mínimo entre predições
            if len(traj_queue) == seq_len and (now - last_proc_time) >= proc_interval:
                last_proc_time = now
                proc_count    += 1

                poses = np.array(traj_queue)  # (seq_len, 15, 3)

                # Pré-filtro de movimento: evita rodar o modelo em pose estática (Desabilitado pelo Marcos em 11/04/2026)
                motion_disp = float(np.abs(np.diff(poses, axis=0)).mean())
                if False: #motion_disp < STILLNESS_THRESHOLD:
                    is_still  = True
                    intention = 'no_action'
                    smoothed_probs = None
                else:
                    # Pré-processamento idêntico ao Dataset.py e run.py
                    poses_norm  = 2 * (poses - poses.min()) / (poses.max() - poses.min() + 1e-8)
                    poses_world = camera_to_world(poses_norm, quat)  # H2: quat pode ser identidade
                    poses_world[:, :, 2] -= poses_world[:, :, 2].min()

                    inputs = torch.tensor(
                        poses_world.reshape(1, seq_len, -1)
                    ).float()

                    # H1 / H4 — Diagnóstico: calcula métricas sem restrição
                    if diag_mode:
                        diag = compute_diag(predictor, inputs, poses)
                        last_diag = diag

                    _, pred_intention = predictor.predict(inputs, restrict=restrict)

                    # Suavização exponencial das probabilidades (reduz flickering)
                    #n_classes = len(INTENTION_LIST)
                    #one_hot = np.zeros(n_classes, dtype=np.float32)
                    #one_hot[pred_intention[0].item()] = 1.0
                    #alpha = 0.4
                    #if smoothed_probs is None:
                    #    smoothed_probs = one_hot
                    #else:
                    #    smoothed_probs = alpha * one_hot + (1 - alpha) * smoothed_probs

                    #final_idx = int(smoothed_probs.argmax())
                    #intention = get_intention_name(final_idx)
                    
                    intention = get_intention_name(pred_intention[0].item())

                    # Saída de diagnóstico no terminal
                    if diag_mode and diag:
                        probs_str = '  '.join(
                            f'{CLASS_NAMES[i]}={diag["probs"][i]:.2f}'
                            for i in range(len(CLASS_NAMES))
                        )
                        print(
                            f'[frame {frame_count:04d}] '
                            f'intenção={intention:<18} '
                            f'entropia={diag["entropy"]:.3f}  '
                            f'motion={motion_disp:.4f}  '
                            f'qrot={"ON" if quat is None else "OFF"}  |  {probs_str}'
                        )

                # Persiste estado para o HUD (não pisca entre predições)
                last_motion_disp = motion_disp
                last_is_still    = is_still
                if intention:
                    last_intention = intention

                # Janela de confirmação antes de enviar a intenção
                if intention and intention != 'no_action' and not is_still:
                    if len(intention_queue) < send_win:
                        if not intention_queue or intention == intention_queue[-1]:
                            intention_queue.append(intention)
                        else:
                            intention_queue = [intention]
                    else:
                        if intention == intention_queue[-1] and intention != old_intention:
                            send_intention(intention)
                            old_intention = intention
                        intention_queue = []
                else:
                    intention_queue = []

            # ── Esqueleto no frame original (para salvar) ─────────────────────
            frame = pose_module.draw(frame, body)
            last_score = body.score

        else:
            last_intention = 'sem pessoa detectada'
            last_score     = 0.0

        # Salva frame com esqueleto mas sem HUD (dados mais limpos)
        if save_video and video_out:
            video_out.write(frame)
        elif save_frames and body:
            cv2.imwrite(str(img_dir / f'{frame_count}.png'), frame)

        # ── HUD desenhado no display_frame (resolução ampliada → fontes nítidas) ─
        if show:
            display_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            hud_scale = overlay_scale(display_frame)
            hud_margin = int(10 * hud_scale)
            hud_line = int(34 * hud_scale)
            hud_thick = max(1, int(round(hud_scale)))

            qrot_label   = 'qrot:ON' if (quat is None) else 'qrot:OFF'
            status_color = (0, 200, 0) if not last_is_still else (180, 180, 180)

            # Linha de status no topo direito
            cv2.putText(display_frame, f'fps:{fps:.1f}  {qrot_label}',
                        (disp_w - int(245 * hud_scale), int(30 * hud_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65 * hud_scale,
                        (255, 255, 255), hud_thick, cv2.LINE_AA)

            if last_score == 0.0:
                cv2.putText(display_frame, 'Nenhuma pessoa detectada',
                            (hud_margin, int(55 * hud_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9 * hud_scale,
                            (0, 0, 255), hud_thick, cv2.LINE_AA)
            else:
                # Rodapé com métricas (espaçamento de 34 px para legibilidade)
                cv2.putText(display_frame, f'intention: {last_intention}',
                            (hud_margin, disp_h - hud_line * 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75 * hud_scale,
                            status_color, hud_thick, cv2.LINE_AA)
                cv2.putText(display_frame,
                            f'motion: {last_motion_disp:.4f}  thresh: {STILLNESS_THRESHOLD}',
                            (hud_margin, disp_h - hud_line * 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * hud_scale,
                            (200, 200, 200), hud_thick, cv2.LINE_AA)
                cv2.putText(display_frame,
                            f'frame: {frame_count}  proc_fps: {proc_fps_real:.1f}',
                            (hud_margin, disp_h - hud_line * 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * hud_scale,
                            (255, 255, 255), hud_thick, cv2.LINE_AA)
                cv2.putText(display_frame,
                            f'score: {last_score:.2f}  restrict: {restrict}',
                            (hud_margin, disp_h - hud_margin),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * hud_scale,
                            (255, 255, 255), hud_thick, cv2.LINE_AA)

                # Painel de diagnóstico — usa last_diag (persiste entre predições)
                if diag_mode and last_diag is not None:
                    draw_diag_panel(
                        display_frame, last_diag, last_motion_disp, last_is_still,
                        qrot_active=(quat is None),
                        proc_fps_target=proc_fps
                    )

            cv2.imshow('HRC Webcam', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Contadores de FPS
        fps_counter += 1
        now = time.monotonic()
        if now - fps_start > 1.0:
            fps = fps_counter / (now - fps_start)
            fps_counter = 0
            fps_start   = now
        if now - proc_fps_start > 2.0:
            proc_fps_real = proc_count / (now - proc_fps_start)
            proc_count    = 0
            proc_fps_start = now

    # ── Limpeza ──────────────────────────────────────────────────────────────────
    cap.release()
    pose_module.close()
    if video_out:
        video_out.release()
    cv2.destroyAllWindows()

    with open(str(ROOT_DIR / f'{task}.pkl'), 'wb') as f:
        pickle.dump(traj_save, f)
    print(f'Trajetória salva em {ROOT_DIR}/{task}.pkl ({len(traj_save)} frames)')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pipeline HRC com webcam (sem OAK-D) — versão de diagnóstico'
    )

    # ── Argumentos originais ────────────────────────────────────────────────────
    parser.add_argument('--show', action='store_true',
                        help='Exibir vídeo em tempo real')
    parser.add_argument('--task', default='webcam001',
                        help='Nome da tarefa (6 chars, 3 dígitos no final, ex: webcam001)')
    parser.add_argument('--camera', type=str, default='auto',
                        help='Fonte da câmera: auto, índice ou caminho (ex: auto, 0, /dev/video2)')
    parser.add_argument('--capture_backend', type=str, default='v4l2',
                        choices=['v4l2', 'any'],
                        help='Backend de captura OpenCV. Em Linux, v4l2 costuma ser mais estável para webcams USB.')
    parser.add_argument('--cam_width', type=int, default=1280,
                        help='Largura solicitada para captura. Use 0 para manter o padrão do driver.')
    parser.add_argument('--cam_height', type=int, default=720,
                        help='Altura solicitada para captura. Use 0 para manter o padrão do driver.')
    parser.add_argument('--cam_fps', type=int, default=30,
                        help='FPS solicitado para captura. Use 0 para manter o padrão do driver.')
    parser.add_argument('--cam_fourcc', type=str, default='MJPG',
                        help='Formato solicitado ao driver, ex: MJPG, H264 ou YUYV. MJPG reduz banda USB em webcams compatíveis.')
    parser.add_argument('--camera_buffer', type=int, default=1,
                        help='Tamanho do buffer de captura. 1 reduz latência e frames antigos.')
    parser.add_argument('--seq_len', type=int, default=5,
                        help='Tamanho da janela de frames para predição')
    parser.add_argument('--send_window', type=int, default=3,
                        help='Intenção enviada após N confirmações consecutivas')
    parser.add_argument('--restrict', type=str, default='ood',
                        choices=['no', 'ood', 'working_area', 'all'],
                        help=(
                            'Modo de restrição. '
                            '"ood" aplica filtro de entropia. '
                            '"no" mostra predição bruta sem filtro (útil para diagnóstico).'
                        ))
    parser.add_argument('--model_type', type=str, default='final_intention',
                        choices=['final_intention', 'final_traj'],
                        help='Tipo de modelo de predição')
    parser.add_argument('--video', action='store_true',
                        help='Salvar saída como vídeo')
    parser.add_argument('--save_frames', action='store_true',
                        help='Salvar cada frame com esqueleto como PNG. Desativado por padrão para evitar travamentos no vídeo ao vivo.')

    # ── Argumentos de diagnóstico ───────────────────────────────────────────────
    parser.add_argument('--diag', action='store_true',
                        help=(
                            '[H1/H4] Ativa modo de diagnóstico: exibe entropia, '
                            'probabilidades por classe e estatísticas do eixo Z no terminal e '
                            'no vídeo.'
                        ))
    parser.add_argument('--no_qrot', action='store_true',
                        help=(
                            '[H2] Desativa a rotação câmera→mundo (usa quaternion identidade). '
                            'Testa se o quaternion calibrado para a OAK-D prejudica a webcam.'
                        ))
    parser.add_argument('--proc_fps', type=int, default=8,
                        help=(
                            '[H3] Limita o processamento do modelo a N fps (padrão: 8, '
                            'igual à taxa usada no treino). Use 0 para sem limite.'
                        ))
    parser.add_argument('--replay', type=str, default=None,
                        help=(
                            '[H5] Caminho para um arquivo .pkl gravado com a OAK-D. '
                            'Reproduz os landmarks reais sem câmera para verificar se o '
                            'modelo funciona com dados de treino.'
                        ))

    args = parser.parse_args()

    if args.replay:
        run_replay(args)
    else:
        run_live(args)
