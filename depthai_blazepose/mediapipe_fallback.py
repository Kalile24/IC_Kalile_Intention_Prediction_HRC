"""
Substituto do BlazeposeDepthaiModule para uso sem câmera OAK-D.
Usa MediaPipe Pose via CPU com qualquer webcam ou câmera USB.

Interface compatível com o pipeline original de run.py:
  body.landmarks       → np.array (15, 3) coords normalizadas (imagem)
  body.landmarks_world → np.array (15+, 3) coords mundo (metros, origin no hip)
  body.xyz             → np.array (3,) posição estimada da pessoa (approx)
  body.score           → float, confiança da detecção
"""

import numpy as np
import cv2
import mediapipe as mp

# Índices MediaPipe que correspondem aos 15 joints do sistema original
# Original: body.landmarks[11:25] + body.landmarks[0:1]
# MediaPipe usa os mesmos índices que BlazePose
UPPER_BODY_MP_INDICES = list(range(11, 25)) + [0]  # 14 joints + nariz = 15

mp_pose = mp.solutions.pose

# Visualização compacta, próxima ao esqueleto usado no paper:
# torso + braços, sem face, mãos completas ou caixa de coordenadas.
DISPLAY_JOINTS = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
}
TORSO_CONNECTIONS = [
    ('left_shoulder', 'right_shoulder'),
    ('right_shoulder', 'right_hip'),
    ('right_hip', 'left_hip'),
    ('left_hip', 'left_shoulder'),
]
ARM_CONNECTIONS = [
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
]


class FakeBody:
    """Imita o objeto 'body' retornado pelo BlazeposeDepthaiModule."""
    def __init__(self, landmarks, landmarks_world, score):
        self.landmarks = landmarks          # (15, 3) coords normalizadas
        self.landmarks_world = landmarks_world  # (15, 3) coords mundo em metros
        self.xyz = np.zeros(3, dtype=np.float32)  # posição absoluta desconhecida
        self.score = score


class MediaPipePoseModule:
    """
    Substituto do BlazeposeDepthaiModule que roda com webcam normal.

    Uso:
        module = MediaPipePoseModule()
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            body = module.inference(frame)
            if body:
                upperbody = body.landmarks  # (15, 3)
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 model_complexity=1, smoothing=True):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smoothing,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._last_pose_landmarks = None

    def inference(self, frame_bgr):
        """
        Processa um frame BGR e retorna um FakeBody ou None.

        Parâmetros:
            frame_bgr: np.array (H, W, 3) em formato BGR (saída do cv2)

        Retorna:
            FakeBody com landmarks de 15 joints do corpo superior, ou None
            se nenhuma pessoa for detectada.
        """
        if frame_bgr.ndim == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        self._last_pose_landmarks = results.pose_landmarks

        if not results.pose_landmarks:
            return None

        lms = results.pose_landmarks.landmark
        lms_world = results.pose_world_landmarks.landmark

        # Extrai os 15 joints do corpo superior (coords normalizadas 0-1)
        landmarks = np.array(
            [[lms[i].x, lms[i].y, lms[i].z] for i in UPPER_BODY_MP_INDICES],
            dtype=np.float32
        )

        # Extrai os mesmos 15 joints em coordenadas mundo (metros, origin no quadril)
        landmarks_world = np.array(
            [[lms_world[i].x, lms_world[i].y, lms_world[i].z] for i in UPPER_BODY_MP_INDICES],
            dtype=np.float32
        )

        # Score de visibilidade médio dos joints principais (ombros + pulsos)
        key_indices_in_15 = [0, 1, 4, 5]  # left_shoulder, right_shoulder, left_wrist, right_wrist
        score = float(np.mean([lms[UPPER_BODY_MP_INDICES[i]].visibility for i in key_indices_in_15]))

        return FakeBody(landmarks, landmarks_world, score)

    def draw(self, frame_bgr, body):
        """Desenha o esqueleto no frame (compatível com BlazeposeRenderer.draw)."""
        if body is None or self._last_pose_landmarks is None:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        lms = self._last_pose_landmarks.landmark

        def point(name, min_visibility=0.45):
            lm = lms[DISPLAY_JOINTS[name]]
            if lm.visibility < min_visibility:
                return None
            return int(lm.x * w), int(lm.y * h)

        points = {name: point(name) for name in DISPLAY_JOINTS}

        def draw_connection(start, end, color, thickness=3):
            p1 = points.get(start)
            p2 = points.get(end)
            if p1 is not None and p2 is not None:
                cv2.line(frame_bgr, p1, p2, color, thickness, cv2.LINE_AA)

        for start, end in TORSO_CONNECTIONS:
            draw_connection(start, end, (255, 80, 0), 3)  # azul
        for start, end in ARM_CONNECTIONS:
            draw_connection(start, end, (0, 0, 255), 3)   # vermelho

        for p in points.values():
            if p is not None:
                cv2.circle(frame_bgr, p, 5, (0, 220, 0), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, p, 7, (0, 80, 0), 1, cv2.LINE_AA)

        return frame_bgr

    def close(self):
        self.pose.close()
