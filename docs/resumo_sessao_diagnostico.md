# Resumo da Sessão — Diagnóstico do Modelo de Intenção sem OAK-D

## Contexto

O sistema HRC original usa uma câmera **OAK-D Lite** (estéreo + VPU) com **BlazePose** rodando na VPU para estimar pose 3D. O modelo **DLinear** (`traj_intention/`) foi treinado com esses dados. Ao substituir por webcam comum + MediaPipe (CPU), o modelo não prediz bem.

---

## Hipóteses de Falha Levantadas

### H1 — Dados fora da distribuição (OOD)
A OAK-D fornece profundidade Z **real** via câmera estéreo. O MediaPipe com webcam 2D fornece Z **estimado** por rede neural — distribuição diferente do treino. A métrica de diagnóstico é a **entropia de Shannon** da saída softmax:
- `< 0.4` → modelo confiante (entrada próxima do treino)
- `0.4–0.8` → incerto
- `> 0.8` → OOD (distribuição uniforme = modelo não reconhece a entrada)

### H2 — Quaternion de rotação incorreto
O quaternion `[0.14070565, -0.15007018, -0.7552408, 0.62232804]` foi calibrado para a posição exata da OAK-D no experimento. Com webcam do notebook em posição diferente, a rotação câmera→mundo deforma a trajetória.

### H3 — Taxa de frames incompatível
O treino usou ~8 fps (`pose_fps = 8` em `run.py`). A webcam processa 20–30 fps. A janela `seq_len=5` cobre um intervalo de tempo menor — movimentos parecem "mais lentos" para o modelo.

### H4 — Restrição OOD suprimindo predições
O modo `--restrict ood` força `no_action` quando `entropia > 0.4`. Se os dados da webcam têm entropia alta, **todas** as predições são suprimidas — parece que o modelo não funciona, mas na verdade está detectando OOD.

### H5 — Validação com dados reais (confirmatória)
Reproduzir um `.pkl` gravado com a OAK-D para confirmar que o modelo funciona com dados da distribuição de treino. Se funcionar no replay e falhar na webcam, o problema está nos dados de entrada.

---

## Alterações no `run_webcam.py`

Quatro novos argumentos adicionados, com fidelidade total ao pré-processamento original:

| Argumento | Hipótese | Comportamento |
|-----------|----------|---------------|
| `--diag` | H1, H4 | Calcula e exibe entropia, probabilidades por classe e stats do eixo Z. Painel visual sobreposto no vídeo. |
| `--no_qrot` | H2 | Usa quaternion identidade `[1,0,0,0]` em vez do quaternion calibrado para a OAK-D. |
| `--proc_fps N` | H3 | Limita o modelo a rodar N vezes por segundo (padrão `8`, igual ao treino). O MediaPipe continua rodando em todo frame. |
| `--replay PKL` | H5 | Carrega um `.pkl` da OAK-D e itera pelos landmarks reais sem câmera. |

No branch `webcam-runtime-optimizations`, o runtime ao vivo também foi endurecido para
uso com webcam USB:

| Recurso | Comportamento |
|---------|---------------|
| `--camera auto` | Procura fontes `/dev/video*` e índices OpenCV comuns até encontrar uma câmera que entregue frames. |
| `--capture_backend v4l2` | Usa V4L2 por padrão no Linux para reduzir inconsistências de abertura da câmera. |
| `--cam_fourcc MJPG` | Solicita MJPEG para reduzir banda USB em webcams compatíveis. |
| `--camera_buffer 1` | Mantém o buffer pequeno para reduzir latência e frames antigos. |
| Normalização BGR | Converte frames cinza/BGRA para BGR antes do MediaPipe, desenho do esqueleto e HUD. |

O pré-processamento permanece **idêntico** ao `run.py` e `Dataset.py`:
```python
poses_norm  = 2 * (poses - poses.min()) / (poses.max() - poses.min() + 1e-8)
poses_world = camera_to_world(poses_norm, quat)
poses_world[:, :, 2] -= poses_world[:, :, 2].min()
```

---

## Roteiro de Testes

```bash
# H5 — Validar com dados reais (rodar primeiro)
python run_webcam.py --diag --restrict ood \
    --replay human_traj/<tarefa>/<tarefa>.pkl

# H1 — Medir entropia ao vivo
python run_webcam.py --show --task diag001 --diag --restrict ood

# H4 — Remover filtro OOD para ver predições brutas
python run_webcam.py --show --task diag001 --diag --restrict no

# H2 — Comparar com e sem quaternion
python run_webcam.py --show --task diag001 --diag --no_qrot --restrict no
python run_webcam.py --show --task diag001 --diag --restrict no

# H3 — Comparar FPS de processamento
python run_webcam.py --show --task diag001 --diag --proc_fps 0 --restrict no
python run_webcam.py --show --task diag001 --diag --proc_fps 8 --restrict no
```

---

## Interpretação dos Resultados

| Observação | Hipótese confirmada | Próximo passo |
|------------|---------------------|---------------|
| Replay funciona, webcam não | H1, H2 ou H3 | Continuar com H1→H4→H2→H3 |
| Entropia sempre > 0.8 ao vivo | H1 | O Z do MediaPipe é muito diferente do treino |
| Entropia cai com `--no_qrot` | H2 | Recalibrar o quaternion para a posição da webcam |
| `motion` muito baixo com `--proc_fps 0` | H3 | Usar `--proc_fps 8` como padrão |
| Com `--restrict no` aparecem predições, com `ood` não | H4 | Ajustar limiar de entropia ou desativar filtro |
| Replay também falha | Bug no pipeline | Revisar Dataset.py e slicing de landmarks |

---

## Arquivos Relevantes

| Arquivo | Papel |
|---------|-------|
| `run_webcam.py` | Pipeline principal com diagnóstico (versão modificada) |
| `run.py` | Pipeline original com OAK-D (referência) |
| `traj_intention/predict.py` | `IntentionPredictor` — aplicação da restrição OOD |
| `traj_intention/Dataset.py` | Pré-processamento de treino (referência de fidelidade) |
| `depthai_blazepose/mediapipe_fallback.py` | Substituto do BlazePose para webcam |
| `docs/DIAGNOSTICO_WEBCAM.md` | Guia detalhado de cada teste |
