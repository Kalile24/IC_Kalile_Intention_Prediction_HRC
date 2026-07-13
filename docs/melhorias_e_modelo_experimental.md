# Melhorias ao Algoritmo e Modelo Experimental Sem OAK-D

## Parte 1 — Melhorias ao Algoritmo Existente

---

### 1.1 Predição de Intenção (`traj_intention/`)

#### Problema: Janela temporal fixa e sem memória contextual

O modelo recebe sempre exatamente `seq_len=5` frames, tratando cada predição como independente. Isso ignora o contexto de longo prazo do que o humano já fez.

**Melhoria:** Adicionar o estado atual da `PlanGraph` (estágio, histórico de ações, contagem de tubos) como vetor de contexto concatenado ao input do modelo. Isso permite que o modelo saiba, por exemplo, que se o humano já pegou 3 conectores, a próxima ação provavelmente é o 4º ou parafusos.

```python
# Exemplo de input enriquecido
context = [stage_one_hot, tube_count_norm, screw_count_norm, wheels_count_norm]
input = torch.cat([trajectory_features, context_vector], dim=-1)
```

---

#### Problema: Normalização min/max instável por batch

Em `run.py:297-298`, a normalização é feita com `min/max` do batch de 5 frames:
```python
poses_norm = 2*(poses-poses.min())/(poses.max()-poses.min())
```
Um único frame com movimento brusco distorce a escala de todos os outros frames do batch.

**Melhoria:** Calcular média e desvio padrão offline no dataset de treino e usar normalização z-score com valores fixos:
```python
POSE_MEAN = np.load('traj_intention/stats/mean.npy')
POSE_STD  = np.load('traj_intention/stats/std.npy')
poses_norm = (poses - POSE_MEAN) / (POSE_STD + 1e-6)
```

---

#### Problema: Detecção OOD por entropia com limiares fixos

Os thresholds `0.4` e `0.5` em `predict.py:85-87` foram ajustados manualmente e não generalizam bem para variações de usuário ou iluminação.

**Melhoria — Temperature Scaling:** Calibrar uma temperatura `T` sobre o conjunto de validação para que a distribuição de probabilidade do modelo seja bem-calibrada. É um pós-processamento de uma linha:
```python
calibrated_logits = pred_intention / T  # T otimizado via NLL no val set
```

**Melhoria — Ensemble simples:** Treinar 3 modelos com seeds diferentes e usar a variância das predições como estimativa de incerteza — mais robusto que entropia de um único modelo.

---

#### Problema: Arquitetura DLinear não captura interações entre joints

O modelo trata cada coordenada (x, y, z de cada joint) de forma independente na decomposição sazonal/tendência. Movimentos correlacionados entre pulso e cotovelo não são modelados.

**Melhoria — Graph Convolutional Network (GCN):** Representar o esqueleto como grafo onde joints conectados anatomicamente têm arestas. Uma camada GCN antes do DLinear captura relações espaciais entre joints:
```
Input skeleton → GCN (adjacency = corpo humano) → features por joint → DLinear temporal
```
Bibliotecas: `torch_geometric` ou implementação manual com matriz de adjacência fixa.

---

#### Problema: `intention_queue` com votação naïve

A janela de `send_window=3` frames consecutivos idênticos ignora a estrutura temporal das transições. Uma intenção pode oscilar entre `get_screws` e `no_action` sem nunca acumular 3 consecutivos mesmo sendo a correta.

**Melhoria — Bayesian smoothing:** Manter um vetor de probabilidade acumulado e atualizar com produto de probabilidades:
```python
# Ao invés de contar consecutivos:
accumulated_probs *= softmax(new_logits)
accumulated_probs /= accumulated_probs.sum()
if accumulated_probs.max() > threshold:
    send_intention(argmax(accumulated_probs))
    accumulated_probs = uniform_prior()
```

---

### 1.2 Controle Hierárquico (`controller/`)

#### Problema: Waypoints hardcoded por posição física absoluta

Todos os waypoints em `receiver.py:generate_waypoints()` são coordenadas absolutas calibradas para a posição exata da bancada do experimento original. Qualquer reconfiguração quebra o sistema.

**Melhoria — Calibração automática de workspace:** Numa etapa de setup, o robô move-se para posições de referência (marcadores ArUco na bancada) e aprende os offsets. Os waypoints passam a ser relativos a âncoras calibradas:
```python
PICK_CONNECTORS = BASE_ANCHOR + offset_connectors  # offset aprendido no setup
```

---

#### Problema: `decide_send_action` é um mapeamento fixo codificado em `if/elif`

A lógica de planejamento está misturada com a lógica de execução, tornando difícil alterar a tarefa de montagem ou adicionar novas tarefas.

**Melhoria — Representação explícita como grafo de tarefas:**
```python
TASK_GRAPH = {
    "start": {"get_connectors": "bottom_layer"},
    "bottom_layer": {"get_screws": "spin_bottom", "get_connectors": "bottom_layer"},
    "spin_bottom": {"get_connectors": "top_layer", ...},
    ...
}
```
O `PlanGraph` vira um walker nesse grafo, separando estrutura de tarefa da lógica de execução.

---

#### Problema: Fusão multimodal simplista (voz sobrescreve visão)

Atualmente os comandos de voz simplesmente sobrescrevem as intenções visuais, sem ponderação. Isso pode causar conflitos quando o comando de voz chega atrasado ou é ambíguo.

**Melhoria — Fusão com confiança:** Combinar intenção visual (com sua probabilidade) e comando de voz (com seu score de reconhecimento do DeepSpeech) num vetor de decisão unificado antes de acionar o robô.

---

## Parte 2 — Modelo Experimental Sem a Câmera OAK-D

A câmera OAK-D Lite é usada para três coisas: detecção de pessoa, estimativa de pose, e coordenadas de profundidade 3D. Cada uma pode ser substituída.

---

### 2.1 Opções de Substituição de Hardware/Software

| Função original | Substituto sem OAK-D |
|----------------|----------------------|
| Detecção de pessoa (MobileNet spatial) | Webcam USB + YOLOv8 |
| Estimativa de pose 2D | MediaPipe Pose via CPU (sem DepthAI) |
| Coordenada Z (profundidade) | Estimativa monocular com MiDaS ou DPT |
| Coordenadas 3D do corpo | Pose lifting com VideoPose3D ou MotionBERT |

---

### 2.2 Arquitetura do Pipeline Experimental

```
Webcam USB (qualquer câmera)
    ↓
MediaPipe Pose (CPU) → 33 landmarks 2D normalizados
    ↓
MiDaS v3 (monocular depth) → mapa de profundidade
    ↓
Lifting 3D: usar a profundidade do mapa na posição do joint → pseudo-3D
    ↓
Coordenadas 3D aproximadas → normalização z-score (com stats do treino)
    ↓
IntentionPredictor (DLinear) → intenção prevista
    ↓
[Sem robô] imprimir intenção / simular via ROS fake publisher
```

---

### 2.3 Implementação Passo a Passo

#### Passo 1 — Instalar dependências leves

```bash
conda activate hrc
pip install mediapipe
pip install timm  # para MiDaS
pip install opencv-python
```

#### Passo 2 — Substituir `BlazeposeDepthaiModule` por MediaPipe

Criar `depthai_blazepose/mediapipe_fallback.py`:

```python
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

class MediaPipePoseModule:
    """Substituto do BlazeposeDepthaiModule para uso sem OAK-D."""

    UPPER_BODY_INDICES = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0]
    # corresponde aos 15 joints usados no sistema original

    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def inference(self, frame):
        """Retorna array (15, 3) com coordenadas normalizadas."""
        rgb = frame[:, :, ::-1]
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        upper = np.array([
            [landmarks[i].x, landmarks[i].y, landmarks[i].z]
            for i in self.UPPER_BODY_INDICES
        ], dtype=np.float32)
        return upper
```

#### Passo 3 — Substituir detecção espacial por detecção 2D + depth monocular

```python
import torch
import cv2

# Carregar MiDaS para profundidade monocular
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.eval()
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform

def get_depth_at_point(frame, x, y):
    """Estima profundidade relativa no pixel (x, y)."""
    input_batch = midas_transforms(frame).unsqueeze(0)
    with torch.no_grad():
        depth_map = midas(input_batch)
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=frame.shape[:2],
        mode='bicubic',
        align_corners=False
    ).squeeze().numpy()
    return depth_map[y, x]

def estimate_3d_from_2d_depth(landmarks_2d, depth_map, img_w, img_h):
    """Combina landmarks 2D com mapa de profundidade para pseudo-3D."""
    landmarks_3d = []
    for lm in landmarks_2d:
        px = int(lm[0] * img_w)
        py = int(lm[1] * img_h)
        px = np.clip(px, 0, img_w - 1)
        py = np.clip(py, 0, img_h - 1)
        z_rel = depth_map[py, px]
        landmarks_3d.append([lm[0], lm[1], z_rel])
    return np.array(landmarks_3d, dtype=np.float32)
```

#### Passo 4 — Script principal experimental (`run_experimental.py`)

```python
import cv2
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'traj_intention'))
sys.path.append(str(Path(__file__).parent / 'depthai_blazepose'))

from predict import IntentionPredictor
from Dataset import INTENTION_LIST
from mediapipe_fallback import MediaPipePoseModule

# Inicializar módulos
pose_module = MediaPipePoseModule()
predictor = IntentionPredictor(model_type='final_intention')

# Parâmetros
SEQ_LEN = 5
traj_queue = []

cap = cv2.VideoCapture(0)  # webcam USB

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_h, img_w = frame.shape[:2]

    # Inferência de pose
    landmarks = pose_module.inference(frame)

    if landmarks is not None:
        # Acumular trajetória
        if len(traj_queue) >= SEQ_LEN:
            traj_queue.pop(0)
        traj_queue.append(landmarks)

        # Predição de intenção
        if len(traj_queue) == SEQ_LEN:
            poses = np.array(traj_queue)
            poses_norm = (poses - poses.mean()) / (poses.std() + 1e-6)
            inputs = torch.tensor(
                poses_norm.reshape(1, SEQ_LEN, -1)
            ).float()

            _, pred_intention = predictor.predict(inputs, restrict='ood')
            intention_idx = pred_intention[0].item()
            intention_name = [k for k, v in INTENTION_LIST.items() if v == intention_idx][0]

            cv2.putText(frame, f'Intention: {intention_name}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('HRC Experimental', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### 2.4 Limitações do Modelo Experimental e Como Mitigá-las

| Limitação | Causa | Mitigação |
|-----------|-------|-----------|
| Profundidade imprecisa | MiDaS produz profundidade relativa, não métrica | Normalizar por altura estimada do ombro como referência de escala |
| Sem verificação de working área | Sem coordenadas absolutas em mm | Usar zona 2D na imagem como proxy da área de trabalho |
| Maior latência | MediaPipe + MiDaS juntos são mais lentos que o pipeline on-device da OAK-D | Usar MiDaS_small ou desativar depth e usar apenas pose 2D com `restrict='ood'` |
| Drift de coordenadas entre sessões | Câmera montada em posição diferente a cada sessão | Calibrar com ArUco marker no início de cada sessão |
| Modelo treinado com dados OAK-D | Os dados de treino foram coletados com a câmera original | Recoletar pelo menos 20% dos dados com a webcam para fine-tuning |

---

### 2.5 Fine-tuning do Modelo com Novos Dados (Webcam)

Coletar dados com `run_experimental.py` em modo de gravação:

```bash
# Gravar trajetórias com webcam e anotar manualmente
python run_experimental.py --record --task webcam001

# Fine-tuning a partir do checkpoint existente
python traj_intention/train.py \
    --checkpoint traj_intention/checkpoints/seq5_pred5_epoch40_whole_pkl_final_intention_nomaskFalse.pth \
    --data_dir human_traj/webcam \
    --epochs 10 \
    --lr 1e-4
```

O fine-tuning de 10 épocas sobre dados novos geralmente é suficiente para adaptar o modelo a uma câmera diferente sem perder o conhecimento pré-treinado.

---

## Resumo das Prioridades

| Prioridade | Melhoria | Esforço | Impacto |
|-----------|----------|---------|---------|
| Alta | Normalização z-score com stats offline | Baixo | Alto — estabiliza predições |
| Alta | Pipeline experimental com MediaPipe (sem OAK-D) | Médio | Alto — desbloqueia desenvolvimento |
| Média | Temperature scaling para OOD | Baixo | Médio — calibração melhorada |
| Média | Estado de PlanGraph como contexto do modelo | Médio | Alto — predições cientes da tarefa |
| Média | Bayesian smoothing na intention queue | Baixo | Médio — menos oscilações |
| Baixa | GCN para relações entre joints | Alto | Médio — ganho marginal vs. complexidade |
| Baixa | Grafo de tarefas explícito | Alto | Alto — mas só vale se houver novas tarefas |
