# Módulo de Predição de Plano Hierárquico

## Visão Geral

O sistema é dividido em **duas camadas hierárquicas** que se comunicam via ROS:

---

## Camada 1 — Predição de Intenção (`traj_intention/`)

### `Model_FinalIntention` (DLinear.py:39)

Modelo de deep learning que recebe a trajetória do esqueleto humano e produz a intenção prevista.

**Arquitetura:**
```
Input: [Batch, seq_len=5 frames, channels=45 (15 joints × 3 coords)]
    ↓
series_decomp (kernel=3) → separa tendência e sazonalidade
    ↓
Traj_Seasonal + Traj_Trend → pred_traj [5 frames futuros]
    ↓
Intention_Predictor: Linear((seq+pred)*channels → 4 classes)
    ↓
Output: (pred_traj, logits das 4 intenções)
```

Existe também `Model_FinalTraj` (DLinear.py:96), variante que usa um `Intention_Vector` para condicionar a predição de trajetória na intenção — os dois modelos compartilham a mesma arquitetura DLinear base.

**Classes de intenção:**
| Índice | Nome |
|--------|------|
| 0 | `no_action` |
| 1 | `get_connectors` |
| 2 | `get_screws` |
| 3 | `get_wheels` |

---

### `IntentionPredictor` (predict.py:44)

Wrapper de inferência que:
1. Carrega o checkpoint correto baseado nos hiperparâmetros (`seq_len`, `pred_len`, `epochs`, `filter_type`)
2. Aplica restrições pós-inferência no método `predict()`:

| Modo (`restrict`) | Comportamento |
|-------------------|---------------|
| `ood` | Calcula entropia de Shannon sobre o softmax — se entropia > 0.4, retorna `no_action` (detecta situações fora da distribuição) |
| `working_area` | Verifica coordenadas do pulso direito/esquerdo no mundo real para confirmar se o humano está alcançando a área correta |
| `all` | Aplica ambas as restrições |
| `no` | Passa a predição diretamente sem filtros |

---

## Camada 2 — Controle Hierárquico de Tarefas (`controller/receiver.py`)

### `PlanGraph` (receiver.py:34)

Máquina de estados que representa o progresso da tarefa de montagem:

```python
stage: None | "bottom" | "four_tubes" | "top"
tube_count:   {"short": 0..8, "long": 0..4}
screw_count:  {"bottom": 0..4, "four_tubes": 0..4, "top": 0..4}
wheels_count: 0..4
stage_record: histórico de ações por estágio
stage_history: estágios já concluídos
```

Suporta retomada de estágio via flags `--stageI_done` e `--user_spin`.

---

### `Receiver` (receiver.py:57)

Orquestrador central que integra as duas modalidades (visão + voz):

```
ROS "chatter" topic
    ↓
receive_data()
    ├─ Se é intenção visual (get_connectors / get_screws / get_wheels)
    │      ↓ decide_send_action() → mapeia intenção para ação de robô
    │      ↓ execute_action() → publica waypoints no /kinova_demo/pose_cmd
    │
    └─ Se é comando de voz (stop / short / long / spin / rotate)
           ↓ modifica estado do PlanGraph (stage = "four_tubes", etc.)
           ↓ pode interromper ou redirecionar execute_action() em curso
```

---

## Comunicação entre Camadas

```
run.py (Camada de Percepção)
  │
  │  [OAK-D + BlazePose → 5 frames de esqueleto]
  │         ↓
  │  IntentionPredictor.predict()
  │         ↓
  │  intention_queue (janela de send_window=3 confirmações consecutivas)
  │         ↓
  │  send_intention_to_ros() → publica em "chatter" (ROS String)
  │
  ▼
controller/receiver.py (Camada de Controle)
  │
  │  receive_data() ← assina "chatter"
  │         ↓
  │  decide_send_action() + PlanGraph.stage
  │         ↓
  │  execute_action() → publica em /kinova_demo/pose_cmd (PoseStamped)
  │                   → publica em /siemens_demo/gripper_cmd (Float64MultiArray)
```

---

## Ponto-Chave da Hierarquia

A **hierarquia** opera em dois níveis distintos:

### 1. Percepção Hierárquica (`run.py`)
Detecção de pessoa → pose do esqueleto → intenção — cada nível só ativa o próximo se o anterior confirmar presença.

### 2. Filtragem Dupla de Intenção
Antes de enviar ao robô, a intenção passa por dois filtros em camadas diferentes:

- **Filtro interno** — `IntentionPredictor.predict()`: OOD + working_area em coordenadas normalizadas
- **Filtro externo** — `intention_sender()` em `run.py`: `outer_restrict` em coordenadas de mundo reais, após janela de confirmação de `send_window` frames consecutivos

Isso garante que **só intenções consistentes, fisicamente plausíveis e confirmadas por múltiplos frames** disparam ações no robô.
