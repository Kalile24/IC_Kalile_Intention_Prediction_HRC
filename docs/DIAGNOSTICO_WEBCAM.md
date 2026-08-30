# Diagnóstico do Modelo de Predição de Intenção — Webcam vs OAK-D

Este guia explica as hipóteses de falha levantadas para o modelo de predição de intenção quando executado com webcam (sem OAK-D Lite), e os passos para testar cada uma delas usando o `run_webcam.py`.

---

## Contexto do problema

O modelo DLinear foi treinado com dados coletados pela câmera **OAK-D Lite**, que é uma câmera estéreo com sensor de profundidade real. Ao substituir por uma webcam comum do notebook, o modelo apresenta predições ruins. As hipóteses investigadas são:

| ID | Hipótese | Impacto esperado |
|----|----------|-----------------|
| H1 | Entrada está fora da distribuição de treino (OOD) | Alto |
| H2 | Quaternion de rotação câmera→mundo está errado para a webcam | Alto |
| H3 | Taxa de frames da webcam é diferente do treino (8fps) | Médio |
| H4 | A restrição OOD está suprimindo predições válidas | Médio |
| H5 | O modelo funciona com dados reais, mas não com dados da webcam | Confirmatório |

---

## Pré-requisitos

```bash
conda activate hrc
# Confirmar que as dependências estão instaladas:
python -c "import cv2, mediapipe, torch; print('OK')"
```

---

## Configuração de captura no branch `webcam-runtime-optimizations`

O branch atual mantém o diagnóstico original e adiciona ajustes de runtime para tornar a
webcam mais estável no Linux:

| Opção | Padrão | Quando alterar |
|-------|--------|----------------|
| `--camera` | `auto` | Use `/dev/video0`, `/dev/video2` ou um índice se quiser fixar a fonte manualmente |
| `--capture_backend` | `v4l2` | Use `any` se o backend V4L2 não abrir sua câmera |
| `--cam_width` / `--cam_height` | `1280x720` | Use `0` para aceitar o padrão do driver |
| `--cam_fps` | `30` | Use `0` para aceitar o padrão do driver |
| `--cam_fourcc` | `MJPG` | Teste `H264` ou `YUYV` se a câmera não entregar frames em MJPG |
| `--camera_buffer` | `1` | Aumente apenas se houver instabilidade de captura |
| `--save_frames` | off | Salva cada frame com esqueleto como PNG. Desativado por padrão para evitar travamentos no vídeo ao vivo |

Comando recomendado para iniciar o teste:

```bash
python run_webcam.py --show --task diag001 --diag --proc_fps 8 --restrict ood
```

Se a câmera não abrir, o script lista os dispositivos `/dev/video*` encontrados. Nesse
caso, force uma fonte específica:

```bash
python run_webcam.py --show --task diag001 --camera /dev/video0 --diag
```

O pipeline também normaliza frames para BGR antes do MediaPipe e do HUD. Isso evita falhas
quando o driver entrega frames em escala de cinza ou BGRA.

---

## H1 — A entrada está fora da distribuição de treino?

**Diagnóstico:** O modelo foi treinado com profundidade Z real (câmera estéreo). A webcam fornece Z estimado por rede neural, que tem distribuição diferente. Quando a entrada está muito fora do treino, a distribuição de probabilidades do modelo fica uniforme — alta entropia.

**Interpretação da entropia:**

| Entropia | Significado |
|----------|-------------|
| < 0.4    | Modelo confiante — entrada próxima da distribuição de treino |
| 0.4–0.8  | Incerteza moderada — entrada parcialmente OOD |
| > 0.8    | Modelo incerto — entrada muito fora do treino |
| ≈ 1.386  | Entropia máxima (4 classes uniformes) — modelo não sabe nada |

**Comando:**

```bash
python run_webcam.py --show --task diag_h1 --diag --restrict ood
```

**O que observar no terminal:**
```
[frame 0012] intenção=no_action         entropia=1.254  motion=0.0231  qrot=ON  |  no_action=0.27  get_connectors=0.24  get_screws=0.26  get_wheels=0.23
```

- **Entropia consistentemente alta (>0.8) mesmo em movimento**: confirma H1. Os dados da webcam são OOD para o modelo.
- **Entropia baixa (<0.4) em movimentos**: o modelo está conseguindo classificar. Problema pode ser outro.

**O que observar no vídeo:**
- Painel superior direito com barra de probabilidade por classe.
- Se todas as barras forem do mesmo tamanho → entropia alta → OOD.

---

## H2 — O quaternion de rotação está errado?

**Diagnóstico:** O quaternion `[0.14070565, -0.15007018, -0.7552408, 0.62232804]` foi calibrado para a posição específica da OAK-D no experimento. Com a webcam do notebook em ângulo diferente, essa rotação deforma a trajetória de um jeito que o modelo nunca viu.

**Teste A — Com quaternion desativado (identidade):**

```bash
python run_webcam.py --show --task diag_h2_noquat --diag --no_qrot --restrict no
```

**Teste B — Com quaternion original:**

```bash
python run_webcam.py --show --task diag_h2_quat --diag --restrict no
```

**Compare a entropia nos dois casos.** Se com `--no_qrot` a entropia for menor (modelo mais confiante), o quaternion está prejudicando.

**Interpretação:**

| Resultado | Conclusão |
|-----------|-----------|
| `--no_qrot` tem entropia menor | Quaternion errado para esta câmera — **considere recalibrá-lo** |
| Entropia igual nos dois | O quaternion não é o principal problema |

**Nota:** Para recalibrar o quaternion para sua webcam, você precisa fazer uma calibração de câmera e estimar a rotação câmera→mundo com a câmera na posição de uso.

---

## H3 — A taxa de frames está errada?

**Diagnóstico:** O modelo foi treinado com dados gravados a ~8 fps (ver `pose_fps = 8` em `run.py`). Se a webcam processa a 30fps, a janela de 5 frames cobre apenas ~167ms em vez de ~625ms. Os movimentos parecem "mais lentos" para o modelo — deslocamento por frame muito menor.

**Teste A — Sem limite de FPS (webcam a velocidade máxima):**

```bash
python run_webcam.py --show --task diag_h3_fps0 --diag --proc_fps 0 --restrict no
```

**Teste B — Limitado a 8fps (igual ao treino):**

```bash
python run_webcam.py --show --task diag_h3_fps8 --diag --proc_fps 8 --restrict no
```

**Teste C — Limitar a 15fps (intermediário):**

```bash
python run_webcam.py --show --task diag_h3_fps15 --diag --proc_fps 15 --restrict no
```

**O que observar:**
- Valor `motion` no terminal: com `--proc_fps 0` o `motion` será muito menor (menos deslocamento por frame).
- Com `--proc_fps 8`, o `motion` deve ser próximo ao limiar `STILLNESS_THRESHOLD=0.015`.
- Se a entropia cair com `--proc_fps 8` em relação a `--proc_fps 0`, confirma H3.

**Acompanhe no terminal:**
```
proc_fps: X.X   (FPS real de processamento exibido no vídeo)
motion: 0.0089  (muito baixo → parece estático para o modelo)
motion: 0.0231  (razoável → movimento detectável)
```

---

## H4 — A restrição OOD está suprimindo predições válidas?

**Diagnóstico:** O parâmetro `--restrict ood` (padrão) aplica um filtro de entropia:
- Se entropia > 0.4 → força `no_action` para classes 0-2
- Se entropia > 0.5 → força `no_action` para classe 3 (get_wheels)

Se os dados da webcam tiverem alta entropia mesmo em movimentos corretos, o filtro suprime todas as predições e o resultado sempre parece `no_action`.

**Teste sem restrição:**

```bash
python run_webcam.py --show --task diag_h4 --diag --restrict no
```

**Compare com restrição ood:**

```bash
python run_webcam.py --show --task diag_h4_ood --diag --restrict ood
```

**Interpretação:**

| Observação | Conclusão |
|------------|-----------|
| Com `no` detecta intenções, com `ood` sempre retorna `no_action` | A restrição OOD é muito agressiva para dados de webcam — a entropia é alta o suficiente para sempre suprimir |
| Ambos retornam `no_action` | O problema é anterior — H1 ou H2 |
| Com `no` detecta intenções erradas | O modelo está classificando, mas nas classes erradas — problema de domínio (H1/H2) |

---

## H5 — O modelo funciona com dados de treino reais?

**Diagnóstico:** Reproduz um arquivo `.pkl` gravado com a OAK-D Lite durante o experimento original. Confirma se o modelo funciona corretamente com dados dentro da distribuição de treino.

**Pré-requisito:** Você precisa de um arquivo `.pkl` gravado com `run.py` na OAK-D.

```bash
# Estrutura esperada: human_traj/<nome_tarefa>/<nome_tarefa>.pkl
# Exemplo:
ls human_traj/
```

**Comando:**

```bash
python run_webcam.py --diag --restrict ood \
    --replay human_traj/<nome_tarefa>/<nome_tarefa>.pkl
```

Substitua `<nome_tarefa>` pelo nome de uma tarefa existente, por exemplo `abc_connectors/abc_connectors001.pkl`.

**Exemplo de saída esperada (dados OAK-D):**
```
[0012] intenção=get_connectors     entropia=0.187  motion=0.0412  |  no_action=0.05  get_connectors=0.82  get_screws=0.08  get_wheels=0.05
```

**Interpretação:**

| Resultado | Conclusão |
|-----------|-----------|
| Replay detecta intenções com entropia baixa | Modelo funciona — problema está nos dados da webcam (H1, H2 ou H3) |
| Replay também falha | Problema no próprio modelo ou no pipeline de pré-processamento |

---

## Roteiro de diagnóstico recomendado

Execute os testes nesta ordem para isolar a causa:

```
1. [H5] Replay com pkl da OAK-D — confirma se o modelo funciona
         ↓ funciona
2. [H1] run com --diag --restrict ood — observa entropia da webcam
         ↓ entropia alta (>0.8)
3. [H4] run com --restrict no — verifica se sem filtro aparecem predições
         ↓ aparecem predições (mesmo que erradas)
4. [H2] run com --no_qrot --restrict no — compara entropia com/sem quaternion
         ↓ entropia cai com --no_qrot
5. [H3] run com --proc_fps 0 vs --proc_fps 8 — compara motion e entropia
```

---

## Guia rápido de comandos

```bash
# Baseline (comportamento original)
python run_webcam.py --show --task test001 --restrict ood

# Diagnóstico completo ao vivo
python run_webcam.py --show --task diag001 --diag --proc_fps 8 --restrict ood

# H1+H4: Sem restrição, ver predições brutas
python run_webcam.py --show --task diag001 --diag --restrict no

# H2: Sem quaternion
python run_webcam.py --show --task diag001 --diag --no_qrot --restrict no

# H2: Com quaternion (padrão)
python run_webcam.py --show --task diag001 --diag --restrict no

# H3: FPS ilimitado (não limitar processamento)
python run_webcam.py --show --task diag001 --diag --proc_fps 0 --restrict no

# H3: FPS igual ao treino
python run_webcam.py --show --task diag001 --diag --proc_fps 8 --restrict no

# H5: Replay de pkl da OAK-D
python run_webcam.py --diag --restrict ood \
    --replay human_traj/abc_connectors/abc_connectors001.pkl
```

---

## Explicação do código run_webcam.py

### Estrutura geral

```
run_webcam.py
├── camera_to_world()      — rotação câmera→mundo (suporta quaternion customizado)
├── compute_diag()         — calcula entropia e probabilidades sem restrição
├── draw_diag_panel()      — desenha painel de diagnóstico no frame
├── run_replay()           — reproduz arquivo .pkl sem câmera
└── run_live()             — pipeline principal com webcam
```

### Argumento `--diag`

Ativa o modo de diagnóstico. A cada predição, chama `compute_diag()` que acessa o modelo diretamente (`predictor.model(inputs)`) sem passar pela lógica de restrição de `predict.py`. Isso permite ver as probabilidades brutas antes de qualquer filtro.

O painel no vídeo mostra:
- **Barra de probabilidade** por classe
- **Entropia**: verde (<0.4) → confiante; laranja (0.4–0.8) → incerto; vermelho (>0.8) → OOD
- **qrot ON/OFF**: indica se o quaternion está ativo
- **Z raw stats**: min, max e desvio-padrão do eixo Z dos 5 frames

### Argumento `--no_qrot`

Passa `_IDENTITY_Q = [1, 0, 0, 0]` para `camera_to_world()`. O quaternion identidade não rotaciona nada — os landmarks ficam no sistema de coordenadas da câmera. Útil para isolar o efeito da rotação.

### Argumento `--proc_fps`

Controla com que frequência o modelo é executado, independentemente do FPS da câmera. O MediaPipe ainda roda em todo frame para manter o tracking suave, mas a janela `traj_queue` e o modelo DLinear só são atualizados a cada `1/proc_fps` segundos.

Isso replica a cadência do treino (8fps em `run.py`) sem perder a qualidade de tracking do MediaPipe.

### Argumento `--replay`

Carrega um `.pkl` salvo pelo `run.py` original e itera pelos objetos `body` gravados. Para cada body:
1. Extrai landmarks (converte de 33→15 se necessário)
2. Aplica o mesmo pré-processamento (min-max + camera_to_world)
3. Roda o modelo e imprime o resultado

Não usa câmera — serve exclusivamente para validar o modelo com dados conhecidos.

### Fidelidade com o algoritmo original

O pré-processamento em `run_live()` é **idêntico** ao `run.py` e ao `Dataset.py`:

```python
# run.py (original):
poses_norm = 2*(poses-poses.min())/(poses.max()-poses.min())
poses_world = camera_to_world(poses_norm)
poses_world[:, :, 2] -= np.min(poses_world[:, :, 2])

# run_webcam.py (esta versão):
poses_norm  = 2 * (poses - poses.min()) / (poses.max() - poses.min() + 1e-8)
poses_world = camera_to_world(poses_norm, quat)   # quat=None usa o mesmo quaternion
poses_world[:, :, 2] -= poses_world[:, :, 2].min()
```

A única diferença intencional é o `+ 1e-8` para evitar divisão por zero quando o corpo está parado.
