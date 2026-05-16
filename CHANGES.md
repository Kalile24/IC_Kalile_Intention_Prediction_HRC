# Changes & Improvements — Intention Prediction Module

This document describes additions and improvements made on top of the original repository
[intelligent-control-lab/Robust-Hierarchial-Multimodal-HRC](https://github.com/intelligent-control-lab/Robust-Hierarchial-Multimodal-HRC),
scoped exclusively to the **intention prediction pipeline**.

---

## Motivation

The original system requires an **OAK-D Lite** camera with a Myriad VPU to run BlazePose
on-device. This makes it impossible to develop, debug, or test the intention prediction
model without the exact hardware used in the original experiment. The additions here
decouple the intention predictor from the OAK-D so it can be evaluated and improved on any
machine with a standard USB webcam.

---

## New File: `run_webcam.py`

Drop-in replacement for `run.py` that runs the full **perception → prediction** pipeline
using any OpenCV-compatible camera and **MediaPipe Pose** (CPU) instead of the OAK-D +
BlazePose (VPU) stack.

### Preprocessing parity with the original

Every step matches the original `run.py` and `Dataset.py` exactly so the pre-trained
DLinear checkpoint can be reused without retraining:

| Step | Original `run.py` | `run_webcam.py` |
|------|-------------------|-----------------|
| Landmark source | OAK-D BlazePose (VPU) | MediaPipe Pose (CPU) |
| Joint selection | `landmarks[11:25] + landmarks[0:1]` | Same indices from MediaPipe |
| Normalisation | min-max → [0, 2] | Identical |
| Camera→world rotation | quaternion `[0.14, -0.15, -0.76, 0.62]` | Same (can be disabled for debug) |
| Z-floor correction | `poses[:,:,2] -= min(Z)` | Identical |
| Sequence length | `seq_len = 5` | Configurable, default 5 |
| Confirmation window | 3 consecutive matching predictions | Identical |

### FPS limiter (`--proc_fps`)

The training data was captured at ~8 fps. Standard webcams deliver 20–30 fps. Without
throttling, the 5-frame window covers a much shorter time span — movements appear "faster"
to the model, hurting accuracy. `--proc_fps 8` (default) rate-limits model inference to
match the training cadence while the camera still captures at full speed.

### Exponential smoothing

A lightweight EMA (α = 0.4) over per-class probabilities is applied before argmax. This
reduces flickering between consecutive predictions with no added latency.

### Replay mode (`--replay <file.pkl>`)

Loads a `.pkl` file recorded by the original `run.py` (OAK-D data) and runs it through
the same prediction pipeline without any camera. Used to verify that the DLinear model
itself is correct before blaming the webcam as the source of error.

### ROS stubs

All `rospy` calls are present but commented out. The script runs on machines without a
ROS installation.

### Usage

```bash
# Basic run
python run_webcam.py --show --task webcam001

# Let the script find the first usable /dev/video* source
python run_webcam.py --show --task webcam001 --camera auto

# Force a specific device and camera format
python run_webcam.py --show --task webcam001 --camera /dev/video0 --cam_fourcc MJPG

# Full diagnostics at training FPS
python run_webcam.py --show --task webcam001 --diag --proc_fps 8

# Disable camera-to-world rotation (tests whether the OAK-D quaternion hurts)
python run_webcam.py --show --task webcam001 --diag --no_qrot

# Replay a .pkl from the OAK-D to verify the model itself is correct
python run_webcam.py --diag --replay human_traj/task001/task001.pkl

# Raw prediction with no OOD filter (shows what the model actually predicts)
python run_webcam.py --show --task webcam001 --restrict no
```

### CLI reference

| Argument | Default | Purpose |
|----------|---------|---------|
| `--show` | off | Display video window |
| `--task` | `webcam001` | Task ID (6-char, 3-digit suffix) |
| `--camera` | `auto` | OpenCV camera source: auto-detect, numeric index, or `/dev/video*` path |
| `--capture_backend` | `v4l2` | OpenCV capture backend (`v4l2` on Linux, or `any`) |
| `--cam_width` | 1280 | Requested capture width (`0` keeps driver default) |
| `--cam_height` | 720 | Requested capture height (`0` keeps driver default) |
| `--cam_fps` | 30 | Requested camera FPS (`0` keeps driver default) |
| `--cam_fourcc` | `MJPG` | Requested camera format (`MJPG`, `H264`, `YUYV`, etc.) |
| `--camera_buffer` | 1 | Capture buffer size; lower values reduce latency and stale frames |
| `--seq_len` | 5 | Frames in the prediction window |
| `--send_window` | 3 | Confirmations before forwarding intention |
| `--restrict` | `ood` | `no` / `ood` / `working_area` / `all` |
| `--model_type` | `final_intention` | `final_intention` or `final_traj` |
| `--video` | off | Save output as MP4 |
| `--diag` | off | Enable diagnostic overlay and terminal logs |
| `--no_qrot` | off | Use identity quaternion (disable camera rotation) |
| `--proc_fps` | 8 | Max inference FPS (0 = unlimited) |
| `--replay` | — | Path to `.pkl` for offline replay |

---

## New File: `depthai_blazepose/mediapipe_fallback.py`

Implements `MediaPipePoseModule`, a direct substitute for `BlazeposeDepthaiEdge` that runs
on any machine without DepthAI.

### Drop-in interface

Returns a `FakeBody` object with the same attributes as the original `body` object so it
can be swapped in without changing any downstream code:

```python
body.landmarks        # np.array (15, 3) — normalised image coords
body.landmarks_world  # np.array (15, 3) — metric world coords (origin at hip)
body.xyz              # np.array (3,)    — absolute position (zeroed; unavailable)
body.score            # float            — mean visibility of shoulder + wrist joints
```

### Joint mapping

Selects the same 15 upper-body joints as the original system using MediaPipe's indices:

```python
UPPER_BODY_MP_INDICES = list(range(11, 25)) + [0]  # shoulders→wrists + nose
```

This maps exactly to `body.landmarks[11:25]` + `body.landmarks[0:1]` from the OAK-D
pipeline, preserving the 45-dimensional feature vector expected by DLinear.

---

## Diagnostic Tools (inside `run_webcam.py`)

`--diag` activates structured output designed to test five failure hypotheses for why the
model may underperform when switching from OAK-D to webcam:

| Hypothesis | Metric | How to test |
|------------|--------|-------------|
| H1 — depth distribution shift | Shannon entropy of softmax | entropy < 0.4 = in-distribution |
| H2 — wrong quaternion for webcam | Prediction quality with/without rotation | `--no_qrot` |
| H3 — FPS mismatch | `proc_fps` in diagnostic panel | `--proc_fps 8` (match training rate) |
| H4 — OOD filter suppressing all predictions | Per-class probabilities | `--restrict no` |
| H5 — model broken with OAK-D data too | Prediction on original `.pkl` | `--replay <file>` |

Terminal output per frame with `--diag`:

```
[frame 0042] intention=get_connectors   entropy=0.312  motion=0.0231  qrot=ON  |  no_action=0.05  get_connectors=0.81  get_screws=0.09  get_wheels=0.05
```

---

## Runtime Optimizations on `webcam-runtime-optimizations`

This branch keeps the diagnostic behavior above and adds webcam runtime hardening:

- **Automatic camera discovery:** `--camera auto` scans `/dev/video*` sources and common
  numeric indices, then keeps the first source that actually returns frames.
- **Linux-friendly capture path:** `--capture_backend v4l2` is the default because it is
  usually more stable for USB webcams on Ubuntu.
- **Lower USB bandwidth by default:** `--cam_fourcc MJPG` requests MJPEG from compatible
  cameras. `H264` and `YUYV` are still available for comparison.
- **Lower live latency:** `--camera_buffer 1` reduces queued stale frames.
- **Frame normalization before processing:** grayscale and BGRA frames are converted to
  3-channel BGR before MediaPipe, drawing, and the diagnostic HUD.
- **Readable high-resolution HUD:** diagnostic overlays scale with frame height so the
  panel stays legible at 720p and higher.

---

## Documentation Added (`docs/`)

| File | Contents |
|------|----------|
| `modulo_predicao_hierarquica.md` | Architecture reference: `IntentionPredictor` layer, `PlanGraph` layer, and the ROS data-flow between them |
| `melhorias_e_modelo_experimental.md` | Prioritised improvement roadmap: z-score normalisation, temperature scaling, Bayesian smoothing of the intention queue, PlanGraph context injection, GCN skeleton encoder |
| `resumo_sessao_diagnostico.md` | Root-cause analysis session: the five hypotheses above, expected entropy ranges, and interpretation guide |
| `DIAGNOSTICO_WEBCAM.md` | Quick-reference card for interpreting `--diag` output |

---

## Dependency Changes

```bash
pip install mediapipe   # only new dependency — CPU pose estimation
```

No existing dependencies were changed or removed.

---

## What Was NOT Changed

- `run.py` — original OAK-D pipeline is untouched.
- `traj_intention/` — model architecture, weights, training code, and `IntentionPredictor` are identical to the original.
- `depthai_blazepose/` — all original DepthAI modules are preserved.
- `controller/receiver.py` — ROS-based receiver and `PlanGraph` are unchanged.
- `speech/` — speech recognition pipeline is unchanged.
- `env_ubuntu2004.yml` — conda environment is unchanged.

---

## Suggested Next Steps

The diagnostic work done in this branch points to three high-impact, low-effort improvements
for the intention predictor:

1. **Offline z-score normalisation** — compute `mean.npy` / `std.npy` from the training
   set once and replace the per-batch min-max in `run.py:297-298`. A single outlier frame
   currently distorts the scale of all 5 frames in the window; fixed statistics eliminate
   this. This is likely the largest source of accuracy degradation on webcam data.

2. **Temperature scaling** — fit a scalar `T` on the validation set so that
   `logits / T` produces calibrated probabilities. The OOD entropy thresholds (0.4 / 0.5
   in `predict.py:85-87`) were tuned manually and do not generalise across cameras or users.

3. **Fine-tune on webcam data** — after the normalisation fix, collect ~20% of the original
   dataset size with `run_webcam.py` and fine-tune the checkpoint for 10 epochs at `lr=1e-4`
   to close the remaining domain gap between OAK-D depth coordinates and MediaPipe estimates.
