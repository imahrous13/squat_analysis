# Taqwin Gym Analyzer

A computer-vision gym exercise analysis system built for **Taqwin**. It processes exercise footage (uploaded video or live webcam), extracts human pose landmarks, identifies the exercise being performed, counts repetitions with a per-exercise state machine, evaluates form against biomechanical rules, and renders annotated video plus corrective feedback.

Everything in this document is derived from the current source tree.

---

## Overview

Training without feedback is the core problem the project addresses. A person performing a squat in front of a phone has no objective signal about depth, knee tracking, or tempo; a coach is not always present. **Taqwin Gym Analyzer** turns any ordinary camera recording into an automated, rule-based form report.

The system is a classic *landmark → geometry → rules* pipeline rather than a black-box action-recognition model:

1. A pose estimator produces 33 body landmarks per frame.
2. A rule-based detector inspects a short temporal window of landmarks to guess the exercise.
3. An exercise-specific analyzer converts landmark coordinates into joint angles and geometric ratios, drives a finite-state machine, counts repetitions, and grades each repetition.
4. Feedback (state, rep totals, fault list, advice) is burned into an annotated output video and streamed live.

The exercise classifier is **heuristic and rule-based**; the only learned model in the repository is an optional SigLIP2 image classifier used as a *hint* when the heuristic detector cannot decide (see [Rule-Based Exercise Detection](#rule-based-exercise-detection)).

---

## Key Features

- **Pose estimation from a single RGB camera** using MediaPipe Pose, with an optional YOLOv8 + MediaPipe hybrid path.
- **14 selectable exercise modes**, each backed by a dedicated analyzer class with its own rules, thresholds, and state machine.
- **Automatic exercise detection** from a temporal landmark buffer using range-of-motion, body-orientation, and symmetry heuristics, with a majority-vote confidence stage.
- **Finite-state-machine repetition counting** with per-transition debouncing, so noise and partial movements are not counted as reps.
- **Per-repetition form grading**: each rep is classified as correct or incorrect, and most analyzers emit a numeric rep score (0–100) plus a list of specific faults.
- **Fault detection including knee valgus, hip lift, elbow flare, lumbar rounding, torso swing, sag/pike hips, knee-over-toes, and excessive torso lean.**
- **Side-view and front-view handling** — analyzers detect the viewing angle, lock the more visible body side per rep, and adapt the rules to the view.
- **Landmark dropout tolerance** — a short hold on the last valid landmark set prevents the state machine from resetting when the pose model briefly loses the person.
- **Annotated video output** with an on-screen HUD (exercise, state, target muscles, reps, correct/incorrect totals, feedback, advice, score, faults, detected view), downloadable from the app.
- **Two input modes**: uploaded video and a live webcam stream over WebRTC (with session recording and download).
- **Automatic video orientation correction** for phone-recorded clips, including a vision-based fallback when rotation metadata is missing.

---

## Supported Exercises

| Exercise | Analysis method | Main joints / landmarks | Rep counting | Form feedback |
|---|---|---|---|---|
| Squat | Dedicated FSM analyzer (`SquatAnalyzer`), knee-angle driven | Hip, knee, ankle (knee angle), shoulders, hips, ankles for valgus | Yes — depth + tempo gated | Depth band, knee valgus ("Knee Cave"), per-rep pass/fail |
| Push-Up | Dedicated FSM analyzer (`PushUpAnalyzer`), elbow-angle driven | Shoulder, elbow, wrist (elbow angle), shoulder/hip/ankle body line | Yes — ROM gated (>30° elbow sweep) | Depth, sagging hips, pike hips, numeric score |
| Bench Press (lying) | `BenchPressAnalyzer(variant="standard")` | Shoulder, elbow, wrist (elbow angle), hip, ankle | Yes — ROM gated (min_rom) | Partial ROM, uneven press, elbow flare, hips lifted off bench, numeric score |
| Seated Bench Press | `BenchPressAnalyzer(variant="seated")` | Same as above, seated thresholds | Yes — ROM gated | Partial ROM, asymmetry, elbow flare, shoulder shrug, numeric score |
| Deadlift | `DeadliftAnalyzer`, 4-phase FSM | Shoulder, hip, knee, ankle, wrist, heel, toe (torso/hip/knee angles) | Yes — only *valid* reps increment the counter | Early hip rise, bar drift, lumbar rounding, incomplete lockout, overextension; all-or-nothing rep validity |
| Lunge | `LungeAnalyzer`, lead-leg knee-angle FSM | Hip, knee, ankle, toe, nose, shoulders | Yes | Depth, knee caving, hips not level, knee over toes, head over knee, forward lean, rep-too-fast, numeric score |
| Jumping Jacks | `JumpingJacksAnalyzer`, positional FSM | Wrists, shoulders, elbows, ankles | Yes | Elbow extension at the top ("Lazy Arms") |
| Plank | `PlankAnalyzer`, posture/hold timer | Shoulder, hip, ankle (body-line angle) | Not a rep count — accumulates **good hold seconds** and counts **form breaks** | Body-line angle outside 160–200°, hips too high or sagging |
| Chest Fly (cable/standing) | `ChestFlyAnalyzer(variant="standing")` | Wrists, shoulders, elbows, hip | Yes — squeeze ROM gated | Incomplete squeeze, "pressing instead of flying" (elbow ROM), non-upright torso, numeric score |
| Seated Chest Fly | `ChestFlyAnalyzer(variant="seated")` | Same as above, tighter thresholds | Yes | Same as above |
| Dips | `DipsAnalyzer(variant="normal")` | Shoulder, elbow, wrist, hip | Yes — ROM gated | Partial ROM, chest vs. triceps focus indication (torso lean), numeric score |
| Seated Dips | `DipsAnalyzer(variant="seated")` | Same as above | Yes | Same as above |
| Bent-over Row | `RowAnalyzer`, elbow-flexion rep counter | Shoulder, elbow, wrist, hip (trunk vector) | Yes | Torso swing, loss of hip hinge ("Too Upright"), arm asymmetry, incomplete pull, incomplete extension, numeric score |
| T-bar Row | `TBarRowAnalyzer` (subclass of `RowAnalyzer`) | Same as above | Yes | Same as above, with tighter swing and pull thresholds |

**Auto-detection coverage.** The heuristic detector can emit: Squat, Deadlift, Push-Up, Bench Press, Seated Bench Press, Lunge, Jumping Jacks, Plank, Chest Fly, Seated Chest Fly, Dips, Seated Dips, and Bent-over Row. **T-bar Row is manual-selection only** — the detector never produces that label, and the analyzer is only reachable through the exercise selector in the UI.

---

## System Architecture

```
Video file / Webcam stream
        |
        v
Orientation normalization  (metadata rotation scan + vision-based fallback)
        |
        v
Pose estimation
  |-- default: MediaPipe Pose on the full frame
  `-- hybrid  : YOLOv8 person track -> ROI crop -> MediaPipe Pose -> map landmarks back
        |
        v
Landmark stabilization  (hold last valid landmarks for up to 8 dropped frames)
        |
        v
Exercise resolution
  |-- manual selection (exercise chosen in the UI)
  `-- auto-detect: heuristic rule engine -> vote -> AutoDetectResolver
                     (+ optional SigLIP2 image hint)
        |
        v
Exercise-specific analyzer
  (joint angles, geometric ratios, view detection, side locking)
        |
        v
State machine + rep validation
  (debounced transitions, per-rep statistics, fault counters)
        |
        v
Rep counting + form scoring
  (correct / incorrect counters, rep score, fault list, advice text)
        |
        v
Visual feedback / results
  (on-frame HUD, annotated video, live WebRTC stream, downloadable MP4)
```

---

## Pose Estimation Pipeline

### Default path — MediaPipe Pose

`app.get_pose_detector()` returns `MoveNetEstimator` from `src/core/movenet_pose.py` when enhanced detection is disabled.

**Important naming note:** the class is named `MoveNetEstimator` and lives in `movenet_pose.py`, but it does **not** use MoveNet or any TensorFlow model. It is a thin wrapper around **MediaPipe Pose** (`mp.solutions.pose`). The name is a legacy label, not a description of the implementation. The same applies to `src/core/pose_detector.py` (`PoseDetector`), which is a second MediaPipe wrapper that is present in the tree but not wired into `app.py`.

MediaPipe is instantiated through a single configuration point, `src/core/pose_config.py`:

| Setting | Value |
|---|---|
| `static_image_mode` | `False` (video mode) |
| `model_complexity` | `1` |
| `smooth_landmarks` | `True` |
| `enable_segmentation` | `False` |
| `min_detection_confidence` | `0.6` |
| `min_tracking_confidence` | `0.7` |

The estimator also applies a **side-view visibility enhancement** step. It compares the average visibility of the left and right shoulder/hip landmarks, selects an "active side" with a hysteresis margin of `0.15` so the choice does not flicker between frames, then boosts the active side's visibility by a factor of `1.08` (capped at 1.0) and reduces the opposite side by `0.92`. This makes the downstream "pick the more visible limb" logic stable in profile views.

### Optional hybrid path — YOLOv8 + MediaPipe

Enabled by the **"Use AI-Enhanced Detection (YOLO + MediaPipe)"** checkbox in the sidebar. Implemented in `src/core/hybrid_pose.py` as `HybridPoseEstimator`:

1. **Person detection and tracking** — Ultralytics `YOLO('yolov8n.pt')` is called with `track(persist=True, classes=[0], conf=0.5)`, i.e. COCO class 0 (person) only. The model file `yolov8n.pt` is bundled in the repository root.
2. **Track selection** — YOLO is run every 3rd frame (and on the first frame) for speed; between runs the last ROI is reused. A **sticky track ID** is maintained. The candidate is selected purely by bounding-box **area** (largest person = closest to camera), and the lock is only switched to a different person if a candidate is more than 30% larger in area. A lock is dropped after 10 consecutive missed detections.
3. **ROI cropping** — `RoiManager` expands the detected box by 25% padding on each side, clamps it to the frame, and applies exponential moving-average smoothing (factor `0.3`) to the ROI coordinates to suppress jitter.
4. **Landmark extraction** — MediaPipe Pose runs on the **cropped ROI** only, which reduces the effective search area and improves landmark precision for the tracked person.
5. **Mapping back to the original frame** — every ROI-relative normalized landmark is converted to full-frame pixel coordinates and re-normalized against the full frame dimensions, so downstream analyzers always work in original-frame coordinates. The ROI rectangle is drawn on the output frame for transparency.
6. **Fallback** — if no person box is available, MediaPipe runs on the full frame instead.

**YOLO is used only for person detection, tracking and ROI generation. It performs no exercise classification and no form analysis.** Exercise recognition and biomechanical evaluation remain entirely MediaPipe-landmark-based and rule-based.

> **Dependency note:** `ultralytics` is currently commented out in `requirements.txt` (`# ultralytics (Disabled temporarily to ensure MediaPipe stability)`). Hybrid mode therefore requires `pip install ultralytics` explicitly; the default path does not.

### Landmark stabilization

`src/core/landmark_stabilizer.py` provides a short "hold" on the last valid landmark set. When the pose model drops the person, the previous landmarks are reused for up to 8 consecutive frames; after that the frame is treated as having no pose. This prevents the finite-state machines from resetting mid-rep because of a single dropped frame. It deliberately does not smooth coordinates — each analyzer applies its own angle smoothing.

---

## Rule-Based Exercise Detection

Detection lives in `src/core/exercise_detector.py` (`ExerciseDetector`) and is a pure heuristic engine over a **temporal landmark buffer**.

### Temporal buffer and confidence voting

- `buffer`: a `deque(maxlen=60)` holding the landmark set of each processed frame (~2 seconds at 30 FPS).
- `confidence_votes`: a `deque(maxlen=30)` holding the label guessed for each evaluated frame.
- Detection requires at least 30 buffered frames (50% of the window) before any guess is produced.
- A label is locked in when at least 18 votes (60% of the voting window) exist and the winning label holds **more than 70%** of the votes.

### Feature extraction

All features are computed over the buffer, in MediaPipe normalized coordinates (scale-independent):

- **Range of motion (ROM)** per joint group — hips (23, 24), elbows (13, 14), shoulders (11, 12) — as `max(Δy, Δx)`, with separate vertical-only ROM variants used for kinematic checks.
- **Body orientation ratios** — `horizontal_ratio` (shoulder-to-ankle horizontal distance vs. vertical distance) and `vertical_ratio`.
- **Torso lean ratio** — `|Δx| / |Δy|` between the mid-shoulder and mid-hip points (1.0 ≈ 45° of lean).
- **Arm spread ROM** — the range of the left-to-right wrist distance (used to separate flys from presses).
- **Knee asymmetry** — the difference in left/right knee heights, its average, maximum, and its ROM (used to separate lunges from squats).
- **Hand position** — wrists above/below shoulders, hands below hips, and the wrist-to-shoulder distance.
- **Movement-partition ratios** — for example `wrist_rom_y / shoulder_rom_y` (> 1.5 means the hands move more than the shoulders → bench-press-like; < 0.7 means the shoulders move more → push-up-like), and elbow ROM on the X axis.
- **Joint-angle ROM** — computed with the same three-point angle formula used by the analyzers, e.g. knee and elbow angle sweeps.

### Decision order

The detector evaluates candidates in a fixed priority order, and the first match wins:

1. **Deadlift** (priority) — hands consistently below the hips (>70% of frames), average hip height above the floor line, hip vertical ROM > 0.12, a predominantly vertical body, and a horizontal ratio below 0.5 (which keeps deadlifts from being labelled as planks).
2. **Jumping Jacks** — vertical body plus either hands above shoulders with high arm-spread and leg-spread variation, or a large arm-spread signature.
3. **Bent-over Row** — pulling elbow motion, a continuous forward-lean metric, non-trivial standing height, and arms moving at least as much as the hips.
4. **Horizontal branch** (horizontal body + forward lean) — separates:
   - **Bench Press** — wrists above the shoulders (supine), a flat vertical body profile, and either a hands-dominant movement ratio or a wide elbow-X signature. Bench press vs. **Chest Fly** is then decided by elbow-angle ROM: a large sweep (> 45°) is a press, a small one is a fly.
   - **Push-Up** — wrists at or below the shoulders (prone) with a shoulders-dominant movement ratio.
   - **Plank** — the horizontal fallback when the body is flat and movement is minimal.
5. **Vertical branch** (standing, `standing_height > 0.15`) — separates:
   - **Lunge** — knee asymmetry (max > 0.10 or mean > 0.04), a standard lunge signature (knee difference + ankle separation), or a changing asymmetry (knee-difference ROM); guarded against rows/deadlifts (torso lean or elbow motion) and against seated flys (large arm-spread ROM).
   - **Deadlift vs. Squat** — torso ROM with hands held low indicates a deadlift; otherwise, if the knees are symmetric (`avg_knee_diff < 0.06`) and the knee-angle ROM exceeds 20°, the result is a **Squat**; an asymmetric result falls back to Lunge.
   - **Dips** — an alternative branch when the elbow ROM is large and the torso remains upright.
6. **Seated branch** (vertical torso, hands in the chest zone, not horizontal) — separates:
   - **Seated Chest Fly** vs. **Chest Fly** — by arm-spread ROM and a low elbow-angle ROM, plus near-zero hip movement for the seated case.
   - **Seated Dips** — high elbow ROM with hands held low or dominant vertical wrist motion.
   - **Seated Bench Press** — high elbow ROM with hands at chest height and low hip movement.

Several of these branches carry explicit cross-exclusion guards (e.g. a wide stance suppresses the seated classification, knee-angle ROM above 25° suppresses the seated classification, and hip ROM above 0.12 marks the movement as a standing squat).

### Auto-detect resolution and the optional AI hint

`src/core/auto_detect.py` (`AutoDetectResolver`) finalises the decision:

- The heuristic result wins if there is one, with a single override: if the heuristic says **Squat** but the buffer shows a lunge signature (knee asymmetry combined with ankle separation), the result is **Lunge**.
- If the heuristic cannot decide, the resolver falls back to a **SigLIP2 image classifier** hint (`src/core/classifier.py`), but only after the hint has been repeated on **two consecutive evaluations** and the buffer shows real hip movement.

The SigLIP2 classifier is `prithivMLmods/Gym-Workout-Classifier-SigLIP2`, loaded lazily through `transformers` and `torch`, and returns the top-3 softmax probabilities. It is sampled **once every 60 frames** and only hints are accepted above a **0.65** confidence. It is strictly an *advisory* signal — it never overrides a confident heuristic verdict, and it never performs biomechanical analysis.

> **Caveat:** two of the model's mapped labels — `Biceps Curl` and `Lat Pulldown` — have no analyzer in this repository. If such a hint is accepted, the app locks onto that label with no analyzer attached and no rep analysis is produced.

---

## Exercise Form Analysis

Every exercise is analysed by its own analyzer class in `src/analyzers/`. The analyzers share a common pattern:

- Convert normalized landmarks to pixel coordinates (`get_landmark_pixel`).
- Compute joint angles with a three-point angle helper (`calculate_angle`).
- Detect the viewing angle (**FRONT** vs **SIDE**) and lock the better-visible side for the duration of a rep, so measurements do not flip mid-rep.
- Apply per-analyzer EMA smoothing to the driving angle before it reaches the state machine.
- Track per-rep statistics (minimum angle, maximum angle, fault counters, timestamps).
- Score the rep when the state machine closes the cycle.

### Example — Squat (`src/analyzers/squat_analyzer.py`)

The squat analyzer is the reference implementation and uses a five-state machine.

**States:** `START` → `STANDING` → `DESCENDING` → `BOTTOM` → `ASCENDING` → `STANDING` → …

| State | Entry condition | Behaviour |
|---|---|---|
| **START** | Initial state | Moves to `STANDING` when the smoothed knee angle exceeds the standing threshold of **165°**. Otherwise it prompts the user to stand straight and shows the live angle. |
| **STANDING** | Knee angle > 165° | A debounced transition (condition true for **3 consecutive frames**) on `knee_angle < 160°` starts the descent. Rep statistics are reset, the start hip height and rep start time are recorded, and the active side is locked. |
| **DESCENDING** | Knee angle < 160° | Tracks the minimum knee angle. A debounced transition to `BOTTOM` occurs when the knee angle drops below **130°** *or* the hips descend to knee level. If the knee angle climbs back above 165° instead, the state falls back to `STANDING` with a "Go deeper" message and the rep is abandoned. |
| **BOTTOM** | Knee angle < 130° or hips at knee level | Continues tracking the minimum knee angle. Exits to `ASCENDING` when the knee angle rises above **140°** (`bottom_angle + 10`), i.e. the ascent has clearly begun. |
| **ASCENDING** | Knee angle > 140° | The rep closes when the knee angle returns above **165°** for 3 consecutive frames. A second completion path exists: if more than 4 seconds have elapsed since the bottom was reached and the knee angle exceeds 150°, the movement is accepted as a completed rep (this prevents a slow, partial lockout from hanging the state machine). If the knee angle drops back below 130° during the ascent, the state returns to `BOTTOM`. |

**Depth scoring.** The minimum knee angle reached during the rep maps to a depth score:

| Minimum knee angle | Depth score | Interpretation |
|---|---|---|
| `< 100°` | 100 | Full depth |
| `< 120°` | 80 | Acceptable depth |
| `< 135°` | 50 | Too shallow |

**Tempo gate.** A rep is only counted if `depth_score > 0` (the knee angle got below 135°) **and** the rep took longer than **1.0 second**. Faster or shallower movements produce the message "Rep too shallow or fast" and are not counted.

**Correct vs. incorrect.** A counted rep is classified as **correct** when the depth score is at least 80 *and* no sustained knee valgus was detected. Otherwise it is classified as **incorrect**, with the advice string listing `"Go Deeper"` and/or `"Knee Cave"`.

**Knee valgus.** Only evaluated in the FRONT view: when the knee separation drops below **65% of the ankle separation**, the frame is flagged. A `SustainedFaultTracker` (5 of the last 10 frames required) converts the per-frame flag into a stable, user-facing "KNEE VALGUS!" warning, while the raw per-rep flag counter (> 10 frames) drives the correct/incorrect decision.

**View and side handling.** The view is classified as FRONT when the shoulder width exceeds 40% of the vertical shoulder-to-hip distance, otherwise SIDE. In the FRONT view the FSM is driven by the *average* of both knee angles; in the SIDE view it is driven by the knee angle of the locked side. The side is chosen by landmark visibility, locked at the start of each rep, and re-selected when the user returns to standing.

**Debounce.** All state transitions except the `BOTTOM → ASCENDING` exit use `debounced_transition`, which requires the trigger condition to hold for 3 consecutive frames and resets the counter to zero whenever the condition fails. This is what prevents landmark jitter from producing phantom reps.

### Other analyzers, briefly

- **Push-Up** — elbow-angle FSM (`TOP_PLANK → DESCENDING → BOTTOM → ASCENDING`) with an 8-frame debounce. Body rigidity is measured by the perpendicular distance of the hip from the shoulder-to-ankle line, separating **sagging hips** from **pike hips**. A rep counts only if the elbow swept more than 30°.
- **Bench Press** — elbow-angle FSM (`EXTENDED → DESCENDING → BOTTOM → ASCENDING → EXTENDED`) with variant-specific thresholds. Faults: elbow asymmetry (> 25°), elbow flare (hip-shoulder-elbow > 75°), hip lift (standard), shoulder shrug (seated). The seated variant uses looser lockout/depth thresholds and a shorter debounce.
- **Deadlift** — four-phase FSM (`SETUP → LIFT → LOCKOUT → RETURN`). Faults are debounced over 20 frames and are *all-or-nothing*: any confirmed fault (early hip rise, bar drift, lumbar rounding, incomplete lockout, overextension) invalidates the rep, and only valid reps increment the rep counter.
- **Lunge** — lead-leg knee-angle FSM with a lead-leg hysteresis. Faults are scored as a ratio of frames in the rep: depth, knee caving, hips not level, knee over toes, head over knee, forward lean, and rep duration.
- **Jumping Jacks** — positional FSM keyed on hands above the shoulders and feet separated by more than 1.5× shoulder width. A rep is counted when both conditions clear; the elbow angle captured at the top decides "Good Rep" vs. "Lazy Arms".
- **Plank** — a hold timer rather than a rep counter: it accumulates *good hold seconds* (exposed as the rep counter) and counts *form breaks*. The body line is the shoulder-hip-ankle angle, which must stay between 160° and 200°.
- **Chest Fly** — wrist distance normalised by shoulder width drives the FSM (`START → OPEN → CLOSING → CLOSED → OPENING`). Elbow-angle variance across the rep distinguishes a fly from a press; a standing variant also checks torso uprightness.
- **Dips** — elbow-angle FSM with a torso-lean measurement that labels the rep as chest-focused or triceps-focused.
- **Bent-over Row / T-bar Row** — a dedicated two-state `RepCounter` (elbow flexion down → up → extension) plus trunk-angle analysis against vertical, with penalties for torso swing, loss of the hip hinge, arm asymmetry, an incomplete pull, and incomplete extension.

---

## Rep Counting

Repetition counting is **not** a threshold-crossing counter on a raw angle. Every analyzer implements a finite-state machine that requires a complete, ordered cycle before a rep is registered:

```
START/EXTENDED/TOP  ->  DESCENDING  ->  BOTTOM  ->  ASCENDING  ->  back to start
```

Three mechanisms protect the counter:

1. **Debounced transitions** — a transition only fires when its condition has held for N consecutive frames (3 for squat and lunge, 8 for push-up, 5 for bench press, 4 for dips, 2 for chest fly). A single noisy frame cannot move the machine.
2. **Per-rep range-of-motion validation** — the rep must have actually covered the movement. The push-up requires a > 30° elbow sweep, the bench press a minimum ROM, the chest fly an actual squeeze, the squat a knee angle below 135°, and the row a real pull.
3. **Tempo and depth gating** — the squat rejects reps completed in under 1.0 s; the lunge penalises reps completed in fewer than 20 frames.

Incomplete movements are handled explicitly: the squat returns to `STANDING` with "Go deeper" if the user reverses early, the push-up and bench press report "Incomplete movement!" / "Incomplete range of motion!", and the deadlift discards invalidated reps entirely.

---

## Form Feedback

Form evaluation is implemented in two layers.

**Fault detection** — each analyzer maintains a set of named fault counters, incremented when a rule is violated and decremented when it is not, so brief transients do not accumulate. Faults are surfaced only after a debounce (for example 20 consecutive frames in the deadlift analyzer, 15 in the dips analyzer) or via the `SustainedFaultTracker` helper (5 of the last 10 frames), which the squat and lunge analyzers use for knee valgus.

**Per-rep scoring** — most analyzers subtract points from a starting score of 100 for each confirmed fault and then classify the rep:

| Analyzer | Pass threshold | Scoring style |
|---|---|---|
| Squat | Depth band ≥ 80 and no valgus | Categorical (pass/fail) |
| Deadlift | No fault at all | All-or-nothing validity |
| Push-Up | ≥ 70 | Numeric, faults listed |
| Bench Press | ≥ 60 | Numeric, faults listed |
| Lunge | ≥ 60 and no critical fault | Numeric, faults listed |
| Chest Fly | ≥ 60 | Numeric, faults listed |
| Dips | ≥ 60 | Numeric, faults listed |
| Row | ≥ 70 | Numeric, faults listed |

**User-facing output** — each analyzer returns a common dictionary containing `state`, `rep_count`, `correct_reps`, `incorrect_reps`, `feedback` (short status), `advice` (corrective instruction), `last_rep_score`, `reasons` (the fault list), `target_muscles`, and where relevant `view`, `valgus_detected`, and the individual joint angles. The app renders this as an on-frame HUD with the exercise name, the current state, the target muscles, the rep total, correct/incorrect totals, the feedback line, up to three wrapped advice lines, the last rep score, up to three detected faults, a "KNEE VALGUS!" banner, and the detected view.

---

## Application Modes

### Uploaded video analysis

- Accepts `.mp4`, `.mov`, and `.avi` uploads.
- The clip is written to a temporary file and previewed immediately.
- Pressing **"Analyze <exercise>"** runs `process_video()`, which renders a progress bar and writes an annotated MP4.
- The output is re-encoded with **PyAV** to H.264 / `yuv420p` with `movflags=faststart` so it plays in the browser, and is offered for download as `analyzed_<exercise>.mp4`.
- **Orientation handling** happens before analysis: a PyAV metadata scan reads the `rotate` tag and the `DISPLAY_MATRIX` side data; if the metadata is missing but the frame is landscape, a short vision scan checks whether the person is already upright (in which case no rotation is applied) or lying sideways using the nose-vs-hip geometry and, in hybrid mode, the YOLO box aspect ratio. A portrait guard prevents a double flip on clips that are already upright.

### Live webcam analysis

- Provided by `streamlit-webrtc` with PyAV frame handling, using a public STUN server.
- The webcam tab is only rendered when `streamlit-webrtc` and `av` import successfully; otherwise the app shows an informational message and keeps the upload tab.
- The frame is processed by the same `BaseVideoProcessor` pipeline as uploaded video, with a real-time HUD overlay and an FPS counter.
- The session is recorded to a temporary file and, once the stream stops, is re-encoded for playback and offered as `webcam_<exercise>.mp4` for download.

### Manual exercise selection

The sidebar radio control exposes all 14 modes directly. Selecting a mode bypasses detection entirely and instantiates the corresponding analyzer immediately, which is the most reliable path for unusual equipment or camera angles.

### Automatic exercise detection

Selecting **"Auto-Detect"** leaves the analyzer unset. The app continuously feeds the landmark buffer to the heuristic detector, shows a live progress indication ("Collecting data… %", "Analyzing pattern… %"), and once a label is locked it instantiates the matching analyzer and continues seamlessly into rep counting. The SigLIP2 hint path is only active in this mode.

### Optional enhanced detection mode

The sidebar checkbox **"Use AI-Enhanced Detection (YOLO + MediaPipe)"** switches the pose backend to the hybrid YOLOv8 + ROI + MediaPipe pipeline for both video and webcam input. It improves person localisation when several people are in frame, at the cost of additional compute and the `ultralytics` dependency.

---

## Project Structure

```
.
├── app.py                     # Streamlit entry point: UI, upload/webcam modes, video pipeline, HUD
├── requirements.txt           # Python dependencies
├── packages.txt               # System packages for Streamlit Cloud (libgl1, ffmpeg, ...)
├── .python-version            # Python 3.11
├── .streamlit/
│   └── config.toml            # Streamlit server/browser configuration
├── yolov8n.pt                 # Bundled YOLOv8 nano weights for the hybrid pipeline
├── run_app.bat                # Windows launcher (python -m streamlit run app.py)
├── start_tunnel.bat           # Windows helper: exposes the local app via a Cloudflare tunnel
├── RUNNING_THE_APP.md         # Notes on the local run procedure and environment conflicts
├── CLOUD_TROUBLESHOOTING.md   # Streamlit Cloud deployment notes
├── error_log.txt              # Historical Streamlit traceback log
└── src/
    ├── __init__.py
    ├── core/
    │   ├── __init__.py
    │   ├── pose_config.py     # Central MediaPipe Pose configuration (default + hybrid modes)
    │   ├── movenet_pose.py    # MoveNetEstimator - the default estimator (MediaPipe-based despite the name)
    │   ├── pose_detector.py   # PoseDetector - an alternative MediaPipe wrapper (not used by app.py)
    │   ├── hybrid_pose.py     # HybridPoseEstimator + RoiManager: YOLOv8 -> ROI -> MediaPipe -> remap
    │   ├── landmark_stabilizer.py # Short hold of last valid landmarks across pose dropouts
    │   ├── exercise_detector.py   # Rule-based exercise detector (ROM, orientation, voting)
    │   ├── auto_detect.py     # AutoDetectResolver: heuristic + squat/lunge guard + AI-hint consensus
    │   ├── classifier.py      # Optional SigLIP2 workout image classifier (lazy loaded, hint only)
    │   └── utils.py           # Angle maths, landmark helpers, EMA smoothing, debounce, fault tracker
    └── analyzers/
        ├── __init__.py
        ├── squat_analyzer.py          # Squat: 5-state FSM, depth scoring, knee valgus
        ├── pushup_analyzer.py         # Push-Up: elbow FSM, sag/pike detection
        ├── bench_press_analyzer.py    # Bench Press: standard + seated variants
        ├── deadlift_analyzer.py       # Deadlift: 4-phase FSM, all-or-nothing rep validity
        ├── lunge_analyzer.py          # Lunge: lead-leg FSM, multiple fault checks
        ├── jumping_jacks_analyzer.py  # Jumping Jacks: positional FSM
        ├── plank_analyzer.py          # Plank: hold timer and form-break counter
        ├── chest_fly_analyzer.py      # Chest Fly: standing + seated variants
        ├── dips_analyzer.py           # Dips: normal + seated variants
        └── row_analyzer.py            # Bent-over Row and T-bar Row
```

---

## Installation

**Requirements:** Python 3.11 (as pinned in `.python-version`).

```bash
git clone https://github.com/imahrous13/squat_analysis.git
cd squat_analysis

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install the application dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the versions that are known to work together:

```
streamlit
opencv-contrib-python
mediapipe==0.10.9
protobuf==3.20.3
numpy<2.0.0
streamlit-webrtc
av
transformers
torch
Pillow
```

**Optional — hybrid YOLO mode.** `ultralytics` is intentionally left commented out of `requirements.txt`. Install it manually if you want to use the "AI-Enhanced Detection" checkbox:

```bash
pip install ultralytics
```

**Optional — first run of the SigLIP2 hint.** The classifier model is downloaded from Hugging Face the first time the auto-detect path needs it, so the first detection attempt requires network access.

---

## Running the Application

```bash
python -m streamlit run app.py
```

On Windows, the bundled helper does the same thing:

```bat
run_app.bat
```

`start_tunnel.bat` is provided for exposing the locally running app through a Cloudflare quick tunnel for remote review; it requires `cloudflared` to be installed separately.

The app is then available at `http://localhost:8501` (Streamlit prints the actual port on start).

> Use `python -m streamlit run app.py` rather than a bare `streamlit run app.py` if several Python installations are present on your machine — see `RUNNING_THE_APP.md` for the environment-conflict notes this repository was developed against.

---

## Usage

1. **Start the application** with the command above and open the local URL.
2. **Choose an exercise** in the radio control at the top of the page — pick a specific exercise, or choose **"Auto-Detect"** to let the system decide.
3. **Choose the detection backend** in the sidebar. Leave *Use AI-Enhanced Detection* unchecked for the lighter MediaPipe-only pipeline; enable it for YOLOv8 person tracking and ROI-cropped pose estimation.
4. **Provide input**:
   - *Upload Video* — upload an `.mp4`, `.mov`, or `.avi` clip, then press **"Analyze <exercise>"**.
   - *Live Webcam* — start the stream, exercise in frame, and stop the stream to review and download the recorded session.
5. **Perform the exercise** in view of the camera. Watch the on-screen state, the rep counter, and the correct/incorrect totals. Under auto-detect, the HUD reports the data-collection and voting progress before locking the exercise.
6. **Inspect the results** — review the annotated video (playable in the browser and downloadable), the fault list, the per-rep score, and the corrective advice.

**Camera guidance.** Keep the whole body in frame, fix the camera at roughly torso height, and avoid backlighting. Side views are well supported for squats, lunges and rows; front views are required for the knee-valgus check.

---

## Technology Stack

| Technology | Role in the project |
|---|---|
| **Python 3.11** | Application language |
| **MediaPipe** (`mediapipe==0.10.9`) | 2D human pose landmark estimation (`mp.solutions.pose`) — the only pose model in the default path |
| **OpenCV** (`opencv-contrib-python`) | Frame I/O, colour conversion, video writing, text/landmark overlay drawing |
| **Streamlit** | Web UI, file upload, progress reporting, video playback and download |
| **streamlit-webrtc** + **PyAV** (`av`) | Live webcam streaming and H.264 re-encoding of the output videos |
| **NumPy** (`<2.0.0`) | Array maths for ROM, voting, angle computation and normalisation |
| **Ultralytics** (optional, `ultralytics` + bundled `yolov8n.pt`) | Person detection and tracking plus ROI generation in the hybrid pipeline |
| **Transformers** + **PyTorch** | Optional SigLIP2 image classifier used only as an auto-detect hint |
| **Pillow** | Frame → PIL image conversion for the SigLIP2 classifier |

All heavy dependencies (MediaPipe, Ultralytics, Transformers) are loaded lazily inside functions, which keeps the Streamlit Cloud startup footprint small.

---

## Design Decisions

**Landmark-based analysis instead of pixel-level modelling.** Every rule in the system is expressed in terms of a small set of body landmarks (hips, knees, ankles, shoulders, elbows, wrists, nose, heels, toes). Landmarks are invariant to the person's distance from the camera and to background, so a threshold expressed in joint angle behaves consistently across different recordings — something raw image-space heuristics cannot guarantee.

**Joint-angle geometry as the primary measurement.** The angle at a joint (e.g. hip–knee–ankle) is a biomechanically meaningful quantity and is directly comparable to coaching cues such as "knees below 90°" or "elbows tucked to 45–75°". Angles are computed with a single shared three-point helper so that every analyzer measures the same way.

**Finite-state machines over threshold counting.** A rep is a *sequence* (extend → descend → bottom → ascend), not a single excursion. Encoding that sequence explicitly guarantees that shrugging, walking, or partial bounces cannot be counted as repetitions, and it gives the UI a natural, human-readable state to display.

**Debouncing and hysteresis everywhere.** Landmark estimates jitter frame to frame. Requiring a condition to hold for N consecutive frames before transitioning, applying EMA smoothing to the driving angle, and using hysteresis margins on the side selection (and on the return-to-top thresholds) keep the state machine stable on real video.

**Rule-based biomechanical checks instead of a learned form classifier.** The faults that matter for coaching (knee cave, hip lift, elbow flare, lumbar rounding, sag/pike) are all expressible as geometric relations, and rule-based checks are transparent: every fault has an explicit threshold that can be explained, tuned, and justified. They also require no training data and remain inspectable, which matters for an academic project.

**Per-exercise analyzers rather than one generic engine.** Different exercises use genuinely different landmarks and different notions of a valid rep. Separate analyzer classes keep the rules explicit and allow exercise-specific thresholds (for example, a seated machine press has a different achievable lockout angle than a barbell bench press).

---

## Limitations

- **Single-camera, 2D pose estimation.** Depth is inferred from a monocular RGB image, so angles that are foreshortened (for example, the true hip angle when the camera is not perpendicular to the sagittal plane) are systematically biased.
- **Viewpoint dependence.** The rules are written for a specific camera placement. Frontal views are needed for the knee-valgus check; side views are needed for torso-lean, knee-over-toes and bar-drift checks. A three-quarter view degrades several of these.
- **Occlusion and visibility.** Landmarks are inferred rather than observed when a body part is hidden. The analyzers mitigate this by locking onto the more visible side, but a fully occluded limb will produce unreliable angles. The `LandmarkStabilizer` holds stale landmarks for up to 8 frames, which smooths over a dropout but also delays the system's reaction to a genuine exit from frame.
- **Lighting, background and clothing.** MediaPipe's confidence degrades under low light, strong backlighting, cluttered backgrounds, or loose clothing.
- **Hard-coded, non-personalised thresholds.** All biomechanical limits are fixed constants, tuned against typical adult proportions. Individual limb lengths, mobility, and training level are not modelled, so a very tall or very short athlete, or someone with limited ankle mobility, may be judged against the wrong limit.
- **The rule-based detector contains a few incomplete branches.** For example, the vertical-branch code path that would separate squats from dips references a variable that is not defined in the module, which raises an error on that specific path. Detection therefore remains sensitive to camera placement and to how distinctly the exercise is performed.
- **Auto-detect has a limited label set.** T-bar Row must be selected manually. The SigLIP2 hint can name exercises (`Biceps Curl`, `Lat Pulldown`) that have no analyzer in this repository, in which case no rep analysis is produced.
- **The hybrid pipeline still analyses one person.** YOLO selects and tracks the largest visible person and MediaPipe produces a single set of landmarks, so only one athlete is analysed at a time.
- **Webcam mode is platform-dependent.** WebRTC requires a secure context and is not available in every deployment target; the app falls back to upload-only in that case.
- **No performance evaluation is published.** There is no held-out dataset, no measured accuracy, and no validation protocol in the repository, so no accuracy or precision figures are claimed here.
- **`ultralytics` is not in `requirements.txt`.** The hybrid mode is off by default and requires an extra manual install.

---

## Future Work

The following are **proposed** directions, not implemented features:

- **A trained exercise classifier** replacing or augmenting the heuristic detector, evaluated on a labelled video corpus.
- **Personalised thresholds** derived from the user's own limb proportions captured in a calibration pose, instead of fixed constants.
- **A benchmarked evaluation set** with annotated ground-truth rep counts and form labels, enabling measurable accuracy reporting.
- **3D or multi-view pose estimation** to remove the monocular depth bias and improve angle accuracy.
- **Better handling of multiple people**, including per-athlete tracking and separate scoreboards.
- **Mobile deployment** as an on-device application.
- **Rep tempo analysis** (eccentric/concentric split) and a training-load summary across sets.
- **Audio or haptic cues** for live coaching without looking at the screen.

---

## Disclaimer

Taqwin Gym Analyzer provides **automated, rule-based exercise-form feedback**. It is a training and coaching aid, and it is **not a substitute for professional medical advice, physiotherapy, or certification by a qualified coach**. The thresholds are generic heuristics, and the system cannot diagnose injury, pain, or musculoskeletal conditions. If you are injured, recovering from an injury, or unsure whether an exercise is appropriate for you, consult a qualified healthcare professional or certified trainer before acting on any feedback the system provides.

---

## Repository

**Taqwin Gym Analyzer** — <https://github.com/imahrous13/squat_analysis>

Issues, bug reports, and contributions are welcome via the repository's issue tracker.
