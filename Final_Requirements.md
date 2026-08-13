# Final Requirements

**Project:** Adaptive Neuro-AI - A Self-Healing Brain-Computer Interface System
**Programme:** B.Tech, Computer Science and Engineering (AI & ML)
**Institution:** IMS Engineering College, Ghaziabad
**Session:** 2025-2026
**Team:** Aditya Chauhan (2201431530002), Shiksha Pandey (2201431530050),
Vishal Kumar (2201431530059), Yogesh Pal (2201431530063)
**Supervisor:** Mr. Sachin Agrawal, Assistant Professor

This document is the single authoritative statement of everything the
finished project must contain. It consolidates the problem statement, the
approved synopsis, the sealed final report and the audit of the submitted
source code. Nothing outside this list is required; nothing inside it may be
dropped.

---

## 1. Purpose

Traditional human-computer interaction depends on keyboards, touchscreens and
voice, none of which are usable by people with severe physical disabilities.
This project delivers an adaptive, AI-driven brain-computer interface that
decodes EEG motor-imagery signals into digital commands, and that keeps
working when the signal degrades - without asking the user to recalibrate.

---

## 2. Functional requirements

### FR-1 Signal acquisition
- FR-1.1 Load EEG recordings in `.edf` format from the PhysioNet EEG Motor
  Movement/Imagery Database (109 subjects, 64 channels, 160 Hz).
- FR-1.2 Provide a documented signal simulator that models contralateral
  mu/beta desynchronisation, with a user-controllable noise level.
- FR-1.3 Provide simulated device profiles (OpenBCI Cyton, Muse 2, NeuroSky,
  PhysioNet 64-channel) with differing channel counts and sampling rates.
- FR-1.4 Expose a single extension point for future live hardware
  (`BCIEngine.scan_devices`). No fake hardware detection anywhere.

### FR-2 Preprocessing
- FR-2.1 Band-pass filter 7-30 Hz (FIR, firwin).
- FR-2.2 Segment into 4-second epochs starting at each annotated event.
- FR-2.3 Derive labels from the annotation description (T0 = REST,
  T1 = LEFT, T2 = RIGHT), never from raw MNE integer codes.
- FR-2.4 Resample to 160 Hz and fix the length at 640 samples.
- FR-2.5 Conform any channel count to the model's 64-channel input.
- FR-2.6 Per-channel z-score normalisation, safe against zero-variance
  channels.
- FR-2.7 Training and inference must execute the same preprocessing code.

### FR-3 Classification
- FR-3.1 EEGNet-TCN deep model: temporal convolution, depthwise spatial
  convolution over electrodes, separable temporal convolution, two dense
  layers.
- FR-3.2 Three-class output in the fixed order LEFT, RIGHT, REST, with
  softmax probabilities.
- FR-3.3 A single model definition shared by training and inference.
- FR-3.4 Automatic device selection: CUDA, then Apple MPS, then CPU.
- FR-3.5 Checkpoints store weights, optimiser state, validation accuracy,
  epoch, class order and input geometry.

### FR-4 Self-healing (the core contribution)
- FR-4.1 **Bad-channel detection** - flat electrodes and saturated or
  artefact-ridden electrodes, via a robust median-absolute-deviation z-score.
- FR-4.2 **Channel repair** - interpolation from healthy neighbours, capped
  at one third of the montage.
- FR-4.3 **Drift detection** - a rolling 20-window measure of mean prediction
  confidence and normalised output entropy, with configurable thresholds.
- FR-4.4 **Online adaptation** - entropy-minimisation fine-tuning with a
  diversity term, applied to the dense head only, over the recent unlabelled
  buffer, with the backbone frozen.
- FR-4.5 **Cool-down** - a minimum interval between adaptations so healing
  cannot thrash the weights.
- FR-4.6 **Rollback** - the trained baseline is retained and restorable at
  any time.
- FR-4.7 **Healing log** - every detection, repair and adaptation is
  timestamped, described and surfaced in the dashboard.
- FR-4.8 **Toggle** - self-healing can be switched off at runtime so healed
  and unhealed behaviour can be compared side by side.

### FR-5 Command output
- FR-5.1 Map LEFT / RIGHT / REST to cursor left / cursor right / recentre.
- FR-5.2 Cursor control is opt-in, lazily initialised, and degrades to a
  clear warning on headless systems.
- FR-5.3 Platform-specific permission guidance for macOS, Windows and Linux.
- FR-5.4 A working emergency stop (pyautogui fail-safe).

### FR-6 Dashboard
- FR-6.1 **Real-time BCI** - live probabilities, EEG traces, decoded command,
  confidence, prediction timeline, noise slider, fault injection, healing
  toggle, cursor toggle.
- FR-6.2 **EDF file analysis** - upload, per-epoch classification, class
  counts, confidence histogram, class-distribution chart, per-epoch table and
  accuracy against the file's own annotations.
- FR-6.3 **Device settings** - honest hardware scan plus simulated profiles.
- FR-6.4 **Model information** - checkpoint status and accuracy, architecture
  summary, parameter count, active device, pipeline description.
- FR-6.5 **Self-healing panel** - confidence, entropy, repairs, adaptations,
  drift status and the event log, visible on the relevant pages.
- FR-6.6 The UI contains presentation logic only.

### FR-7 Training and evaluation
- FR-7.1 Command-line training with configurable dataset root, epochs, batch
  size, learning rate, device and run group.
- FR-7.2 Subject-level split: 1-100 for training and validation (80/20
  stratified), 101-109 held out for testing.
- FR-7.3 Motor-**imagery** runs 4, 8 and 12 by default, with executed-movement
  runs selectable for comparison.
- FR-7.4 Class-weighted cross-entropy against the REST imbalance.
- FR-7.5 Adam with `ReduceLROnPlateau`; save the best-validation checkpoint.
- FR-7.6 Report accuracy, a per-class classification report and a confusion
  matrix on the held-out subjects.
- FR-7.7 Persist the loss and accuracy history for the report's figures.
- FR-7.8 Fixed random seed for reproducibility.

### FR-8 Resilience evaluation
- FR-8.1 Compare accuracy and confidence with self-healing on versus off.
- FR-8.2 Compare across increasing injected noise levels.
- FR-8.3 Compare with and without injected electrode faults.
- FR-8.4 Present the comparison as a table and a figure in the report.

---

## 3. Non-functional requirements

- NFR-1 **Performance** - a single window must be decoded well within its
  4-second duration on CPU.
- NFR-2 **Reliability** - a missing checkpoint, a bad EDF file, an absent
  desktop session or a broken electrode must never crash the application.
- NFR-3 **Portability** - runs on Windows 10/11, macOS 12+ and Ubuntu 20.04+,
  on CPU, CUDA or Apple MPS, with no absolute developer paths.
- NFR-4 **Reproducibility** - pinned dependencies, fixed seed, and a
  documented command behind every reported number.
- NFR-5 **Maintainability** - separated modules, a single source of truth for
  configuration, and no duplicated model definitions.
- NFR-6 **Testability** - a test suite that runs without the dataset and
  without a checkpoint.
- NFR-7 **Usability** - the dashboard must be operable by a non-programmer,
  with the current state always visible.
- NFR-8 **Honesty** - no simulated capability may be presented as real
  hardware or as real decoding accuracy.
- NFR-9 **Security and privacy** - EEG is biometric data; the project stores
  no personal data and processes uploaded files in a temporary location that
  is deleted immediately after use.
- NFR-10 **Ethics** - assistive research software, explicitly not a medical
  device and not clinically validated.

---

## 4. Technology stack (as approved in the synopsis)

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Signal processing | MNE-Python, SciPy, NumPy, pyEDFlib |
| Machine learning | PyTorch, scikit-learn, EEGNet architecture |
| Visualisation | Streamlit, Plotly, Matplotlib, Seaborn |
| Dataset | PhysioNet EEG Motor Movement/Imagery Database |
| Testing | pytest |

---

## 5. Deliverables

| # | Deliverable | Status in this package |
|---|---|---|
| D-1 | Complete, runnable source code | Delivered - `src/` |
| D-2 | Self-healing implementation | Delivered - `src/core/self_healing.py` |
| D-3 | Training and evaluation script | Delivered - `src/train_pytorch_real.py` |
| D-4 | Streamlit dashboard | Delivered - `src/app.py` |
| D-5 | Pinned dependencies | Delivered - `src/requirements.txt` |
| D-6 | Test suite | Delivered - `src/tests/`, 6 tests passing |
| D-7 | Project flow documentation | Delivered - `01_Project_Flow/` |
| D-8 | Rules, key points and code audit | Delivered - `02_Development_Rules_and_Key_Points/` |
| D-9 | Setup, run and deploy guides | Delivered - `03_Setup_Run_Deploy/` |
| D-10 | This requirements document | Delivered |
| D-11 | Trained checkpoint (`.pth`) | **Pending** - requires the dataset; see `03_Setup_Run_Deploy/02_Training.md` |
| D-12 | Retrained results on corrected labels and imagery runs | **Pending** - the 67% figure predates the label fix |
| D-13 | Self-healing evaluation table and figures | **Pending** - harness ready, needs a checkpoint |
| D-14 | Final report (9 chapters, as sealed) | Owned by the team; this package supplies the technical content |
| D-15 | Presentation slides and screenshots | Owned by the team; capture from the running dashboard |
| D-16 | Research paper draft | Owned by the team (March 2026 milestone) |

---

## 6. Acceptance criteria

The project is complete when all of the following hold.

1. `pip install -r src/requirements.txt` succeeds on a clean Python 3.10-3.12
   environment.
2. `python -m pytest src/tests` reports all tests passing.
3. `python -m src.train_pytorch_real --dataset-root <path>` completes and
   writes `models/pytorch_bci_model.pth`.
4. Held-out three-class accuracy on subjects 101-109 is materially above the
   33% chance level, and is reported with a confusion matrix and per-class F1.
5. `streamlit run src/app.py` opens all four pages with no errors.
6. Uploading a PhysioNet `.edf` file produces per-epoch predictions and an
   accuracy figure against the file's annotations.
7. Injecting an electrode fault produces visible repair entries in the healing
   log, and decoding continues.
8. Raising the noise level with self-healing disabled degrades confidence;
   enabling self-healing detects drift, logs an adaptation and recovers
   confidence - and this contrast is documented with numbers.
9. Cursor control works on a local desktop session and degrades with a clear
   message elsewhere.
10. No absolute developer path exists anywhere in the codebase.
11. Every claim in the final report is reproducible from a documented command.
12. The system never presents simulated data or untrained weights as real
    results.

---

## 7. Explicit exclusions

To keep the scope defensible, the finished project does **not** include:

- live EEG hardware acquisition (the extension point exists and is documented);
- clinical validation or any medical claim;
- more than three classes, or continuous cursor trajectory decoding;
- multi-user accounts, cloud storage of EEG data, or a mobile application;
- real-time performance guarantees below the 4-second window.

---

## 8. Immediate next actions

1. Download the PhysioNet dataset.
2. Retrain with `--runs imagery` on the corrected labels and record the new
   accuracy, confusion matrix and per-class F1.
3. Run the resilience comparison (FR-8) and produce the table and figures.
4. Capture dashboard screenshots for Chapter 9 of the report.
5. Update Chapter 7 of the report with the retrained numbers, replacing the
   pre-fix 67% figure.
6. Deploy to Streamlit Community Cloud and put the public URL in the report.
