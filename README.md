# Adaptive Neuro-AI

**A self-healing brain-computer interface for EEG motor-imagery decoding.**

Imagined left-hand, right-hand and rest states are decoded from EEG into
digital commands by an EEGNet-TCN network. A self-healing layer repairs
broken electrodes, detects decoder drift and re-tunes the model online, so
the interface keeps working without manual recalibration.

B.Tech final-year project, Department of CSE (AI & ML), IMS Engineering
College, Ghaziabad, 2025-2026.
Aditya Chauhan, Shiksha Pandey, Vishal Kumar, Yogesh Pal.
Supervisor: Mr. Sachin Agrawal.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r src/requirements.txt
python -m pytest src/tests -q          # 6 tests, no dataset needed
streamlit run src/app.py               # dashboard on http://localhost:8501
```

To train:

```bash
python -m src.train_pytorch_real --dataset-root data/physionet
```

---

## Contents

```text
Adaptive-Neuro-AI/
  Final_Requirements.md                 authoritative requirements list
  README.md                             this file
  run.sh                                one-command launcher
  src/
    app.py                              Streamlit dashboard
    train_pytorch_real.py               training and evaluation
    requirements.txt                    pinned dependencies
    core/
      config.py                         all paths, constants, thresholds
      model.py                          EEGNet-TCN
      preprocessing.py                  filtering, epoching, normalisation
      self_healing.py                   repair, drift detection, adaptation
      bci_engine.py                     orchestration and inference
      mouse_control.py                  optional cursor control
    tests/test_pipeline.py              smoke tests
  01_Project_Flow/                      how the system works, end to end
  02_Development_Rules_and_Key_Points/  rules, code audit, domain notes
  03_Setup_Run_Deploy/                  setup, training, running, deployment
  models/                               checkpoints (generated)
  data/                                 dataset (downloaded)
  reports/                              generated figures
```

Start with `01_Project_Flow/01_End_to_End_Project_Flow.md`, then
`02_Development_Rules_and_Key_Points/01_Golden_Rules.md`.

---

## What changed from the submitted version

The original archive contained three files and a 1244-line `app.py`. This
package fixes four correctness defects (including a label-mapping bug that
shifted every training label by one class), eight crash-level defects, and
implements the self-healing layer that the report describes but the code did
not contain. The complete audit is in
`02_Development_Rules_and_Key_Points/02_Known_Issues_Found_And_Fixed.md`.

**Important:** the previously reported 67% accuracy was obtained with the
label bug present and on executed-movement runs. Retrain before quoting any
number.

---

## Dataset and citation

PhysioNet EEG Motor Movement/Imagery Database:
https://physionet.org/content/eegmmidb/1.0.0/

- Schalk G. et al. *BCI2000: A General-Purpose Brain-Computer Interface (BCI)
  System.* IEEE Transactions on Biomedical Engineering, 51(6):1034-1043, 2004.
- Goldberger A. et al. *PhysioBank, PhysioToolkit, and PhysioNet.*
  Circulation, 101(23):e215-e220, 2000.

## Disclaimer

Assistive research software. Not a medical device, not diagnostic, and not
clinically validated.
