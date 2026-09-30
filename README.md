# How should a strength program spend a small wearable data-collection budget?

SWACSM 2026 poster project. A wrist accelerometer can recognize resistance exercises, but a program must first record labeled examples. Using a public dataset, we simulate a program starting from zero. We ask how many athletes and sets it should record, whether each new athlete must be recorded, and whether cheap shortcuts (a short exercise battery, label-free calibration) can substitute. Results: `drafts/findings.md`. Progress log and decisions: `STATUS.md`.

Data: uLift (Lim, Oh, Choi, "uLift: Adaptive Workout Tracker Using a Single Wrist-Worn Accelerometer," *IEEE Access* 12:21710-21722, 2024), public at https://github.com/JeiKeiLim/uLift-dataset. Raw data are not included in this repo; the dataset repo has no license file, so cite the paper and do not redistribute.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install numpy pandas scipy scikit-learn matplotlib joblib pytest pyyaml
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/JeiKeiLim/uLift-dataset data/uLift-dataset   # tested at 58edaf0
.\.venv\Scripts\python -m pytest tests
```

## Pipeline

Each step is its own script, run from the repo root. Every run uses fixed seeds and reproduces exactly; most steps have a `--verify` flag that reruns part of the work and asserts identical results. Runtimes are on a 24-core CPU.

| Step | Command | What it does | Runtime |
|---|---|---|---|
| 01 | `python steps/01_manifest.py` | One row per exercise set (bout) → `outputs/tables/manifest.csv` | ~1 min |
| 02 | `python steps/02_audit.py` | Coverage and eligibility → `DATA_AUDIT.md` | seconds |
| 03 | `python steps/03_rf_single_run.py` | One random-forest run (sanity check) | ~1 min |
| 04 | `python steps/04_rf_repeats.py --verify` | RF, all team sizes × sets, 30 repeats | ~2 min |
| 05 | `python steps/05_deep_repeats.py --model cnn --verify` (and `--model lstm`) | CNN / LSTM, same grid | 8 / 42 min |
| 06 | `python steps/06_battery.py --model rf --n-team 8 17 --verify` (and `--model cnn`, `--model lstm`) | 2-exercise onboarding battery | 3-13 min each |
| 06b | `python steps/06b_label_free.py --model cnn --verify` (and `--model lstm`) | Label-free calibration (AdaBN) | 3 / 10 min |
| 07 | `python steps/07_sensor_ablation.py --verify` | Posture vs motion vs orientation-free signals (RF) | ~2 min |
| 08 | `python steps/08_analysis.py` | Paired statistics on the saved results (no training) | ~1 min |

Shared code is in `src/`: loading and resampling (`load.py`), windows and features, team sampling (`team.py`, `experiment.py`), models (`models.py`), battery and calibration (`battery.py`, `adapt.py`), signal variants (`variants.py`), metrics and summaries.

An earlier exploratory analysis (a 105-pair screen of which exercises a wrist sensor confuses) is in `src/screen.py` → `results/pair_screen.csv`, `figs/pair_screen.png`. Run it with `python -m src.screen`. It is kept as a descriptive side panel.

## Methods, briefly (for exercise scientists)

- **Signal processing:** keep every sample (the authors' loader discards ~35% by de-duplicating Bluetooth timestamps); resample to 30 Hz; 2 s windows with 50% overlap; channels x, y, z and total acceleration. Each set is classified by majority vote of its windows.
- **Split:** athletes contribute day-1 sets for training; testing uses their later days only, so training and test sets never come from the same workout.
- **Statistics:** each athlete's score is averaged over the repeats in which they appear in a role. Comparisons are paired by athlete (Wilcoxon signed-rank), with 95% bootstrap confidence intervals across the 23 athletes. Windows are never treated as independent observations.
- **No tuning on test athletes.** Each model has one fixed training recipe.
