# How much labeled data does a wrist accelerometer need to recognize exercises?

SWACSM 2026 poster project. We treat a learning curve as a **power analysis for a classifier**: how many athletes, and how many labeled examples per exercise, before a wrist-worn accelerometer reliably tells two lifts apart?

Data: uLift (Lim, Oh, Choi, "uLift: Adaptive Workout Tracker Using a Single Wrist-Worn Accelerometer," *IEEE Access* 12:21710-21722, 2024), public at https://github.com/JeiKeiLim/uLift-dataset. The raw data is not included in this repo.

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install numpy pandas scipy scikit-learn matplotlib joblib pytest pyyaml
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/JeiKeiLim/uLift-dataset data/uLift-dataset   # tested at 58edaf0

.\.venv\Scripts\python scripts/audit.py        # data audit -> results/audit_*.csv
.\.venv\Scripts\python -m pytest tests          # sanity tests
.\.venv\Scripts\python -m src.screen            # pair screen -> results/pair_screen.csv, figs/pair_screen.png
```

`scripts/check_loader.py` (which needs `pip install tqdm p_tqdm`) cross-checks our parser against the authors' loader.

## Decisions so far

See `STATUS.md` for the evidence behind each decision.
- Keep every sample; treat samples as evenly spaced within a segment; low-pass at 12 Hz; resample to 30 Hz. The authors' loader deduplicates timestamps, which drops ~35% of real samples.
- Units are m/s² (±16 g sensor). Segments with ≤1 rep are excluded, as in the authors' loader. That leaves 2,355 segments and 35 athletes.
- Channels are x, y, z and magnitude. Windows are 2 s long with 50% overlap and never cross a segment boundary.
- **Pair difficulty is descriptive.** It comes from a full-data screen and is used only to choose one easy, one medium and one hard pair. It is not a model-selection step, and every learning curve is evaluated on athletes held out separately in each repeat.
