# STATUS

## Checkpoint 1 — Data audit (2026-09-28)

Dataset clone: `data/uLift-dataset` @ `58edaf05c26c24fc1e856dc2383647efa2cc0be0` (2025-12-21). Git-ignored, read-only.
Scripts: `scripts/audit.py`, `scripts/check_loader.py`, `scripts/check_timing.py` → `results/audit_*.csv`.

### 1. License
- **The dataset repo (`JeiKeiLim/uLift-dataset`) has no LICENSE file.** The README asks for the IEEE Access 2024 paper to be cited and says nothing else about terms.
- The companion code repo (`JeiKeiLim/uLift`) is **MIT (c) 2025 Jongkuk Lim**. That covers the code, not clearly the data.
- Action: cite the paper; say on the poster that the data is "publicly available at github.com/JeiKeiLim/uLift-dataset"; do not redistribute raw data. Optional: email the authors to confirm research use is fine.

### 2. Sampling rate and timestamps
- Raw rate (samples − 1)/duration: **median 95.7 Hz** (IQR ~93-99; 1st pct 74 Hz; min 56 Hz). This looks like a nominal ~100 Hz sensor with some packet loss or jitter.
- Timestamps are BLE *arrival* times. Samples arrive in bursts: **~35% of consecutive rows share a timestamp**, yet only 0.02% of those have identical xyz values, so **they are distinct samples, not duplicates**. The median maximum inter-arrival gap is 166 ms (max 535 ms). No negative time steps.
- ⚠️ **The authors' loader runs `drop_duplicates(subset="Timestamp")`, which throws away ~35% of real samples** and reports ~62 Hz as a result. We do **not** do this.
- If samples are assumed evenly spaced between the first and last arrival time, the result differs from the arrival times by a median of 270 ms at worst within a segment (p95 640 ms). Most of that difference is BLE latency jitter, and the local sampling rate is preserved.
- **Decision:** keep all samples → assign evenly spaced times across the segment (the same idea as the authors' `fix_timestamp`, but without deduplicating) → low-pass filter (4th-order Butterworth, zero-phase, 12 Hz) → linear interpolation to **30 Hz**.
  - Why 30 Hz: rep rates are 0.3-1.0 Hz, and wrist movement energy relevant to lifting sits well below 15 Hz (Nyquist at 30 Hz). The ssl-wearables pretrained model also expects 30 Hz, so every arm can share one rate. It is also 3× cheaper than ~100 Hz for the CNNs. A 2 s window is 60 samples.

### 3. Units
- **m/s²**, not g and not raw counts. At rest the median |acc| is ≈ 10.4 (gravity 9.81 plus movement). Values clip at **±159.98 m/s² ≈ ±16 g**, which points to a ±16 g sensor range. Saturation happens mainly in jumping jacks and burpees.
- For the pretrained arm: divide by 9.81 to get g.

### 4. Per-class counts (after dropping segments with ≤1 rep, as the authors do)
- 35 athletes (folder name = `.info` nickname, 35 unique), 158 sessions, **2,355 workout segments, 23,738 reps**. This matches the README exactly. The raw files hold 2,394 segments; 39 have ≤1 rep (annotation mistakes, per the loader's comment) and are dropped.
- The actual filename pattern is `<nick>_<yyyy>_<mmdd>_<hhmmss>_segment<NN>_<TT|rest>.csv`. `NN` is the segment order and **is not always equal to the class**, so the class comes from `TT`. Class from the filename and class from the `.info` file agree in 100% of cases. The name of class 06 is spelled `DEADLIFT` in the `.info` files (the README has `DEAD_LIFT`).
- **Every class has all 35 athletes.** Median 5 segments per athlete per class (range 1-9), about 10 reps per segment, median segment 15-28 s. Totals: 29 min (jumping jack) to 78 min (burpee) per class. Full table: `results/audit_per_class.csv`.
- Rep rate (median reps/s): 0.34 (burpee) to 1.01 (jumping jack); most lifts are 0.4-0.55.
- **Consequence for the plan:** all 15 classes pass the ≥15-athlete rule → all **105 pairs** are eligible for the screen.

### 5. Loader check
- The authors' `dataset_loader` runs (it needs `tqdm`, `p_tqdm`). It returns 35 subjects, 158 sessions, 2,355 segments and 23,738 reps. **It matches my own parser one-to-one** on file set, class and rep count. The only difference is the timestamp dedup described in item 2.

### Decisions made (defaults for sections 3-4)
- 30 Hz, 4 channels (x, y, z, |a|), 2.0 s windows with 50% overlap, edge trim off.
- Segments with ≤1 rep excluded.
- Windows are generated per segment and never cross segment boundaries; a segment shorter than one window contributes 0 windows (logged).

## Checkpoint 2 — Preprocessing + pair screen (2026-09-28)

**What ran.** `src/` pipeline (load → 30 Hz resample → 4-channel 2 s / 50% windows → 59 features). `pytest tests` passes (4 tests: resampling, windows, features, athlete-disjoint splits). Screen: `python -m src.screen`, 105 pairs × 5 athlete-grouped folds, RF 300 trees with balanced class weights. Takes ~50 s on 24 cores. **Rerun with the same seed reproduces every number exactly (max diff 0.0).**
- 46,347 windows in total (1,496 jumping jack to 4,420 burpee per class). No segment was shorter than 2 s.
- Every generated fold is asserted to have no athlete overlap.

**Key result: most pairs are easy.** 98 of 105 pairs score ≥ 0.95 balanced accuracy on unseen athletes (27 are 1.000). Only 7 pairs fall below 0.95:

| Pair | Bal. acc | Macro-F1 |
|---|---|---|
| Bench press vs Military press | 0.763 | 0.762 |
| Push press vs Military press | 0.844 | 0.844 |
| Bench press vs Push press | 0.904 | 0.904 |
| Deadlift vs Bent-over row | 0.917 | 0.917 |
| Squat vs Lunge | 0.928 | 0.925 |
| Squat vs Back squat | 0.935 | 0.936 |
| Lunge vs Bent-over row | 0.946 | 0.948 |

**Selected pairs** (`results/pair_selection.md`, `results/selected_pairs.csv`, boxed in `figs/pair_screen.png`):
- **Easy:** Lunge vs Back squat (0.982)
- **Medium:** Push press vs Military press (0.844)
- **Hard:** Bench press vs Military press (0.763)

**Surprises / flags**
- ⚠️ **The hard band (0.65-0.78) contains only one pair,** and the medium band only three. I kept the bands as written; the alternates for the hard band are listed as out-of-band.
- A nice poster talking point: the hardest pair is bench vs military press. In both lifts the forearm is vertical and the bar is pushed against gravity, so **the wrist sees almost the same thing**. The lifter's body orientation (lying vs. standing) is invisible to a wrist sensor.
- The first auto-selection rule picked Good morning vs Deadlift (1.000) for "easy", which is pure ceiling. I changed the rule to *same family → closest to band centre → more windows*. This happened before any learning curve was run, so it does not use test outcomes.
- Fold-to-fold SD is 0.05-0.07 for the medium and hard pairs (7 test athletes per fold), so the difficulty labels are approximate.

## Design change (2026-09-28): "new team" sample-size study

This replaces the pair-based headline. The pair screen stays as a descriptive side panel.

- **Question:** a program starts with no data. It records k sets per exercise (k = 1, 2) from N athletes on day 1. How well does the tracker recognize exercises on later days (a) for the recorded athletes and (b) for athletes never recorded?
- **Target:** exercise class of each bout (15 classes). Primary metric is macro-F1; also balanced accuracy, accuracy, per-class P/R/F1 and confusion matrices.
- **Models:** random forest (features), 1D-CNN, LSTM. Poster shows one panel per model.
- **Split:** training uses day 1 only, testing uses later days only. The split is by day because same-day sessions are about 15 min apart.
- **Pool:** 23 participants, shared by both k. See `DATA_AUDIT.md`.
- Code lives in numbered step scripts under `steps/`. Outputs go to `outputs/`.
- The superseded pair-based plan is below for the record.

### Next (superseded)
Checkpoint 3: learning-curve harness (Axis A athletes, Axis B windows/class), RF arm only, R=10, sanity checks (shuffled labels, window-level leakage demo), then look at the curves.
