# STATUS

## ▶ RESUME HERE (paused 2026-09-29, early morning)

**Story (agreed):** a wrist tracker for a strength program, told as three questions.
1. **Starting a team:** how many athletes and sets to record? (steps 04-05, done)
2. **Onboarding a new athlete:** full recording vs a 2-exercise battery (step 06, done: no transfer) vs label-free calibration (step 06b, written but not run). Step 07 explains *why*.
3. **Deploying:** can the best model be compressed to run on a watch, calibration included? (step 08, not started)

**State of the code**
- Last commit: `629875d` (steps 03-04). **Uncommitted:** `src/models.py`, `src/battery.py`, `src/adapt.py`, `steps/05_deep_repeats.py`, `steps/06_battery.py`, `steps/06b_label_free.py`, all `outputs/tables/05_*` and `06_*`, this file, and `drafts/`.
- No background jobs are running. All step 05/06 runs finished and passed their same-seed reproducibility checks.
- Step 06b smoke test **was not run** (interrupted by the user). Next action: `python steps/06b_label_free.py --model cnn` (runtime unmeasured, estimated ~5-10 min), then `--model lstm` (input_norm only; the LSTM has no BatchNorm). The smoke check to run first: in repeat 0, the "none" condition should equal the step 05 CNN new-athlete scores (N=17, k=2).

**To do, in order**
1. Commit the current work (ask the user first).
2. **06b** label-free calibration: CNN (input_norm, AdaBN, both, AdaBN on test days as secondary), LSTM (input_norm). N = 2, 8, 17.
3. **07** "What does the sensor use?", RF at N = 8, 17: (a) gravity/posture-only vs motion-only signals, (b) magnitude-only (orientation-free). Does the recorded-vs-new gap shrink with magnitude only?
4. **08** edge panel: parameter count and size (RF vs CNN vs LSTM), 8-bit quantization accuracy drop on the same test athletes, operations per window, CPU latency per window (proxy; no real device), cost of AdaBN calibration.
5. Evaluation step: formal paired tests (CNN vs RF vs LSTM; recorded vs new; k1 vs k2), moving the exploratory script into `steps/`.
6. Figures: three panels (RF | CNN | LSTM), macro-F1 vs athletes recorded, 4 lines (k × recorded/new), CIs.
7. Optional battery ideas not yet run: (#1) record only the confused lifts (bench, military, push press) and score on all 15; (#3) within-family transfer, analysis only, from the existing random-pair results.
8. Poster ideas: data expressed as coach time (minutes of recording), worst-case athlete distribution, a "why" panel for press confusion (wrist angle vs gravity), per-exercise recall vs N, QR code to an interactive calculator (last).
9. Abstract: draft in `drafts/abstract_draft.md`. **Late-breaking deadline Oct 9, 2026, 6 pm PST** (general and expanded deadlines have passed). Use the new 2026 SWACSM template. Email Dr. Amorim to confirm that a secondary analysis of public data qualifies as late-breaking.

## Step 05 results: CNN and LSTM, 30 repeats (done)
- CNN 8.1 min, LSTM 41.9 min. Both reproduce repeats 0-1 exactly. Tables: `outputs/tables/05_{cnn,lstm}_*`.
- Macro-F1, athlete-level mean:

| N | RF new k2 | CNN new k2 | LSTM new k2 | RF rec k2 | CNN rec k2 | LSTM rec k2 |
|---|---|---|---|---|---|---|
| 2 | **0.66** | 0.61 | 0.49 | 0.93 | 0.90 | 0.76 |
| 8 | 0.81 | **0.83** | 0.74 | 0.94 | 0.94 | 0.87 |
| 17 | 0.86 | **0.90** | 0.85 | 0.94 | **0.96** | 0.92 |

- **Crossover:** RF is best with little data, the CNN is best from ~8 athletes up, and the LSTM trails throughout (data-hungry with a fixed 30-epoch recipe). The user chose to keep the LSTM recipe as is, with no sensitivity run.
- Paired tests between models have not been run yet.

## Step 06 results: battery study (done, 30 repeats, reproducible)
Gain on the new athlete's **other 13 exercises** from recording 2 movements (athlete-level mean, 95% CI), vs recording all 15:

| Model | N | Squat+deadlift | Push-up+push press | Random pair | All 15 recorded (vs none) |
|---|---|---|---|---|---|
| RF | 8 | −0.005 (−0.010 to −0.001) | −0.008 | −0.007 | +0.11 to +0.13 |
| RF | 17 | −0.004 (−0.008 to 0.000) | −0.007 | −0.004 | +0.07 to +0.09 |
| CNN (balanced fine-tune) | 17 | −0.003 (−0.007 to 0.000) | −0.000 (−0.004 to +0.003) | −0.001 | +0.06 to +0.07 |
| LSTM (balanced fine-tune) | 17 | −0.001 | −0.003 | −0.001 | +0.07 |

- **Clear null: a 2-movement battery does not transfer to other exercises** in any model. The benefit of recording an athlete is exercise-specific. Mislabeling of other bouts as battery moves barely changes (e.g., CNN 1.3% → 1.5%).
- Coach message: record the exercises you want tracked; a short generic battery is not a substitute. This motivates 06b (label-free) and 07 (magnitude-only: is any of the gap about watch orientation?).
- Tables: `outputs/tables/06_battery_{rf,cnn,lstm}_{summary,scores}.csv`, predictions in `*_bouts.csv.gz`.

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

### Step 03: single RF run (seed 0, commit efc966c)
Bout-level macro-F1 (mean across test athletes, 95% bootstrap CI). One seed only; the numbers are noisy.

| Team N | Sets/exercise k | Recorded athletes (later days) | New athletes (6) |
|---|---|---|---|
| 2 | 1 | 0.88 (0.80-0.96), 2 athletes | 0.65 (0.59-0.70) |
| 2 | 2 | 0.98 (0.96-1.00), 2 athletes | 0.62 (0.54-0.70) |
| 17 | 1 | 0.93 (0.91-0.96) | 0.87 (0.79-0.94) |
| 17 | 2 | 0.94 (0.92-0.96) | 0.86 (0.80-0.93) |

- There's room for a curve. For **new** athletes, the score rises from ~0.63 to ~0.86 as the team grows, so team size is the main lever.
- **An athlete's own recorded data matters a lot.** Recorded athletes score 0.88-0.98 even with a team of 2. At N=17 the recorded-vs-new gap is ~0.07-0.08.
- **A second set per exercise adds little** at N=17.
- Remaining errors are mostly the press cluster (bench ↔ military ↔ push press), matching the pair screen.

### Step 04: RF, 30 repeats (`steps/04_rf_repeats.py`)
- Runtime: 124 s on 24 cores. Repeats 0-2 rerun and **reproduce exactly**.
- Nested design: teams nest across N and sets nest across k within a repeat. The 6 new athletes are fixed within a repeat.
- Outputs: `outputs/tables/04_rf_athletes.csv` (one row per athlete × repeat × N × k × group), `04_rf_bouts.csv.gz` and `04_rf_summary.csv`.
- Values are bout-level macro-F1. Each athlete is averaged over repeats first, then a 95% bootstrap CI is taken over the 23 athletes:

| N | New, k=1 | New, k=2 | Recorded, k=1 | Recorded, k=2 |
|---|---|---|---|---|
| 2 | 0.64 (0.60-0.67) | 0.66 (0.63-0.70) | 0.89 | 0.93 |
| 8 | 0.78 (0.74-0.82) | 0.81 (0.77-0.85) | 0.91 | 0.94 |
| 17 | 0.84 (0.80-0.87) | 0.86 (0.82-0.89) | 0.92 | 0.94 (0.93-0.96) |

- Paired athlete-level comparisons (n = 23, Wilcoxon signed-rank; exploratory script, to be formalized in the evaluation step):
  - Recorded vs new at N=17, k=2: **+0.086** (0.056-0.122), p = 1e-6
  - New athletes, N 2→8 (k=1): **+0.141** (0.119-0.162); N 8→17: **+0.058** (0.046-0.071). The gains flatten but have not plateaued by 17.
  - New athletes, k 1→2 at N=17: **+0.017** (0.010-0.023). Statistically clear but small.
- **Step 05 (CNN, LSTM):** a fixed recipe (30 epochs, Adam 1e-3), no tuning. The CNN finished in 8 min and repeats 0-1 reproduce. Refactoring `models.py` into train/predict was checked against the saved CNN results for repeat 0 (identical).

### Step 06: battery study, design notes (2026-09-29)
- **Question:** the new athlete records only 2 movements (squat + deadlift, push-up + push press, or 2 random pairs). Does that help recognize their *other* 13 exercises on later days? The reference is recording all 15. Uses the same repeats, new athletes and team order as steps 04-05, with team k=2.
- **RF:** retrains from scratch per condition. By construction the RF cannot transfer person-level adjustments across classes, so it is expected to show little transfer.
- **CNN/LSTM:** the team model is trained once, then fine-tuned (150 steps, lr 1e-4). The no-battery control gets the same fine-tuning budget on team data only.
- **Recipe change after a 1-repeat smoke test (disclosed):** the first recipe put the athlete's battery windows in 50% of every batch. That shifted the class prior toward the 2 battery movements: the battery *hurt* by 2-6 points and more bouts were mislabeled as battery moves (0.6% → 3.8%). This was a confound, not a finding. **Switched to class-balanced batches**: labels are drawn uniformly over the 15 classes, and within a recorded class 50% of examples come from the athlete. The naive recipe is not reported further.
- **1-repeat smoke test with the balanced recipe (CNN, N=17):** battery gain −0.007 to +0.003; recording all 15 gives about +0.05. Full results are in "Step 06 results" at the top.

- **Outlier:** Subject494 is the lowest new athlete (0.60), and was also the participant with extreme jumping-jack magnitudes in the audit. Possibly a different strap placement. Keep them in, and flag it as a limitation.

### Next (superseded)
Checkpoint 3: learning-curve harness (Axis A athletes, Axis B windows/class), RF arm only, R=10, sanity checks (shuffled labels, window-level leakage demo), then look at the curves.
