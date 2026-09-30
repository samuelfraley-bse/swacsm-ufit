# STATUS

## ▶ RESUME HERE (updated 2026-09-29)

**Framing:** *How should a resource-limited strength program allocate a small wearable data-collection budget?* Public uLift data (wrist accelerometer, 15 exercises); pool of 23 adults recorded on ≥2 days. A simulated program records N athletes (2-17) × k sets per exercise (1-2) on day 1; testing uses later days only; 6 never-recorded athletes per repeat; 30 repeats; RF, CNN, LSTM. The athlete is the unit of analysis.

| RQ | Question | Answer (details in the results sections below) | Step |
|---|---|---|---|
| 1 | Breadth: how many athletes? | Big early gains; ~12 athletes reach 90% of the gain seen up to 17; no plateau by 17 | 04, 05, 08-D |
| 2 | Depth: a second set? | Helps (+0.02-0.04), but at an **equal budget more athletes × 1 set always wins** (12/12, p ≤ 0.003); extra sets targeted at the confused presses also lose to more athletes (step 09) | 08-A, 08-B, 09 |
| 3 | Record the target athlete? | Yes: recorded > new at every N; the gap shrinks (0.29 → 0.06, CNN) but never closes | 08-C |
| 4 | Does a 2-exercise battery transfer? | No; if anything it slightly hurts similar lifts (RF p ≤ 0.001) | 06, 08-F |
| 5 | Zero-cost label-free calibration? | Small, reliable gain at small N (CNN +0.04 at N=2, p = 5e-7); ns at N=17 | 06b |
| 5b | Can unlabeled wear substitute for labeled athletes? | Only when labels are scarce: self-training +0.02-0.04 at N=2 (≈ one extra athlete); nothing at N=4-8 | 10 |
| ~~why~~ | ~~What does the sensor use?~~ | **Removed from the paper (user, 2026-09-29).** The "posture shared, movement personal" explanation did not hold up (see step 15). Step 07 results are kept as a record only | 07, 15 |
| models | Which model? | RF best with little data, CNN best at N=17 (+0.04, p = 0.001), LSTM worst | 05, 08-E |

**Pipeline** (all under `steps/`; each run reproduces with the same seed):
01 manifest → 02 audit → 03 single RF run → 04 RF grid → 05 CNN/LSTM grid → 06 battery → 06b label-free → 07 sensor ablation → 08 statistics. Outputs go to `outputs/tables/`. The earlier pair screen (`src/screen.py`, `results/`, `figs/pair_screen.png`) is kept as a descriptive side panel.

**Code state:** last commit `711caac` (step 07). Uncommitted: `steps/08_analysis.py`, `outputs/tables/08_*`, this file, `drafts/`, `README.md`. No background jobs.

**Decisions on record**
- The edge/deployment step is **dropped**: this is not a deployment paper. At most one discussion sentence on model size (CNN ~33k parameters; calibration is one forward pass over one session).
- The power-law asymptote fit is not usable (hits the bound); report within-range N90/N95 instead.
- Battery fine-tuning uses class-balanced batches; the first recipe was confounded (disclosed below).
- Do not claim any posture-based mechanism (neither "forearm vertical" nor "posture shared, movement personal"). The posture finding is out of the paper; fig4 is not used on the poster.

**Figures: done (2026-09-29), `python steps/14_figures.py` → `figs/fig1_learning_curves`, `fig2_budget`, `fig3_shortcuts`, `fig4_posture_motion`, `fig5_realtime` (PNG 300 dpi + PDF).** Reference palette (validated slots; the Node validator was not available to re-run). Bars start at 0. The posture-only gap is 0.07 (0.0747), corrected from 0.08 in STATUS and the abstract.

**To do, in order** (item 1 superseded by the figures above)
1. ~~**Figures**~~: (a) equal-budget comparison (headline); (b) learning curves, one panel per model, recorded vs new; (c) gap G(N); (d) onboarding: record all 15 vs battery vs label-free; (e) posture/motion panel; (f) per-exercise recall vs N; optionally the worst-case athlete distribution.
2. **Revise the abstract** (`drafts/abstract_draft.md`) around the budget framing; see the notes at the top of that file.
3. Verify the citations before using them: the uLift paper (reported 90.06% accuracy?); the large-scale HAR study on labeled subjects vs data per subject; the Prudholme & Lockie SWACSM abstract.
4. **Late-breaking deadline Oct 9, 2026, 6 pm PST.** Use the new 2026 SWACSM template. Email Dr. Amorim (amorim@unm.edu) to confirm that a secondary analysis of public data qualifies.
5. Optional: "record only the confused lifts" battery; the coach-time framing (minutes of recording per budget).

## Step 15 results: poster extras (done 2026-09-29; `python steps/15_poster_extras.py`, no training)
- **fig7_what_the_watch_sees:** raw 3-axis wrist signal for one randomly chosen athlete (seed 7), first 8 s of deadlift, arm curl, bench press and military press. An illustration only; the presses look alike, consistent with the confusion matrix.
- **fig8_individual_athletes:** every never-recorded athlete's curve plus the mean. With 8 athletes recorded, 12/23 (RF) and 15/23 (CNN) new athletes are still below 0.80; the lowest athlete stays far below the rest.
- **fig9_confusion:** CNN, 8 athletes × 1 set, new athletes. The worst class is the military press: 41% correct, 34% called bench press, 20% push press. Bench press 63%, push press 81%. Squat ↔ back squat and row → deadlift are the other main confusions.
- **Regression** (`outputs/tables/15_regression.csv`): new-athlete macro-F1 ~ athlete intercepts + log2(athletes recorded) + log2(sets per athlete), using steps 04/05 (1-2 sets) and 12 (4 sets), with a cluster bootstrap over athletes. **Doubling athletes: RF +0.068 (0.060-0.075), CNN +0.098 (0.092-0.105). Doubling sets per athlete: RF +0.016 (0.012-0.020), CNN +0.039 (0.034-0.044). Ratio: RF 4.2× (3.3-5.5), CNN 2.5× (2.3-2.9).** Within-athlete R² 0.85 / 0.92. Poster sentence: "Doubling the athletes recorded buys 2.5-4× as much accuracy as doubling each athlete's sets."
- ⚠️ **Posture claim walked back (user, 2026-09-29).** A variance decomposition of set-level features (`outputs/tables/15_variance_decomposition.csv`; 1,861 sets, 23 athletes) shows the "posture shared, movement personal" line is too simple. Between-exercise share of variance: orientation x-axis 92%, z 72%, y 52%, orientation change within a set 44%, movement intensity 91%, tempo 36%. Between-athlete (same exercise) share: y-axis orientation 37% and tempo 38% are the personal parts. Bench and military press overlap in average wrist orientation. Step 07's aggregate result (motion-only has a larger recorded-new gap; orientation-free fails) still stands, but its one-line interpretation should not be claimed. **Decision: the posture finding is removed** from the abstract, findings and poster (fig4 not used).

## Step 13 results: real-time recognition (done 2026-09-29; RF 0.3 min, CNN 2.0 min on 24 cores)
Running call over the 2 s windows available so far (new window every 1 s), for never-recorded athletes' later-day sets; 1 set per exercise recorded. End-of-set results reproduce steps 04/05 exactly (720 rows each); repeat 0 reproduces. Tables: `outputs/tables/13_realtime_{rf,cnn}_*`.

Accuracy (share of sets named correctly) by time into the set:

| N | Model | 2 s | 4 s | 6 s | 10 s | end of set | median time to a stable correct call |
|---|---|---|---|---|---|---|---|
| 2 | RF | 0.68 | 0.68 | 0.69 | 0.69 | 0.69 | 2 s (~1.1 reps) |
| 2 | CNN | 0.61 | 0.62 | 0.63 | 0.63 | 0.63 | 2 s |
| 8 | RF | 0.79 | 0.79 | 0.80 | 0.81 | 0.81 | 2 s |
| 8 | CNN | 0.79 | 0.80 | 0.81 | 0.81 | 0.82 | 2 s |
| 17 | RF | 0.84 | 0.84 | 0.85 | 0.85 | 0.86 | 2 s |
| 17 | CNN | 0.87 | 0.87 | 0.88 | 0.89 | 0.89 | 2 s (~1.1 reps) |

- **The tracker knows almost immediately.** Accuracy after the first 2 s (about one rep; median set 18 s) is within 0.01-0.03 of the end-of-set accuracy. When a set is named correctly, it is typically correct from the first window and stays correct.
- **Waiting longer in a set barely helps; recording more athletes helps a lot.** At 2 s, going from 2 to 17 recorded athletes adds +0.16 (RF) / +0.26 (CNN); waiting to the end of the set adds ≤ 0.03.
- It also settles the metric question: single-window accuracy ≈ whole-set accuracy, so set-level voting does not meaningfully compress the effects (the Um et al. comparison concern).
- Accuracy here is the share of sets named correctly (averaged per athlete), not macro-F1; the end-of-set macro-F1 matches steps 04/05.
- **Leakage audit (`scripts/check_leakage.py`, 2026-09-29): all clean.** No new athlete in any repeat's team order (30/30); 0 raw set files shared across participants (0 duplicates overall); 0 of 46,347 windows identical across participants; no two IDs with identical gender + birth year + weight + height. Together with the per-step asserts (no athlete overlap, disjoint sets, training-only normalization, fixed recipes), there is no train/test leakage.
- **Caveats that make step 13 optimistic (state them on the poster):** (1) sets are pre-segmented, so the first window starts at the annotated set start; real use would first need set-start detection, which is not tested. (2) Preprocessing is non-causal within a set (zero-phase 12 Hz filter, tens of ms of look-ahead; the even-spacing time base uses the set's total duration). It uses same-set data only, so it cannot leak labels, but a strictly real-time pipeline would need causal processing.

## Step 12 results: few athletes recorded deeply vs many recorded once (done 2026-09-29; RF 0.9 min, CNN 4.5 min on 24 cores)
Same budget split three ways; all 15 exercises in every condition; only the number of *different people* changes. Deep sets 3-4 come from the athlete's next recording day, so depth includes day-to-day variation. Tested on never-recorded athletes only. All depth ≤ 2 conditions reproduce steps 04/05 exactly (1,080 rows each); repeat 0 reproduces. Tables: `outputs/tables/12_deep_broad_{rf,cnn}_*`.

| Budget (athlete-sets / exercise) | Deep | Middle | Broad | Broad − deep (RF) | Broad − deep (CNN) |
|---|---|---|---|---|---|
| 4 | 1×4: RF 0.57, CNN 0.52 | 2×2: 0.66, 0.61 | 4×1: 0.72, 0.68 | **+0.15**, 23/23 better | **+0.16**, 23/23 |
| 8 | 2×4: 0.69, 0.66 | 4×2: 0.74, 0.73 | 8×1: 0.78, 0.79 | +0.10, 22/23 | +0.13, 23/23 |
| 16 | 4×4: 0.76, 0.78 | 8×2: 0.81, 0.83 | 16×1: 0.83, 0.86 | +0.08, 23/23 | +0.08, 23/23 |

- **At every budget and in both models: broad > middle > deep.** All pairwise p < 0.001, except broad vs middle at 16 (RF p = 0.004, CNN p = 0.0003).
- **One athlete recorded four times is far worse than four athletes recorded once** (+0.15-0.16, every athlete better). Day-to-day variation within one person does not substitute for variation between people.
- This is the strongest version of the headline, and it closes the "depth = same-day second set" limitation.

## Step 11 results: simulated athletes / augmentation — DROPPED from the poster and abstract (user decision, 2026-09-29)
Kept in the repo as a record only. Small effects; the published recipe (Um et al. 2017) did not fit this task. (done 2026-09-29; RF 6.1 min on 12 cores, CNN 5.3 min on 24; both reproduce baselines exactly)
Budget 1 set per exercise. 4 simulated copies per recorded set (`src/augment.py`). The CNN gets the same number of gradient steps in every arm. New athletes; paired by athlete (n = 23). "≈ athletes" = extra real athletes with the same score on the real learning curve (blank when outside the curve). Tables: `outputs/tables/11_sim_{rf,cnn}_*`.

| N | RF tempo+intensity | RF + strap rotation | CNN tempo+intensity | CNN + strap rotation |
|---|---|---|---|---|
| 2 | −0.011 (p = 0.04) | +0.004 (ns) | −0.008 (ns) | +0.014 (p = 0.04) |
| 4 | −0.013 (p = 0.02) | +0.009 (p = 0.03) | −0.009 (ns) | **+0.030 (0.019-0.042, p = 1e-4; 19/23; ≈ +1.1 athletes)** |
| 8 | −0.008 (ns) | +0.010 (p = 0.05; ≈ +1 athlete) | −0.011 (p = 0.015) | +0.010 (p = 0.04; ≈ +0.8) |
| 17 | 0.000 (ns) | +0.015 (p = 0.007) | −0.008 (ns) | −0.001 (ns) |

- **Consistent across both models: simulated tempo and intensity changes slightly hurt; adding simulated strap rotation helps a little** (at best +0.03, about one extra athlete, for the CNN at 4 athletes).
- **The useful simulated variation is how the watch sits on the wrist, not how fast or hard someone lifts.** This agrees with step 07 (orientation carries the shared signal) and with the literature: Um et al. (ICMI 2017; wrist accelerometer, Parkinson's, 25 patients, subject-held-out 5-fold CV) found rotation-based augmentation most useful (best: rotation + permutation + time-warp, 77.5% → 86.9% window accuracy).
- Our gains are far smaller than Um et al.'s: less headroom (baseline 0.57-0.88 vs a struggling CNN on noisy labels), set-level voting already absorbs window errors, and our "tempo" is a global re-timing rather than their local time-warp.
- **Um et al.'s published recipe** (`--arms um2017`; Rot+Perm+TimeW on windows, ported from their official code with default parameters, i.e. *full* random rotation). CNN 4.1 min, RF 3.2 min; baselines reproduce; repeat 0 reproduces. Tables: `outputs/tables/11_sim_{cnn,rf}_um2017_*`.

  | N | CNN | RF |
  |---|---|---|
  | 2 | +0.010 (ns) | **−0.095** (p < 1e-4) |
  | 4 | +0.001 (ns) | −0.051 (p = 0.0004) |
  | 8 | **−0.040** (p = 0.0006) | −0.021 (ns) |
  | 17 | **−0.077** (p < 1e-4; 22/23 worse) | −0.016 (ns) |

  **It hurts, and the reason fits the whole story.** Full random rotation erases wrist orientation, which step 07 showed is the *shared*, informative signal for exercise recognition. In Um et al.'s Parkinson's task, orientation was a nuisance; here it is the signal. Small rotations (±15°, simulating strap placement) help slightly; full rotations destroy posture information. **Take-away: augmentation must respect the physics that carries the label.** The rotation-only arm was not run (dropped by the user).

## Step 10 results: semi-supervised self-training (done 2026-09-29; RF, 30 repeats, 5.4 min on 12 cores, reproducible)
Labeled: N athletes × 1 set. Unlabeled: day-1 wear of the other 17−N candidates (`ssl_team`), plus the 6 new athletes' own unlabeled day-1 wear (`ssl_team_new`, secondary). Pre-specified recipe: keep the most confident 50% then 80% of sets per predicted class, 2 rounds (`src/selftrain.py`). Both baselines reproduce step 04 exactly (960 and 690 rows). Tables: `outputs/tables/10_ssl_*`.

| Labeled N | Supervised | + unlabeled teammates | + teammates + new athletes' own wear | 17 labeled (ceiling) |
|---|---|---|---|---|
| 2 | 0.640 | 0.661 (**+0.022**, 0.006-0.037, p = 0.018) | 0.678 (**+0.038**, 0.026-0.050, p < 1e-4; 19% of gap to ceiling) | 0.839 |
| 4 | 0.717 | 0.718 (+0.002, ns) | 0.727 (+0.010, ns) | |
| 8 | 0.781 | 0.772 (−0.009, ns) | 0.790 (+0.009, ns) | |

- **Same pattern as label-free calibration (06b): unlabeled wear helps only when labeled data are scarce.** At N=2 it is worth about one extra labeled athlete (0.68 vs 0.72 for 4 labeled); by N=4-8 it adds nothing.
- Pseudo-labels on the kept sets are 78-93% correct (70-81% over all unlabeled sets); accuracy rises with N.
- **The 1-repeat smoke test (6 athletes) pointed the other way (−0.02 to −0.06).** That was noise; do not rely on single repeats.
- Not run (parked): the weighted-pseudo-label sensitivity variant.

## Step 09 results: targeted depth (done 2026-09-29; RF, 30 repeats, 1.5 min on 12 cores, reproducible)
Equal budgets of 18 sets per athlete-equivalent: 2nd set of the 3 presses vs 2nd set of 3 random non-press exercises vs 20% more athletes (1 set each). New athletes; paired by athlete (n = 23). Tables: `outputs/tables/09_targeted_*`.

| N | Comparison | Macro-F1 diff | p | Press recall diff | p |
|---|---|---|---|---|---|
| 5 | presses +2nd set vs base | −0.002 | 0.99 | +0.023 (−0.002 to 0.046) | 0.04 |
| 5 | random 3 +2nd set vs base | −0.002 | 0.45 | −0.004 | 0.36 |
| 5 | **more athletes vs presses +2nd set** | **+0.022** (0.012-0.034) | **0.0005** | −0.005 | 0.52 |
| 10 | presses +2nd set vs base | +0.001 | 0.54 | +0.012 | 0.14 |
| 10 | **more athletes vs presses +2nd set** | **+0.023** (0.014-0.034) | **0.0001** | +0.004 | 0.90 |

- **Targeting does not pay.** Extra press sets leave overall macro-F1 unchanged and give at most a marginal press-recall gain (+0.02 at N=5, CI touching 0; ns at N=10). Extra sets of random exercises do nothing.
- **More athletes win again**: +0.02 overall (p ≤ 0.0005). Even on the presses themselves, more athletes match the targeted sets (ns). This strengthens RQ2: spend any spare budget on breadth.
- Caveat: the presses were chosen from earlier descriptive analyses that included all athletes, which if anything favors the targeted condition.
- Fix during the run: the low-priority call in `steps/09_targeted_depth.py` passed a truncated Windows handle and silently failed; fixed (priority does not affect results).

## Step 08 results: statistics on saved results (done 2026-09-29; `steps/08_analysis.py`, no new training)
Framing adopted: **"How should a resource-limited program allocate a small wearable data-collection budget?"** RQ1 breadth, RQ2 depth, RQ3 recording the target athlete, RQ4 transfer from a short battery, RQ5 zero-cost label-free calibration. Step 07 explains *why*. The old step 08 (edge/deployment) is **dropped**; at most one discussion sentence on model size. Tables: `outputs/tables/08_*.csv`. All comparisons are paired by athlete (Wilcoxon; 95% bootstrap CI; n = 23).

- **A. Equal budget: breadth beats depth in all 12 comparisons.** With the same number of recorded sets, 2N athletes × 1 set beats N athletes × 2 sets for new athletes: RF +0.025 to +0.054, CNN +0.033 to +0.070, LSTM +0.041 to +0.060; all p ≤ 0.003; 16-23 of 23 athletes better. E.g. RF 8×1 = 0.78 vs 4×2 = 0.74; CNN 8×1 = 0.79 vs 4×2 = 0.73. **Probably the headline figure.**
- **B. Depth: correction to "a second set adds little".** It helps, just less than breadth: RF +0.02-0.03, CNN +0.04 (N ≤ 8) → +0.02 (N=17), LSTM +0.04-0.09; all p < 0.001.
- **C. Personalization gap G(N) shrinks but never closes.** RF 0.27 (N=2) → 0.13 (8) → 0.09 (17); CNN 0.29 → 0.11 → 0.06; LSTM 0.27 → 0.13 → 0.07. All p < 1e-5. Population diversity reduces but does not replace athlete-specific recordings.
- **D. How many athletes? Correction to "about 8".** Marginal gain per added athlete falls from 0.04-0.055 (2→4) to 0.003-0.015 (12→17). **Reaching 90% of the gain seen from N=2 to N=17 takes about 11.5-13 athletes** (RF k1 11.7, CI 11.0-13.3; CNN k2 11.5, 10.5-12.5); 95% takes about 14-16. At N=8 a program has ~70-75% of that gain. Caveat: this is relative to N=17, and the curve has not plateaued.
  - The power-law asymptote fit is **not usable**: A_inf hits the 1.0 bound and N95 comes out in the hundreds to thousands with huge CIs. Do not report it; just say the curve had not plateaued by 17.
- **E. Model comparisons confirm the crossover.** CNN − RF for new athletes: −0.051 at N=2 (p = 0.003), +0.019 at N=8 (p = 0.22, ns), **+0.039 at N=17 (p = 0.001)**. LSTM < RF at N=2 and 8 (p < 0.001), ns at 17; LSTM < CNN everywhere (p < 1e-5).
- **F. Transfer to *similar* exercises is not positive; if anything it interferes.** On non-battery exercises that are similar to a battery exercise, accuracy *drops* slightly: RF −0.037 for confusable pairs at N=8 (vs −0.002 for dissimilar; p = 2e-5), −0.020 at N=17 (p = 0.001). CNN and LSTM point the same way but are not significant (CNN −0.015, p = 0.08). Interpretation: recording someone's military press makes the model likelier to call their push press a military press. There is no generic athlete signature to learn cheaply; individual information is movement-specific.

## Step 07 results: what does the sensor use? — RECORD ONLY (posture interpretation removed from the paper, 2026-09-29) (done 2026-09-29; RF, k=2, 30 repeats, 1.4 min)
The "full" condition reproduces step 04 (1,110/1,110 rows, max diff 1e-16), and a rerun of repeat 0 reproduces exactly. Tables: `outputs/tables/07_rf_{athletes,summary}.csv`, `07_rf_bouts.csv.gz`. Paired tests: exploratory script (to be formalized in the evaluation step).

| N | Signal | New athletes F1 | Recorded F1 | Recorded − new gap (95% CI) | Press recall, new athletes |
|---|---|---|---|---|---|
| 8 | full | 0.81 | 0.94 | 0.13 (0.09-0.17) | 0.66 |
| 8 | posture (gravity, 0.3 Hz low-pass) | 0.72 | 0.85 | 0.13 (0.11-0.16) | 0.61 |
| 8 | motion (gravity removed) | 0.69 | 0.91 | 0.21 (0.17-0.26) | 0.51 |
| 8 | magnitude only | 0.50 | 0.71 | 0.21 (0.17-0.26) | 0.34 |
| 17 | full | 0.86 | 0.94 | 0.09 (0.06-0.12) | 0.69 |
| 17 | posture | 0.78 | 0.86 | 0.07 (0.06-0.09) | 0.70 |
| 17 | motion | 0.77 | 0.91 | 0.14 (0.11-0.19) | 0.58 |
| 17 | magnitude only | 0.56 | 0.70 | 0.15 (0.11-0.19) | 0.36 |

- **Posture and motion are complementary.** Each alone loses 0.07-0.12 for new athletes, and they are statistically indistinguishable from each other (p = 0.4 and 0.8).
- **The "personal" part lives in the movement dynamics.** The recorded-vs-new gap is significantly larger with motion only than with posture only (+0.08 at N=8, +0.07 at N=17; p ≤ 0.0003). With posture only it equals the full-signal gap (p = 0.5). Interpretation: how each exercise *orients the wrist* is shared across people, while *how someone moves through it* is individual. That is why recording an athlete helps, and why it helps exercise by exercise (consistent with the null battery result).
- **The hypothesis that the gap is about how the watch is worn is NOT supported.** Removing orientation (magnitude only) makes recognition far worse (−0.30) and *widens* the gap (+0.06 to +0.08, p ≤ 0.001). Orientation is the part that generalizes.
- ⚠️ **Revise the planned press "why" panel.** For new athletes at N=17, posture-only press recall (0.70) equals the full signal (0.69), and motion-only is worse (0.58). The presses are separated mostly by *posture*, so "they look the same to the wrist because the forearm is vertical" is too simple. The presses remain the hardest classes in every variant. Look at wrist orientation per press (e.g., watch-face direction) before claiming a mechanism.

## Step 06b results: label-free calibration (done 2026-09-29, 30 repeats, reproducible)
The new athlete wears the watch for their first day with **no labels**. CNN: 2.6 min; LSTM: 10 min. The "none" condition reproduces step 05 exactly. Tables: `outputs/tables/06b_{cnn,lstm}_{athletes,summary}.csv`.

CNN, new athletes, macro-F1 (gain vs none, 95% CI over 23 athletes):

| N recorded | None | AdaBN, day-1 wear | AdaBN + own input scaling | Own input scaling only | AdaBN on test days (secondary) |
|---|---|---|---|---|---|
| 2 | 0.611 | 0.653 (**+0.042**, 0.031-0.054) | 0.654 (+0.043) | 0.621 (+0.010, ns) | 0.663 (+0.052) |
| 8 | 0.829 | 0.855 (**+0.027**, 0.018-0.036) | 0.854 (+0.025) | 0.839 (+0.010, ns) | 0.861 (+0.032) |
| 17 | 0.895 | 0.907 (**+0.012**, 0.002-0.024) | 0.908 (+0.013, ns) | 0.895 (0.000) | 0.916 (+0.021) |

- **Paired tests (AdaBN vs none, Wilcoxon, n = 23):** N=2 p = 5e-7, 22/23 athletes improved (worst −0.006); N=8 p = 6e-5, 18/23 improved; **N=17 p = 0.07, 15/23 improved: treat as inconclusive**, even though the bootstrap CI (0.002-0.024) just excludes 0. Accuracy gain +3.4 / +2.1 / +1.2 points ≈ 0.5 / 0.3 / 0.2 extra correct sets per 15-set session. Statistically clear at small N, practically modest; the value is that it costs nothing.
- **AdaBN helps, and helps most when few athletes are recorded.** With no labels and no retraining, it closes ~15-25% of the recorded-vs-new gap. At N=2 it is worth about one extra recorded teammate (N=2+AdaBN 0.65 vs N=4 0.73 for the CNN); at N=8 it lands between the N=8 and N=12 scores.
- Rescaling inputs with the athlete's own statistics adds nothing on top of AdaBN. For the **LSTM** (no BatchNorm; input scaling only) it slightly **hurts** (−0.017 at N=8, −0.021 at N=17). So the gain comes from recalibrating the network's internal statistics, not from simple rescaling.
- Adapting on the unlabeled test days themselves is slightly better (+0.02-0.05) and is realistic for an on-device tracker, but it is reported as secondary.
- **Poster message:** "Just wearing the watch for one session, with no labels, improves recognition of a new athlete, most when the program has recorded few athletes." It is computationally tiny (forward passes only), which sets up step 08.

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

### Next (superseded by the new-team design)
Checkpoint 3 of the old pair-based plan (learning curves on 3 pairs) was never run.
