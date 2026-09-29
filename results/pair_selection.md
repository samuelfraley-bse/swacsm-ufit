# Pair selection

Selected automatically from `results/pair_screen.csv` (full-data balanced accuracy, athlete-grouped 5-fold CV, RF).
Rule: in each band, prefer same-movement-family pairs, then closeness to the band centre, then more windows in the smaller class.
(All 15 classes have all 35 athletes, so athlete coverage does not discriminate between pairs.)
Pair difficulty is descriptive (used to stratify the learning curves), not a model-selection step;
learning curves are evaluated on separately held-out athletes in every repeat.

## easy (balanced accuracy 0.95-1.00, centre 0.975): 98 candidate pairs
- **Selected**: LUNGE vs BACK_SQUAT, balanced acc 0.982 (SD over folds 0.024), same family=True, windows 3711/3413
- Alternate 1: JUMPING_JACK vs BURPEE, balanced acc 0.983 (SD over folds 0.014), same family=True, windows 1496/4420
- Alternate 2: ARM_CURL vs BB_BENT_OVER_ROW, balanced acc 0.992 (SD over folds 0.016), same family=True, windows 3266/2672

## medium (balanced accuracy 0.80-0.92, centre 0.860): 3 candidate pairs
- **Selected**: PUSH_PRESS vs BB_MILITARY_PRESS, balanced acc 0.844 (SD over folds 0.067), same family=True, windows 3047/3037
- Alternate 1: DEADLIFT vs BB_BENT_OVER_ROW, balanced acc 0.917 (SD over folds 0.036), same family=True, windows 3767/2672
- Alternate 2: BENCH_PRESS vs PUSH_PRESS, balanced acc 0.904 (SD over folds 0.046), same family=False, windows 3217/3047

## hard (balanced accuracy 0.65-0.78, centre 0.715): 1 candidate pairs
Fewer than 3 pairs in band; alternates below include the nearest pairs outside it.
- **Selected**: BENCH_PRESS vs BB_MILITARY_PRESS, balanced acc 0.763 (SD over folds 0.050), same family=False, windows 3217/3037
- Alternate 1: PUSH_PRESS vs BB_MILITARY_PRESS, balanced acc 0.844 (SD over folds 0.067), same family=True, windows 3047/3037 [outside band]
- Alternate 2: BENCH_PRESS vs PUSH_PRESS, balanced acc 0.904 (SD over folds 0.046), same family=False, windows 3217/3047 [outside band]
