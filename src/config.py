"""Single source of configuration. All paths are relative to the repo root."""
from pathlib import Path

DATA_ROOT = Path("data/uLift-dataset/uLift_sensor_dataset")
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")
FIGS_DIR = Path("figs")

CLASS_NAMES = {
    0: "SQUAT", 1: "PUSH_UP", 2: "LUNGE", 3: "JUMPING_JACK", 4: "BENCH_PRESS",
    5: "GOOD_MORNING", 6: "DEADLIFT", 7: "PUSH_PRESS", 8: "BACK_SQUAT", 9: "ARM_CURL",
    10: "BB_MILITARY_PRESS", 11: "BB_BENT_OVER_ROW", 12: "BURPEE", 13: "LEG_RAISED_CRUNCH",
    14: "LATERAL_RAISE",
}

# Preprocessing (see STATUS.md checkpoint 1 for justification)
FS = 30.0                 # resample rate, Hz
LOWPASS_HZ = 12.0         # anti-alias cutoff before resampling
MIN_REPS = 2              # drop segments with <=1 rep (authors' loader does the same)
WINDOW_S = 2.0
OVERLAP = 0.5
EDGE_TRIM_S = 0.0         # optional per-segment edge trim; default off

# Pair screen
MIN_ATHLETES = 15
SCREEN_FOLDS = 5
RF_TREES = 300

# Learning curves
TEST_FRAC = 0.30
ATHLETE_BUDGETS = [2, 3, 5, 8, 12, 16, 20, 24]
WINDOW_BUDGETS = [2, 4, 8, 16, 32, 64, 128, 256]
DIFFICULTY_BANDS = {"easy": (0.95, 1.01), "medium": (0.80, 0.92), "hard": (0.65, 0.78)}
