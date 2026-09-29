"""Step 04: random forest, full new-team grid, R repeats (default 30), in parallel over repeats.

Saves per-athlete metrics and per-bout predictions for every (repeat, N, k, group), plus an
athlete-level summary. With --verify, reruns the first 3 repeats and asserts identical results.

    python steps/04_rf_repeats.py [--repeats 30] [--verify]
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from src import config as C  # noqa: E402
from src.data import get_windows  # noqa: E402
from src.experiment import run_repeat  # noqa: E402
from src.summary import summarize  # noqa: E402
from src.utils import git_hash  # noqa: E402

OUT = ROOT / "outputs" / "tables"


def make_rf_fit_predict(F, y):
    def fit_predict(trm, tem, seed):
        rf = RandomForestClassifier(C.RF_TREES, class_weight="balanced", n_jobs=1, random_state=seed)
        rf.fit(F[trm], y[trm])
        assert list(rf.classes_) == list(range(15))
        return rf.predict_proba(F[tem])
    return fit_predict


def one(r):
    X, F, meta = get_windows()
    return run_repeat(r, make_rf_fit_predict(F, meta.cls.to_numpy()), X, F, meta, "rf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    get_windows()  # build cache once before workers start

    t0 = time.time()
    out = Parallel(n_jobs=min(args.repeats, os.cpu_count()), verbose=5)(delayed(one)(r) for r in range(args.repeats))
    print(f"{args.repeats} repeats in {time.time() - t0:.0f} s")
    res = pd.concat([a for a, _ in out], ignore_index=True).assign(git=git_hash(), fs=C.FS, window_s=C.WINDOW_S)
    bouts = pd.concat([b for _, b in out], ignore_index=True)

    if args.verify:
        again = Parallel(n_jobs=3)(delayed(one)(r) for r in range(3))
        a = pd.concat([x for x, _ in again], ignore_index=True)
        b = res[res.repeat < 3].drop(columns=["git", "fs", "window_s"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a.reset_index(drop=True)[b.columns], b)
        print("verify: repeats 0-2 reproduce exactly")

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "04_rf_athletes.csv", index=False)
    bouts.to_csv(OUT / "04_rf_bouts.csv.gz", index=False)
    s = summarize(res)
    s.to_csv(OUT / "04_rf_summary.csv", index=False)
    piv = s.pivot_table(index="n_team", columns=["group", "k"], values="mean").round(3)
    print("\nMacro-F1, athlete-level mean (rows = team size N; columns = group, k):")
    print(piv.to_string())
    print("\n95% CI width (max over cells):", round((s.ci_hi - s.ci_lo).max(), 3))


if __name__ == "__main__":
    main()
