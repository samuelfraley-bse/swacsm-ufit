"""Step 07: what does the sensor use? Random forest on signal variants (see src/variants.py).

Same repeats, teams and test athletes as step 04 (k=2). Variants: full (baseline; should reproduce
step 04), gravity (posture only), motion (movement only), magnitude (orientation-free).
Questions: (a) posture vs motion: which carries the exercise identity, and which separates the
presses? (b) magnitude-only: does removing watch orientation shrink the recorded-vs-new gap?

    python steps/07_sensor_ablation.py [--n-team 8 17] [--repeats 30] [--verify]
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
from src.experiment import run_repeat  # noqa: E402
from src.metrics import bootstrap_ci  # noqa: E402
from src.utils import git_hash  # noqa: E402
from src.variants import VARIANTS, get_variant_windows  # noqa: E402

OUT = ROOT / "outputs" / "tables"
PRESSES = [4, 7, 10]  # bench press, push press, military press


def one(r, variant, team_sizes):
    X, F, meta = get_variant_windows(variant)
    y = meta.cls.to_numpy()

    def fp(trm, tem, seed):
        rf = RandomForestClassifier(C.RF_TREES, class_weight="balanced", n_jobs=1, random_state=seed)
        rf.fit(F[trm], y[trm])
        return rf.predict_proba(F[tem])

    res, bouts = run_repeat(r, fp, X, F, meta, "rf", team_sizes=team_sizes, ks=[2])
    return res.assign(variant=variant), bouts.assign(variant=variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-team", type=int, nargs="+", default=[8, 17])
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    for v in VARIANTS:  # build caches once, before workers start
        get_variant_windows(v)

    t0 = time.time()
    jobs = [(r, v) for v in VARIANTS for r in range(args.repeats)]
    out = Parallel(n_jobs=os.cpu_count(), verbose=5)(delayed(one)(r, v, args.n_team) for r, v in jobs)
    print(f"{len(jobs)} jobs in {(time.time() - t0) / 60:.1f} min")
    res = pd.concat([a for a, _ in out], ignore_index=True).assign(git=git_hash())
    bouts = pd.concat([b for _, b in out], ignore_index=True)

    # Baseline check: 'full' must reproduce step 04 (same teams, same RF, same seeds)
    s04 = pd.read_csv(OUT / "04_rf_athletes.csv")
    s04 = s04[(s04.k == 2) & s04.n_team.isin(args.n_team) & (s04.repeat < args.repeats)]
    full = res[res.variant == "full"]
    m = full.merge(s04, on=["repeat", "n_team", "group", "athlete"], suffixes=("", "_04"))
    diff = (m.macro_f1 - m.macro_f1_04).abs().max()
    print(f"full vs step 04: {len(m)} of {len(s04)} rows matched, max |diff| = {diff:.2e}")
    assert len(m) == len(s04) and diff < 1e-9, "baseline does not reproduce step 04"

    if args.verify:
        a, _ = one(0, "motion", args.n_team)
        b = res[(res.repeat == 0) & (res.variant == "motion")].drop(columns="git").reset_index(drop=True)
        pd.testing.assert_frame_equal(a.reset_index(drop=True)[b.columns], b)
        print("verify: motion repeat 0 reproduces exactly")

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "07_rf_athletes.csv", index=False)
    bouts.to_csv(OUT / "07_rf_bouts.csv.gz", index=False)

    # Summary: macro-F1 by variant x N x group, gap (recorded - new), change vs full (paired by athlete)
    ath = res.groupby(["variant", "n_team", "group", "athlete"]).macro_f1.mean()
    rows = []
    for n in args.n_team:
        for v in VARIANTS:
            row = dict(n_team=n, variant=v)
            for g in ("new", "recorded"):
                x = ath.loc[(v, n, g)]
                row[f"{g}_f1"], row[f"{g}_lo"], row[f"{g}_hi"] = bootstrap_ci(x)
                d = x - ath.loc[("full", n, g)]
                row[f"{g}_vs_full"], row[f"{g}_vs_full_lo"], row[f"{g}_vs_full_hi"] = bootstrap_ci(d.dropna())
            gap = (ath.loc[(v, n, "recorded")] - ath.loc[(v, n, "new")]).dropna()  # same athlete, both roles
            row["gap"], row["gap_lo"], row["gap_hi"] = bootstrap_ci(gap)
            b = bouts[(bouts.variant == v) & (bouts.n_team == n) & bouts.true.isin(PRESSES)]
            row["press_recall_new"] = (b[b.group == "new"].true == b[b.group == "new"].pred).mean()
            rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "07_rf_summary.csv", index=False)
    cols = ["n_team", "variant", "new_f1", "new_vs_full", "recorded_f1", "gap", "gap_lo", "gap_hi", "press_recall_new"]
    print(s[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
