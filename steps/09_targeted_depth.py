"""Step 09: targeted depth. Spend extra recording on the exercises the tracker confuses, or on more athletes?

Random forest, same repeats / team order / new athletes as steps 04-08. At team size N (5 and 10):
  base      N athletes x 1 set of every exercise                         15N sets
  press     base + a 2nd set of bench, push and military press            18N sets
  random3   base + a 2nd set of 3 random non-press exercises (control)    18N sets
  broad     1.2N athletes x 1 set (6 and 12)                              18N sets
  k2        N athletes x 2 sets of everything (reference)                 30N sets
Scored on the new (never-recorded) athletes' later days; recorded athletes are also saved.
Note: the presses were identified as hard in earlier descriptive analyses that included all athletes.

    python steps/09_targeted_depth.py [--repeats 30] [--jobs 12] [--verify]
"""
import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from src import config as C  # noqa: E402
from src.data import get_windows  # noqa: E402
from src.experiment import repeat_plan, train_ids, train_ids_extra  # noqa: E402
from src.metrics import bootstrap_ci, per_athlete, vote_bouts  # noqa: E402
from src.team import assert_no_leak, load_manifest, test_bouts  # noqa: E402
from src.utils import git_hash  # noqa: E402

OUT = ROOT / "outputs" / "tables"
PRESSES = (4, 7, 10)
TEAM_SIZES = (5, 10)


def low_priority():
    """Windows: below-normal priority; worker processes started afterwards inherit it."""
    if os.name == "nt":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p  # HANDLE; the default int return type truncates it
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        if not k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000):
            print("warning: could not lower process priority")


def conditions(plan, n, r):
    rng = np.random.default_rng([r, 909, n])
    rand3 = tuple(sorted(int(c) for c in rng.choice([c for c in range(15) if c not in PRESSES], 3, replace=False)))
    return {
        "base": train_ids(plan, n, 1),
        "press": train_ids_extra(plan, n, PRESSES),
        "random3": train_ids_extra(plan, n, rand3),
        "broad": train_ids(plan, int(round(1.2 * n)), 1),
        "k2": train_ids(plan, n, 2),
    }, rand3


def one(r):
    X, F, meta = get_windows()
    y = meta.cls.to_numpy()
    seg = meta.segment_id.to_numpy()
    man = load_manifest()
    plan = repeat_plan(r, man)
    te_new = test_bouts(man, plan["new"])
    rows, bouts_all = [], []
    for n in TEAM_SIZES:
        conds, rand3 = conditions(plan, n, r)
        for name, (team, tr) in conds.items():
            te_rec = test_bouts(man, team)
            assert_no_leak(man, tr, te_rec + te_new, plan["new"])
            trm, tem = np.isin(seg, tr), np.isin(seg, te_rec + te_new)
            rf = RandomForestClassifier(C.RF_TREES, class_weight="balanced", n_jobs=1, random_state=r)
            rf.fit(F[trm], y[trm])
            b = vote_bouts(rf.predict_proba(F[tem]), meta[tem])
            b["group"] = np.where(b.athlete.isin(plan["new"]), "new", "recorded")
            info = dict(repeat=r, n_team=n, condition=name, n_athletes_recorded=len(team), n_train_sets=len(tr),
                        random3=",".join(map(str, rand3)))
            for g, bg in b.groupby("group"):
                pa = per_athlete(bg)
                press = bg[bg.true.isin(PRESSES)].assign(ok=lambda d: d.true == d.pred).groupby("athlete").ok.mean()
                rows.append(pa.assign(press_recall=pa.athlete.map(press), group=g, **info))
            bouts_all.append(b.assign(**info))
    return pd.concat(rows, ignore_index=True), pd.concat(bouts_all, ignore_index=True)


def paired(a, b):
    d = (a - b).dropna()
    from scipy.stats import wilcoxon

    m, lo, hi = bootstrap_ci(d.to_numpy())
    return dict(diff=m, ci_lo=lo, ci_hi=hi, p=wilcoxon(d).pvalue if (d != 0).any() else 1.0, n_positive=int((d > 0).sum()), n=len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    low_priority()
    get_windows()

    t0 = time.time()
    out = Parallel(n_jobs=args.jobs, verbose=5)(delayed(one)(r) for r in range(args.repeats))
    print(f"{args.repeats} repeats in {(time.time() - t0) / 60:.1f} min on {args.jobs} workers")
    res = pd.concat([a for a, _ in out], ignore_index=True).assign(git=git_hash())
    bouts = pd.concat([b for _, b in out], ignore_index=True)

    # Budget check: press, random3 and broad must use the same number of training sets
    budget = res.drop_duplicates(["repeat", "n_team", "condition"]).groupby(["n_team", "condition"]).n_train_sets.unique()
    print("training sets per condition:", budget.to_dict())

    if args.verify:
        a, _ = one(0)
        b = res[res.repeat == 0].drop(columns="git").reset_index(drop=True)
        pd.testing.assert_frame_equal(a.reset_index(drop=True)[b.columns], b)
        print("verify: repeat 0 reproduces exactly")

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "09_targeted_athletes.csv", index=False)
    bouts.to_csv(OUT / "09_targeted_bouts.csv.gz", index=False)

    new = res[res.group == "new"]
    ath = new.groupby(["n_team", "condition", "athlete"])[["macro_f1", "press_recall"]].mean()
    rows = []
    for n in TEAM_SIZES:
        a = ath.loc[n]
        for metric in ("macro_f1", "press_recall"):
            m = {c: a.loc[c][metric] for c in ("base", "press", "random3", "broad", "k2")}
            for x, y in (("press", "base"), ("random3", "base"), ("press", "random3"), ("broad", "press"), ("broad", "base")):
                rows.append(dict(n_team=n, metric=metric, comparison=f"{x} - {y}", mean_x=m[x].mean(), mean_y=m[y].mean(),
                                 **paired(m[x], m[y])))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "09_targeted_summary.csv", index=False)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
