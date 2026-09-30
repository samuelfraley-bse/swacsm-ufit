"""Step 16: simpler models, and what level of detail can be trusted.

Same repeats / teams / never-recorded athletes as steps 04-15.
  rules    coach's-rules baseline (src/coachrules.py): typical profile per exercise, set-level, no ML
  logreg   logistic regression on the same 59 window features as the RF (standardized on training windows,
           balanced class weights, C = 1, no tuning); set call = majority vote of windows, as for the RF/CNN
Grids: the main grid (N = 2..17 x k = 1, 2; recorded and new athletes) and the step 12 few-vs-many grid.
Categories: every model's exact-lift calls collapsed into body region / movement pattern / push-pull
(mapping fixed below before looking at results); accuracy per level, new athletes, 1 set per exercise.
Floor for reference: random guessing among 15 exercises = 1/15 = 0.067.

    python steps/16_baselines.py [--repeats 30] [--jobs 24] [--verify]
"""
import argparse
import ctypes
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from src.coachrules import CoachRules, set_measures  # noqa: E402
from src.data import get_windows  # noqa: E402
from src.experiment import KS, TEAM_SIZES, repeat_plan, run_repeat, train_ids, train_ids_depth  # noqa: E402
from src.metrics import bootstrap_ci, per_athlete, vote_bouts  # noqa: E402
from src.summary import summarize  # noqa: E402
from src.team import assert_no_leak, load_manifest, test_bouts  # noqa: E402
from src.utils import git_hash  # noqa: E402

OUT = ROOT / "outputs" / "tables"
BUDGETS = {4: [(1, 4), (2, 2), (4, 1)], 8: [(2, 4), (4, 2), (8, 1)], 16: [(4, 4), (8, 2), (16, 1)]}
# Category mapping (fixed in advance; judgment calls: push press = upper push, lunge = squat pattern, curl = pull)
REGION = {"lower": [0, 8, 2, 6, 5], "upper": [1, 4, 10, 7, 9, 14, 11], "full body": [3, 12], "core": [13]}
PATTERN = {"squat": [0, 8, 2], "hinge": [6, 5], "upper push": [1, 4, 10, 7], "upper pull": [11, 9],
           "shoulder isolation": [14], "conditioning": [3, 12], "core": [13]}
PUSHPULL = {"push": [1, 4, 10, 7], "pull": [11, 9]}  # evaluated on push/pull sets only
MEASURES_FILE = OUT / "16_set_measures.csv"


def low_priority():
    if os.name == "nt":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        if not k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000):
            print("warning: could not lower process priority")


def logreg_fit_predict(F, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from threadpoolctl import threadpool_limits

    def fp(trm, tem, seed):
        # single-threaded linear algebra so results do not depend on the thread count (reproducibility)
        with threadpool_limits(1), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc = StandardScaler().fit(F[trm])
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
            lr.fit(sc.transform(F[trm]), y[trm])
            assert list(lr.classes_) == list(range(15))
            return lr.predict_proba(sc.transform(F[tem]))
    return fp


def rules_sets(M, ids_train, ids_test):
    rules = CoachRules().fit(M.loc[ids_train])
    t = M.loc[ids_test]
    return pd.DataFrame(dict(segment_id=t.index, athlete=t.athlete.to_numpy(), true=t.cls.to_numpy(), pred=rules.predict(t)))


def one(r):
    X, F, meta = get_windows()
    y = meta.cls.to_numpy()
    seg = meta.segment_id.to_numpy()
    man = load_manifest()
    M = pd.read_csv(MEASURES_FILE).set_index("segment_id")
    plan = repeat_plan(r, man)
    te_new = test_bouts(man, plan["new"])
    fp = logreg_fit_predict(F, y)

    # main grid: logistic regression (shared loop), coach's rules (set level)
    lr_res, lr_bouts = run_repeat(r, fp, X, F, meta, "logreg")
    rows, bouts = [], []
    for n in TEAM_SIZES:
        for k in KS:
            team, tr = train_ids(plan, n, k)
            te_rec = test_bouts(man, team)
            assert_no_leak(man, tr, te_rec + te_new, plan["new"])
            b = rules_sets(M, tr, te_rec + te_new)
            b["group"] = np.where(b.athlete.isin(plan["new"]), "new", "recorded")
            info = dict(model="rules", repeat=r, n_team=n, k=k, n_train_bouts=len(tr))
            for g, bg in b.groupby("group"):
                rows.append(per_athlete(bg).assign(group=g, **info))
            bouts.append(b.assign(**info))
    rules_res, rules_bouts = pd.concat(rows, ignore_index=True), pd.concat(bouts, ignore_index=True)

    # few-vs-many grid (new athletes only)
    db = []
    tem = np.isin(seg, te_new)
    for budget, splits in BUDGETS.items():
        for n, d in splits:
            team, tr = train_ids_depth(plan, man, n, d)
            assert not set(man[man.segment_id.isin(tr)].participant_id) & set(plan["new"])
            b_lr = vote_bouts(fp(np.isin(seg, tr), tem, r), meta[tem])
            b_ru = rules_sets(M, tr, te_new)
            for name, b in (("logreg", b_lr), ("rules", b_ru)):
                db.append(per_athlete(b).assign(model=name, repeat=r, budget=budget, n_team=n, depth=d, group="new"))
    return (pd.concat([lr_res, rules_res], ignore_index=True), pd.concat([lr_bouts, rules_bouts], ignore_index=True),
            pd.concat(db, ignore_index=True))


def category_table():
    """Accuracy per level of detail, new athletes, k=1, for every model (athlete-level mean, 95% CI)."""
    def to_cat(v, mapping):
        lut = {c: name for name, cs in mapping.items() for c in cs}
        return v.map(lut)

    files = {"rf": "04_rf_bouts.csv.gz", "cnn": "05_cnn_bouts.csv.gz", "logreg": "16_baselines_bouts.csv.gz",
             "rules": "16_baselines_bouts.csv.gz"}
    rows = []
    for m, f in files.items():
        b = pd.read_csv(OUT / f)
        if "model" in b.columns:
            b = b[b.model == m] if m in ("logreg", "rules") else b
        b = b[(b.group == "new") & (b.k == 1)]
        levels = {"exact lift": (b, b.true == b.pred)}
        for lvl, mp in (("body region", REGION), ("movement pattern", PATTERN)):
            levels[lvl] = (b, to_cat(b.true, mp) == to_cat(b.pred, mp))
        pp = b[b.true.isin(sum(PUSHPULL.values(), []))]
        levels["push vs pull"] = (pp, to_cat(pp.true, PUSHPULL) == to_cat(pp.pred, PUSHPULL).fillna("other"))
        for lvl, (d, ok) in levels.items():
            acc = d.assign(ok=ok.to_numpy()).groupby(["n_team", "athlete"]).ok.mean().groupby(level=[0, 1]).mean()
            for n, g in acc.groupby(level=0):
                mm, lo, hi = bootstrap_ci(g.to_numpy())
                rows.append(dict(model=m, level=lvl, n_team=n, accuracy=mm, ci_lo=lo, ci_hi=hi, n_athletes=len(g)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    low_priority()
    get_windows()
    from src.load import load_all

    seg, signals = load_all()
    set_measures(seg, signals).to_csv(MEASURES_FILE, index=False)

    t0 = time.time()
    out = Parallel(n_jobs=args.jobs, verbose=0)(delayed(one)(r) for r in range(args.repeats))
    print(f"{args.repeats} repeats in {(time.time() - t0) / 60:.1f} min on {args.jobs} workers")
    res = pd.concat([a for a, _, _ in out], ignore_index=True).assign(git=git_hash())
    bouts = pd.concat([b for _, b, _ in out], ignore_index=True)
    db = pd.concat([c for _, _, c in out], ignore_index=True)

    if args.verify:
        a, _, c = one(0)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), res[res.repeat == 0].drop(columns="git").reset_index(drop=True))
        pd.testing.assert_frame_equal(c.reset_index(drop=True), db[db.repeat == 0].reset_index(drop=True))
        print("verify: repeat 0 reproduces exactly")

    res.to_csv(OUT / "16_baselines_athletes.csv", index=False)
    bouts.to_csv(OUT / "16_baselines_bouts.csv.gz", index=False)
    db.to_csv(OUT / "16_deep_broad_baselines_athletes.csv", index=False)
    s = summarize(res)
    s.to_csv(OUT / "16_baselines_summary.csv", index=False)

    ref = pd.concat([pd.read_csv(OUT / "04_rf_summary.csv"), pd.read_csv(OUT / "05_cnn_summary.csv")])
    allm = pd.concat([s, ref])
    print("\nMacro-F1, new athletes, 1 set per exercise (chance = 0.067):")
    print(allm[(allm.group == "new") & (allm.k == 1)].pivot(index="n_team", columns="model", values="mean").round(3).to_string())

    ath = db.groupby(["model", "budget", "n_team", "depth", "athlete"]).macro_f1.mean()
    print("\nFew vs many (new athletes): broad - deep")
    for m in ("logreg", "rules"):
        for budget, splits in BUDGETS.items():
            (n_d, d_d), (n_b, d_b) = splits[0], splits[-1]
            diff = (ath.loc[(m, budget, n_b, d_b)] - ath.loc[(m, budget, n_d, d_d)]).dropna()
            mm, lo, hi = bootstrap_ci(diff.to_numpy())
            print(f"  {m:6s} budget {budget:2d}: {n_b}x{d_b} - {n_d}x{d_d} = {mm:+.3f} ({lo:+.3f} to {hi:+.3f}), "
                  f"{int((diff > 0).sum())}/{len(diff)} athletes better")

    cat = category_table()
    cat.to_csv(OUT / "16_categories.csv", index=False)
    print("\nAccuracy by level of detail (new athletes, 1 set per exercise):")
    print(cat[cat.n_team.isin([2, 8, 17])].pivot_table(index=["level", "n_team"], columns="model", values="accuracy").round(3).to_string())


if __name__ == "__main__":
    main()
