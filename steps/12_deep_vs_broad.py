"""Step 12: few athletes recorded many times vs many athletes recorded once, at the same budget.

Same repeats / team order / new athletes as steps 04-11. Training may use recorded athletes' later days
(sets 3-4 come from their next recording day), so only the never-recorded athletes are tested.
Budgets (athlete-sets per exercise) and splits (athletes x sets per exercise):
  4:  1x4, 2x2, 4x1        8:  2x4, 4x2, 8x1        16: 4x4, 8x2, 16x1
Every 1- and 2-set condition is identical to steps 04 (RF) / 05 (CNN) and is checked against them.

    python steps/12_deep_vs_broad.py --model rf|cnn [--repeats 30] [--jobs 24] [--verify]
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
from scipy.stats import wilcoxon  # noqa: E402

from src import config as C  # noqa: E402
from src.data import get_windows  # noqa: E402
from src.experiment import repeat_plan, train_ids_depth  # noqa: E402
from src.metrics import bootstrap_ci, per_athlete, vote_bouts  # noqa: E402
from src.team import load_manifest, test_bouts  # noqa: E402
from src.utils import git_hash  # noqa: E402

OUT = ROOT / "outputs" / "tables"
BUDGETS = {4: [(1, 4), (2, 2), (4, 1)], 8: [(2, 4), (4, 2), (8, 1)], 16: [(4, 4), (8, 2), (16, 1)]}


def low_priority():
    if os.name == "nt":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        if not k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000):
            print("warning: could not lower process priority")


def one(r, model):
    X, F, meta = get_windows()
    y = meta.cls.to_numpy()
    seg = meta.segment_id.to_numpy()
    man = load_manifest()
    plan = repeat_plan(r, man)
    te_new = test_bouts(man, plan["new"])
    tem = np.isin(seg, te_new)
    if model == "cnn":
        import torch

        torch.set_num_threads(1)
        from src.models import fit_predict
    rows = []
    for budget, splits in BUDGETS.items():
        for n, d in splits:
            team, tr = train_ids_depth(plan, man, n, d)
            assert not set(tr) & set(te_new) and not set(man[man.segment_id.isin(tr)].participant_id) & set(plan["new"])
            trm = np.isin(seg, tr)
            if model == "rf":
                from sklearn.ensemble import RandomForestClassifier

                rf = RandomForestClassifier(C.RF_TREES, class_weight="balanced", n_jobs=1, random_state=r)
                proba = rf.fit(F[trm], y[trm]).predict_proba(F[tem])
            else:
                proba = fit_predict("cnn", X[trm], y[trm], X[tem], r)
            b = vote_bouts(proba, meta[tem])
            n_days = man[man.segment_id.isin(tr)].groupby("participant_id").day_order.nunique().max()
            rows.append(per_athlete(b).assign(model=model, repeat=r, budget=budget, n_team=n, depth=d,
                                              n_train_sets=len(tr), max_days_per_athlete=n_days, group="new"))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rf", "cnn"], required=True)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    low_priority()
    get_windows()

    t0 = time.time()
    out = Parallel(n_jobs=args.jobs, verbose=0)(delayed(one)(r, args.model) for r in range(args.repeats))
    print(f"{args.repeats} repeats in {(time.time() - t0) / 60:.1f} min on {args.jobs} workers")
    res = pd.concat(out, ignore_index=True).assign(git=git_hash())

    # 1- and 2-set conditions (on the step 04/05 grid) must reproduce them exactly
    ref = pd.read_csv(OUT / ("04_rf_athletes.csv" if args.model == "rf" else "05_cnn_athletes.csv"))
    ref = ref[(ref.group == "new") & (ref.repeat < args.repeats)].rename(columns={"k": "depth"})
    a = res[(res.depth <= 2) & res.n_team.isin(ref.n_team.unique())]
    m = a.merge(ref, on=["repeat", "n_team", "depth", "athlete"], suffixes=("", "_ref"))
    dmax = (m.macro_f1 - m.macro_f1_ref).abs().max()
    print(f"depth<=2 vs step {'04' if args.model == 'rf' else '05'}: {len(m)} of {len(a)} rows matched, max |diff| = {dmax:.1e}")
    assert len(m) == len(a) and dmax < 1e-9
    print("sets per training run:", res.drop_duplicates(["budget", "n_team", "depth"]).set_index(["budget", "n_team", "depth"]).n_train_sets.to_dict())

    if args.verify:
        v = one(0, args.model)
        b = res[res.repeat == 0].drop(columns="git").reset_index(drop=True)
        pd.testing.assert_frame_equal(v.reset_index(drop=True)[b.columns], b)
        print("verify: repeat 0 reproduces exactly")

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / f"12_deep_broad_{args.model}_athletes.csv", index=False)

    ath = res.groupby(["budget", "n_team", "depth", "athlete"]).macro_f1.mean()
    rows = []
    for budget, splits in BUDGETS.items():
        s = {f"{n}x{d}": ath.loc[(budget, n, d)] for n, d in splits}
        names = list(s)  # deep, middle, broad
        for x, yv in ((names[2], names[0]), (names[2], names[1]), (names[1], names[0])):
            dd = (s[x] - s[yv]).dropna()
            mm, lo, hi = bootstrap_ci(dd.to_numpy())
            rows.append(dict(model=args.model, budget=budget, comparison=f"{x} - {yv}", mean_x=s[x].mean(), mean_y=s[yv].mean(),
                             diff=mm, ci_lo=lo, ci_hi=hi, p=wilcoxon(dd).pvalue if (dd != 0).any() else 1.0,
                             n_positive=int((dd > 0).sum()), n=len(dd)))
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / f"12_deep_broad_{args.model}_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
