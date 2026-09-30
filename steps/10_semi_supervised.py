"""Step 10: semi-supervised learning. Can unlabeled wear time substitute for labeled athletes?

Random forest, same repeats / team order / new athletes as steps 04-09. Labeled: first N team candidates
(N = 2, 4, 8) x 1 set of every exercise (day 1). Conditions:
  supervised        labeled only (must reproduce step 04, k=1)
  ssl_team          + unlabeled day-1 wear of the other 17-N candidates, self-training (src/selftrain.py)
  ssl_team_new      + also the 6 new athletes' own unlabeled day-1 wear (secondary)
  ceiling           all 17 candidates LABELED x 1 set (= step 04, N=17, k=1)
Scored on the new athletes' later days (primary) and the labeled team's later days.
Key quantity: fraction of the supervised -> ceiling gap closed by unlabeled wear.

    python steps/10_semi_supervised.py [--repeats 30] [--jobs 12] [--verify]
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
from src.experiment import repeat_plan, train_ids  # noqa: E402
from src.metrics import bootstrap_ci, per_athlete, vote_bouts  # noqa: E402
from src.selftrain import _fit, self_train  # noqa: E402
from src.team import assert_no_leak, load_manifest, test_bouts  # noqa: E402
from src.utils import git_hash  # noqa: E402

OUT = ROOT / "outputs" / "tables"
LABELED_N = (2, 4, 8)


def low_priority():
    if os.name == "nt":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        if not k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000):
            print("warning: could not lower process priority")


def one(r):
    X, F, meta = get_windows()
    y = meta.cls.to_numpy()
    seg = meta.segment_id.to_numpy()
    man = load_manifest()
    plan = repeat_plan(r, man)
    te_new = test_bouts(man, plan["new"])
    day1 = man[man.day_order == 1]
    rows, diags = [], []

    def score(model, name, n, team):
        te_rec = test_bouts(man, team)
        tem = np.isin(seg, te_rec + te_new)
        b = vote_bouts(model.predict_proba(F[tem]), meta[tem])
        b["group"] = np.where(b.athlete.isin(plan["new"]), "new", "recorded")
        for g, bg in b.groupby("group"):
            rows.append(per_athlete(bg).assign(group=g, repeat=r, n_labeled=n, condition=name))

    team_all, tr_all = train_ids(plan, len(plan["order"]), 1)
    score(_fit(F, y, np.isin(seg, tr_all), r, C.RF_TREES), "ceiling", len(team_all), team_all)
    for n in LABELED_N:
        team, tr = train_ids(plan, n, 1)
        assert_no_leak(man, tr, test_bouts(man, team) + te_new, plan["new"])
        lab = np.isin(seg, tr)
        score(_fit(F, y, lab, r, C.RF_TREES), "supervised", n, team)
        others = plan["order"][n:]
        unl_team = np.isin(seg, day1[day1.participant_id.isin(others)].segment_id)
        unl_new = np.isin(seg, day1[day1.participant_id.isin(plan["new"])].segment_id)
        assert not (unl_team & lab).any() and not np.isin(seg[unl_new], te_new).any()
        for name, unl in (("ssl_team", unl_team), ("ssl_team_new", unl_team | unl_new)):
            model, d = self_train(F, y, seg, lab, unl, seed=r, trees=C.RF_TREES)
            score(model, name, n, team)
            diags += [dict(repeat=r, n_labeled=n, condition=name, **x) for x in d]
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diags)


def paired(a, b):
    d = (a - b).dropna()
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
    out = Parallel(n_jobs=args.jobs, verbose=0)(delayed(one)(r) for r in range(args.repeats))
    print(f"{args.repeats} repeats in {(time.time() - t0) / 60:.1f} min on {args.jobs} workers")
    res = pd.concat([a for a, _ in out], ignore_index=True).assign(git=git_hash())
    diag = pd.concat([d for _, d in out], ignore_index=True)

    # Baselines must reproduce step 04 (k=1): supervised at N and ceiling at N=17
    s04 = pd.read_csv(OUT / "04_rf_athletes.csv")
    s04 = s04[(s04.k == 1) & (s04.repeat < args.repeats)]
    for cond, ns in (("supervised", LABELED_N), ("ceiling", (17,))):
        a = res[res.condition == cond].rename(columns={"n_labeled": "n_team"})
        m = a.merge(s04[s04.n_team.isin(ns)], on=["repeat", "n_team", "group", "athlete"], suffixes=("", "_04"))
        dmax = (m.macro_f1 - m.macro_f1_04).abs().max()
        print(f"{cond} vs step 04: {len(m)} rows matched of {len(a)}, max |diff| = {dmax:.1e}")
        assert len(m) == len(a) and dmax < 1e-9

    if args.verify:
        a, _ = one(0)
        b = res[res.repeat == 0].drop(columns="git").reset_index(drop=True)
        pd.testing.assert_frame_equal(a.reset_index(drop=True)[b.columns], b)
        print("verify: repeat 0 reproduces exactly")

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "10_ssl_athletes.csv", index=False)
    diag.to_csv(OUT / "10_ssl_pseudolabels.csv", index=False)

    ath = res[res.group == "new"].groupby(["condition", "n_labeled", "athlete"]).macro_f1.mean()
    ceil = ath.loc["ceiling"].droplevel(0)
    rows = []
    for n in LABELED_N:
        sup = ath.loc[("supervised", n)]
        for c in ("ssl_team", "ssl_team_new"):
            s = ath.loc[(c, n)]
            gap = (ceil - sup).mean()
            rows.append(dict(n_labeled=n, condition=c, supervised=sup.mean(), ssl=s.mean(), ceiling_17_labeled=ceil.mean(),
                             gap_closed=(s.mean() - sup.mean()) / gap if gap > 0 else np.nan, **paired(s, sup)))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "10_ssl_summary.csv", index=False)
    print(s.round(4).to_string(index=False))
    print("\nPseudo-label accuracy (sets kept / all unlabeled sets), mean over repeats:")
    print(diag.groupby(["n_labeled", "condition", "round"])[["pseudo_acc_kept", "pseudo_acc_all", "n_pseudo_sets"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
