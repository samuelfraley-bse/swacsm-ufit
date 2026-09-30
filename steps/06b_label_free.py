"""Step 06b: label-free onboarding. The new athlete just wears the watch for one day (no labels).

Same repeats, new athletes and team order as steps 04-06; team uses k=2 sets per exercise. The team
model is trained exactly as in step 05. For each new athlete, conditions:
  none            team model as is
  input_norm      inputs standardized with the athlete's own unlabeled day-1 statistics
  adabn           BatchNorm statistics recomputed on the athlete's unlabeled day-1 windows (CNN only)
  adabn_input     both
  adabn_testdays  secondary: BatchNorm statistics from the unlabeled test-day windows themselves
                  (on-the-fly adaptation during use; CNN only)
Scored on the athlete's later-day bouts, all 15 classes (macro-F1).

    python steps/06b_label_free.py --model cnn [--n-team 2 8 17] [--repeats 30] [--verify]
"""
import argparse
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

from src.data import get_windows  # noqa: E402
from src.experiment import repeat_plan, train_ids  # noqa: E402
from src.metrics import bootstrap_ci, per_athlete, vote_bouts  # noqa: E402
from src.team import load_manifest, test_bouts  # noqa: E402
from src.utils import git_hash  # noqa: E402

OUT = ROOT / "outputs" / "tables"
K_TEAM = 2


def one(r, arch, team_sizes):
    import torch

    torch.set_num_threads(1)
    from src.adapt import adabn, athlete_stats, has_batchnorm
    from src.models import predict, train_model

    X, F, meta = get_windows()
    y = meta.cls.to_numpy()
    seg = meta.segment_id.to_numpy()
    man = load_manifest()
    plan = repeat_plan(r, man)
    rows = []
    for n_team in team_sizes:
        _, tr = train_ids(plan, n_team, K_TEAM)
        trm = np.isin(seg, tr)
        model, mu, sd = train_model(arch, X[trm], y[trm], seed=r)
        bn = has_batchnorm(model)
        for a in plan["new"]:
            day1 = man[(man.participant_id == a) & (man.day_order == 1)].segment_id
            d1m = np.isin(seg, day1)
            tem = np.isin(seg, test_bouts(man, [a]))
            Xu, Xt = X[d1m], X[tem]  # Xu: unlabeled day-1 windows (labels never used)
            mu_a, sd_a = athlete_stats(Xu)
            conds = {"none": (model, mu, sd), "input_norm": (model, mu_a, sd_a)}
            if bn:
                conds["adabn"] = (adabn(model, mu, sd, Xu), mu, sd)
                conds["adabn_input"] = (adabn(model, mu_a, sd_a, Xu), mu_a, sd_a)
                conds["adabn_testdays"] = (adabn(model, mu, sd, Xt), mu, sd)
            for name, (m, mu_c, sd_c) in conds.items():
                bouts = vote_bouts(predict(m, mu_c, sd_c, Xt), meta[tem])
                rows.append(per_athlete(bouts).assign(model=arch, repeat=r, n_team=n_team, condition=name,
                                                      n_unlabeled_windows=int(d1m.sum())))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "lstm"], required=True)
    ap.add_argument("--n-team", type=int, nargs="+", default=[2, 8, 17])
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    get_windows()

    t0 = time.time()
    out = Parallel(n_jobs=min(args.repeats, os.cpu_count()), verbose=5)(
        delayed(one)(r, args.model, args.n_team) for r in range(args.repeats))
    print(f"{args.repeats} repeats in {(time.time() - t0) / 60:.1f} min")
    res = pd.concat(out, ignore_index=True).assign(git=git_hash())

    if args.verify:
        again = one(0, args.model, args.n_team)
        a = res[res.repeat == 0].drop(columns="git").reset_index(drop=True)
        pd.testing.assert_frame_equal(again.reset_index(drop=True)[a.columns], a)
        print("verify: repeat 0 reproduces exactly")

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / f"06b_{args.model}_athletes.csv", index=False)
    ath = res.groupby(["n_team", "condition", "athlete"]).macro_f1.mean().unstack("condition")
    lines = []
    for n, g in ath.groupby(level="n_team"):
        for c in g.columns:
            m, lo, hi = bootstrap_ci(g[c])
            dm, dlo, dhi = bootstrap_ci(g[c] - g["none"])
            lines.append(dict(n_team=n, condition=c, macro_f1=m, ci_lo=lo, ci_hi=hi,
                              gain_vs_none=dm, gain_lo=dlo, gain_hi=dhi, n_athletes=len(g)))
    s = pd.DataFrame(lines)
    s.to_csv(OUT / f"06b_{args.model}_summary.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
