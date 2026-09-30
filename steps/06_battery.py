"""Step 06: battery study. Record a new athlete on 2 movements only; does it help their OTHER exercises?

Same repeats, new athletes and team order as steps 04-05 (team: k=2 sets per exercise). Batteries
fixed in advance: squat+deadlift, push-up+push press, and 2 random pairs per athlete per repeat.
Reference: the athlete recorded on all 15 exercises. See src/battery.py for the scoring rules.
RF retrains from scratch per condition; CNN/LSTM train the team model once and fine-tune it
(150 steps, lr 1e-4, class-balanced batches; within a class the athlete recorded, 50% of examples
come from the athlete; 'none' fine-tunes on replayed team windows only).

    python steps/06_battery.py --model rf --n-team 8 17
    python steps/06_battery.py --model cnn --n-team 17
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

from src import config as C  # noqa: E402
from src.battery import FinetuneLearner, RetrainLearner, run_battery_repeat, score  # noqa: E402
from src.data import get_windows  # noqa: E402
from src.metrics import bootstrap_ci  # noqa: E402
from src.utils import git_hash  # noqa: E402

OUT = ROOT / "outputs" / "tables"


def one(r, model, team_sizes):
    X, F, meta = get_windows()
    y = meta.cls.to_numpy()
    if model == "rf":
        learner = RetrainLearner(F, y, C.RF_TREES)
    else:
        import torch

        torch.set_num_threads(1)
        learner = FinetuneLearner(model, X, y)
    return run_battery_repeat(r, learner, meta, model, team_sizes)


def report(sc):
    sc = sc.assign(kind=sc.battery.where(~sc.battery.str.startswith("random"), "random pair"))
    lines = []
    for (n, kind), g in sc.groupby(["n_team", "kind"]):
        a = g.groupby("athlete")[["f1_none", "f1_battery", "f1_all15", "pull_none", "pull_battery"]].mean()
        gain, lo, hi = bootstrap_ci(a.f1_battery - a.f1_none)
        full = (a.f1_all15 - a.f1_none).mean()
        lines.append(dict(n_team=n, battery=kind, f1_none=a.f1_none.mean(), f1_battery=a.f1_battery.mean(),
                          f1_all15=a.f1_all15.mean(), gain=gain, gain_lo=lo, gain_hi=hi,
                          gap_closed=gain / full if full > 0 else float("nan"),
                          pull_none=a.pull_none.mean(), pull_battery=a.pull_battery.mean(), n_athletes=len(a)))
    return pd.DataFrame(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rf", "cnn", "lstm"], required=True)
    ap.add_argument("--n-team", type=int, nargs="+", default=[17])
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    get_windows()

    t0 = time.time()
    out = Parallel(n_jobs=min(args.repeats, os.cpu_count()), verbose=5)(
        delayed(one)(r, args.model, args.n_team) for r in range(args.repeats))
    print(f"{args.repeats} repeats in {(time.time() - t0) / 60:.1f} min")
    preds = pd.concat(out, ignore_index=True)

    if args.verify:
        again = one(0, args.model, args.n_team)
        a = preds[preds.repeat == 0].reset_index(drop=True)
        pd.testing.assert_frame_equal(again.reset_index(drop=True)[a.columns], a)
        print("verify: repeat 0 reproduces exactly")

    sc = score(preds).assign(git=git_hash())
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"06_battery_{args.model}"
    preds.to_csv(OUT / f"{tag}_bouts.csv.gz", index=False)
    sc.to_csv(OUT / f"{tag}_scores.csv", index=False)
    rep = report(sc)
    rep.to_csv(OUT / f"{tag}_summary.csv", index=False)
    print(rep.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
