"""Step 05: CNN or LSTM on the same new-team grid and the same training/test bouts as step 04.

One process per repeat, torch limited to 1 thread per process.

    python steps/05_deep_repeats.py --model cnn --time-one      # time one repeat, estimate total
    python steps/05_deep_repeats.py --model cnn [--repeats 30] [--verify]
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
from src.data import get_windows  # noqa: E402
from src.experiment import run_repeat  # noqa: E402
from src.summary import summarize  # noqa: E402
from src.utils import git_hash  # noqa: E402

OUT = ROOT / "outputs" / "tables"


def one(r, arch):
    import torch

    torch.set_num_threads(1)
    from src.models import fit_predict

    X, F, meta = get_windows()
    y = meta.cls.to_numpy()

    def fp(trm, tem, seed):
        return fit_predict(arch, X[trm], y[trm], X[tem], seed)

    return run_repeat(r, fp, X, F, meta, arch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "lstm"], required=True)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--time-one", action="store_true")
    args = ap.parse_args()
    get_windows()
    n_workers = min(args.repeats, os.cpu_count())

    if args.time_one:
        t0 = time.time()
        res, _ = one(0, args.model)
        dt = time.time() - t0
        rounds = -(-args.repeats // n_workers)
        print(f"one repeat: {dt:.0f} s single-threaded; {args.repeats} repeats on {n_workers} workers "
              f"~ {rounds * dt / 60:.0f} min (plus some slowdown from parallel load)")
        print(summarize(res).pivot_table(index="n_team", columns=["group", "k"], values="mean").round(3).to_string())
        return

    t0 = time.time()
    out = Parallel(n_jobs=n_workers, verbose=5)(delayed(one)(r, args.model) for r in range(args.repeats))
    print(f"{args.repeats} repeats in {(time.time() - t0) / 60:.1f} min")
    res = pd.concat([a for a, _ in out], ignore_index=True).assign(git=git_hash(), fs=C.FS, window_s=C.WINDOW_S)
    bouts = pd.concat([b for _, b in out], ignore_index=True)

    if args.verify:
        again = Parallel(n_jobs=2)(delayed(one)(r, args.model) for r in range(2))
        a = pd.concat([x for x, _ in again], ignore_index=True)
        b = res[res.repeat < 2].drop(columns=["git", "fs", "window_s"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a.reset_index(drop=True)[b.columns], b)
        print("verify: repeats 0-1 reproduce exactly")

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / f"05_{args.model}_athletes.csv", index=False)
    bouts.to_csv(OUT / f"05_{args.model}_bouts.csv.gz", index=False)
    s = summarize(res)
    s.to_csv(OUT / f"05_{args.model}_summary.csv", index=False)
    print(s.pivot_table(index="n_team", columns=["group", "k"], values="mean").round(3).to_string())


if __name__ == "__main__":
    main()
