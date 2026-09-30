"""Step 13: real-time recognition. How many seconds / reps into a set before the tracker is right?

Same repeats / team order / new athletes as steps 04-12; budget 1 set per exercise (k=1); N = 2, 4, 8, 17.
For each never-recorded athlete's later-day sets, the running call at t = 2, 3, 4, 6, 8, 10, 15 s and at the
end of the set (src/realtime.py). Reps ~ t x the set's own rep rate. Also the time to a stable correct call.
The end-of-set result must reproduce steps 04 (RF) / 05 (CNN) exactly.

    python steps/13_realtime.py --model rf|cnn [--repeats 30] [--jobs 24] [--verify]
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
from sklearn.metrics import f1_score  # noqa: E402

from src import config as C  # noqa: E402
from src.data import get_windows  # noqa: E402
from src.experiment import repeat_plan, train_ids  # noqa: E402
from src.metrics import LABELS, bootstrap_ci  # noqa: E402
from src.realtime import TIMES_S, calls_at_times, running_calls  # noqa: E402
from src.team import load_manifest, test_bouts  # noqa: E402
from src.utils import git_hash  # noqa: E402

OUT = ROOT / "outputs" / "tables"
TEAM_SIZES = (2, 4, 8, 17)


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
    tem = np.isin(seg, test_bouts(man, plan["new"]))
    if model == "cnn":
        import torch

        torch.set_num_threads(1)
        from src.models import fit_predict
    out = []
    for n in TEAM_SIZES:
        _, tr = train_ids(plan, n, 1)
        trm = np.isin(seg, tr)
        if model == "rf":
            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(C.RF_TREES, class_weight="balanced", n_jobs=1, random_state=r)
            proba = rf.fit(F[trm], y[trm]).predict_proba(F[tem])
        else:
            proba = fit_predict("cnn", X[trm], y[trm], X[tem], r)
        out.append(calls_at_times(running_calls(proba, meta[tem])).assign(model=model, repeat=r, n_team=n))
    return pd.concat(out, ignore_index=True)


def per_athlete_t(sets):
    rows = []
    for (rep, n, t, a), g in sets.groupby(["repeat", "n_team", "t_s", "athlete"]):
        rows.append(dict(repeat=rep, n_team=n, t_s=t, athlete=a, accuracy=(g.call == g.true).mean(),
                         macro_f1=f1_score(g.true, g.call, labels=LABELS, average="macro", zero_division=0)))
    return pd.DataFrame(rows)


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
    sets = pd.concat(out, ignore_index=True)
    ath = per_athlete_t(sets)

    ref = pd.read_csv(OUT / ("04_rf_athletes.csv" if args.model == "rf" else "05_cnn_athletes.csv"))
    ref = ref[(ref.k == 1) & (ref.group == "new") & (ref.repeat < args.repeats) & ref.n_team.isin(TEAM_SIZES)]
    end = ath[np.isinf(ath.t_s)]
    m = end.merge(ref, on=["repeat", "n_team", "athlete"], suffixes=("", "_ref"))
    dmax = (m.macro_f1 - m.macro_f1_ref).abs().max()
    print(f"end of set vs step {'04' if args.model == 'rf' else '05'}: {len(m)} of {len(end)} rows matched, max |diff| = {dmax:.1e}")
    assert len(m) == len(end) and dmax < 1e-9

    if args.verify:
        v = one(0, args.model)
        pd.testing.assert_frame_equal(v.reset_index(drop=True), sets[sets.repeat == 0].reset_index(drop=True))
        print("verify: repeat 0 reproduces exactly")

    OUT.mkdir(parents=True, exist_ok=True)
    sets.assign(git=git_hash()).to_csv(OUT / f"13_realtime_{args.model}_sets.csv.gz", index=False)
    ath.to_csv(OUT / f"13_realtime_{args.model}_athletes.csv", index=False)

    # Summary: accuracy by time (athlete-level mean, 95% CI) and time to a stable correct call
    a2 = ath.groupby(["n_team", "t_s", "athlete"]).accuracy.mean()
    rows = []
    for n in TEAM_SIZES:
        for t in TIMES_S + [np.inf]:
            mm, lo, hi = bootstrap_ci(a2.loc[(n, t)].to_numpy())
            rows.append(dict(model=args.model, n_team=n, t_s=t, accuracy=mm, ci_lo=lo, ci_hi=hi))
    s = pd.DataFrame(rows)
    fin = sets[np.isinf(sets.t_s)]
    st = fin.assign(stable_reps=fin.stable_s * fin.rep_rate_hz).groupby("n_team").agg(
        pct_sets_ever_stable_correct=("stable_s", lambda v: np.isfinite(v).mean()),
        median_stable_s=("stable_s", lambda v: np.median(v[np.isfinite(v)])),
        median_stable_reps=("stable_reps", lambda v: np.median(v[np.isfinite(v)])),
        median_set_s=("set_duration_s", "median"))
    s.to_csv(OUT / f"13_realtime_{args.model}_summary.csv", index=False)
    st.to_csv(OUT / f"13_realtime_{args.model}_stable.csv")
    print(s.pivot(index="n_team", columns="t_s", values="accuracy").round(3).to_string())
    print(st.round(2).to_string())


if __name__ == "__main__":
    main()
