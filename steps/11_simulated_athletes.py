"""Step 11: simulated athletes (augmentation). Can simulated tempo / intensity / strap-placement variation
stand in for recording real athletes?

Same repeats / team order / new athletes as steps 04-10; budget 1 set per exercise (k=1); N = 2, 4, 8, 17.
Each recorded training set gets N_COPIES simulated copies (src/augment.py). Arms:
  none              real sets only (must reproduce step 04 RF / step 05 CNN, k=1)
  tempo_intensity   + simulated copies varying tempo and movement intensity
  plus_placement    + the same, plus a random strap rotation (up to 15 degrees)
The CNN gets the same number of gradient steps in every arm (epochs scaled by 1/(1+N_COPIES)), so any gain
comes from the simulated data, not from extra training. The RF keeps 300 trees.
Scored on the new athletes' later days; also expressed as "worth X extra real athletes" by interpolating on
the real k=1 learning curve (steps 04/05).

    python steps/11_simulated_athletes.py --model rf|cnn [--repeats 30] [--jobs 12] [--verify]
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
from src.team import assert_no_leak, load_manifest, test_bouts  # noqa: E402
from src.utils import git_hash  # noqa: E402

OUT = ROOT / "outputs" / "tables"
TEAM_SIZES = (2, 4, 8, 17)
N_COPIES = 4
ARMS = {"none": None, "tempo_intensity": False, "plus_placement": True}  # value = rotate?
CURVE_N = [2, 4, 6, 8, 12, 16, 17]


def low_priority():
    if os.name == "nt":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        if not k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000):
            print("warning: could not lower process priority")


def one(r, model):
    from src.augment import simulated_windows
    from src.features import extract
    from src.load import load_all

    X, F, meta = get_windows()
    y = meta.cls.to_numpy()
    seg = meta.segment_id.to_numpy()
    segdf, signals = load_all()
    sig_by_id = dict(zip(segdf.segment_id, signals))
    man = load_manifest()
    plan = repeat_plan(r, man)
    te_new = test_bouts(man, plan["new"])
    if model == "cnn":
        import torch

        torch.set_num_threads(1)
        from src.models import EPOCHS, fit_predict

    rows = []
    for n in TEAM_SIZES:
        team, tr = train_ids(plan, n, 1)
        te_rec = test_bouts(man, team)
        assert_no_leak(man, tr, te_rec + te_new, plan["new"])
        trm, tem = np.isin(seg, tr), np.isin(seg, te_rec + te_new)
        for arm, rotate in ARMS.items():
            if rotate is None:
                Xa, ya = X[:0], y[:0]
            else:
                Xa, ya = simulated_windows(segdf, sig_by_id, tr, N_COPIES, seed=[r, n, 1111, int(rotate)], rotate=rotate)
            if model == "rf":
                from sklearn.ensemble import RandomForestClassifier

                Ftr = np.vstack([F[trm], extract(Xa)]) if len(Xa) else F[trm]
                rf = RandomForestClassifier(C.RF_TREES, class_weight="balanced", n_jobs=1, random_state=r)
                rf.fit(Ftr, np.concatenate([y[trm], ya]))
                proba = rf.predict_proba(F[tem])
            else:
                epochs = EPOCHS if rotate is None else max(1, round(EPOCHS / (1 + N_COPIES)))
                proba = fit_predict("cnn", np.concatenate([X[trm], Xa]), np.concatenate([y[trm], ya]), X[tem], r, epochs=epochs)
            b = vote_bouts(proba, meta[tem])
            b["group"] = np.where(b.athlete.isin(plan["new"]), "new", "recorded")
            for g, bg in b.groupby("group"):
                rows.append(per_athlete(bg).assign(group=g, repeat=r, n_team=n, arm=arm, model=model,
                                                   n_real_windows=int(trm.sum()), n_sim_windows=len(Xa)))
    return pd.concat(rows, ignore_index=True)


def athlete_equivalent(score, curve_n, curve_f1):
    """Real team size whose (monotone-interpolated) score equals `score`."""
    f = np.maximum.accumulate(np.asarray(curve_f1))
    if score < f[0] or score > f[-1]:
        return np.nan  # outside the observed curve (below 2 or above 17 real athletes): not reported
    return float(np.interp(score, f, curve_n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rf", "cnn"], required=True)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    low_priority()
    get_windows()

    t0 = time.time()
    out = Parallel(n_jobs=args.jobs, verbose=0)(delayed(one)(r, args.model) for r in range(args.repeats))
    print(f"{args.repeats} repeats in {(time.time() - t0) / 60:.1f} min on {args.jobs} workers")
    res = pd.concat(out, ignore_index=True).assign(git=git_hash())

    base_file = OUT / ("04_rf_athletes.csv" if args.model == "rf" else "05_cnn_athletes.csv")
    ref = pd.read_csv(base_file)
    ref = ref[(ref.k == 1) & (ref.repeat < args.repeats)]
    a = res[res.arm == "none"]
    m = a.merge(ref, on=["repeat", "n_team", "group", "athlete"], suffixes=("", "_ref"))
    dmax = (m.macro_f1 - m.macro_f1_ref).abs().max()
    print(f"none vs {base_file.name}: {len(m)} of {len(a)} rows matched, max |diff| = {dmax:.1e}")
    assert len(m) == len(a) and dmax < 1e-9

    if args.verify:
        v = one(0, args.model)
        b = res[res.repeat == 0].drop(columns="git").reset_index(drop=True)
        pd.testing.assert_frame_equal(v.reset_index(drop=True)[b.columns], b)
        print("verify: repeat 0 reproduces exactly")

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / f"11_sim_{args.model}_athletes.csv", index=False)

    curve = ref[ref.group == "new"].groupby(["n_team", "athlete"]).macro_f1.mean().groupby(level=0).mean()
    curve = curve.reindex(CURVE_N)
    ath = res[res.group == "new"].groupby(["n_team", "arm", "athlete"]).macro_f1.mean()
    rows = []
    for n in TEAM_SIZES:
        base = ath.loc[(n, "none")]
        for arm in ("tempo_intensity", "plus_placement"):
            s = ath.loc[(n, arm)]
            d = (s - base).dropna()
            mm, lo, hi = bootstrap_ci(d.to_numpy())
            eq = athlete_equivalent(s.mean(), CURVE_N, curve.to_numpy())
            rows.append(dict(model=args.model, n_team=n, arm=arm, none=base.mean(), simulated=s.mean(), diff=mm, ci_lo=lo,
                             ci_hi=hi, p=wilcoxon(d).pvalue if (d != 0).any() else 1.0, n_positive=int((d > 0).sum()),
                             n=len(d), equivalent_real_athletes=eq, extra_athletes_worth=eq - n if eq == eq else np.nan))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / f"11_sim_{args.model}_summary.csv", index=False)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
