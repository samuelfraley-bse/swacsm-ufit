"""Step 08: statistics on saved results (no new training). Frames results as a data-collection budget.

Reads outputs/tables/{04_rf,05_cnn,05_lstm}_athletes.csv and 06_battery_*_bouts.csv.gz; writes
outputs/tables/08_*.csv. Every comparison is paired by athlete: each athlete's metric is averaged over
the repeats in which they appear in that role, then compared (Wilcoxon signed-rank; 95% percentile
bootstrap CI over athletes).

  A. Equal budget   N athletes x 2 sets vs 2N athletes x 1 set (same number of recorded sets), new athletes
  B. Depth          k=2 vs k=1 at each N, new athletes
  C. Gap G(N)       recorded minus new, same athlete in both roles
  D. Marginal gain  per added athlete between grid points; N90/N95 = athletes needed to reach 90/95% of
                    the gain observed from N=2 to N=17 (linear interpolation, bootstrap CI). Secondary:
                    power-law fit A(N) = A_inf - c * N^-alpha (extrapolation beyond N=17; flagged)
  E. Models         CNN vs RF, LSTM vs RF, CNN vs LSTM at each N, k (new and recorded)
  F. Transfer       battery study: accuracy change on non-battery exercises that are similar to a battery
                    exercise vs dissimilar ones. Similar = (a) same movement family (defined in advance),
                    (b) confusable pair in the pair screen (balanced accuracy < 0.95)

    python steps/08_analysis.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from src.metrics import bootstrap_ci  # noqa: E402
from src.screen import FAMILIES  # noqa: E402

T = ROOT / "outputs" / "tables"
MODELS = ["rf", "cnn", "lstm"]
N_GRID = [2, 4, 6, 8, 12, 16, 17]
METRIC = "macro_f1"
N_BOOT = 2000


def load_main():
    parts = [pd.read_csv(T / "04_rf_athletes.csv")] + [pd.read_csv(T / f"05_{m}_athletes.csv") for m in ("cnn", "lstm")]
    return pd.concat(parts, ignore_index=True)


def athlete_means(res):
    return res.groupby(["model", "n_team", "k", "group", "athlete"])[METRIC].mean()


def paired(a, b):
    """a, b: Series indexed by athlete. Returns dict of paired stats for a - b."""
    d = (a - b).dropna()
    m, lo, hi = bootstrap_ci(d.to_numpy(), n_boot=N_BOOT)
    p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
    return dict(diff=m, ci_lo=lo, ci_hi=hi, p=p, n=len(d), n_positive=int((d > 0).sum()), mean_a=a.loc[d.index].mean(),
                mean_b=b.loc[d.index].mean())


def equal_budget(ath):
    rows = []
    for m in MODELS:
        for n in (2, 4, 6, 8):
            deep = ath.loc[(m, n, 2, "new")]
            broad = ath.loc[(m, 2 * n, 1, "new")]
            rows.append(dict(model=m, sets_per_exercise_total=2 * n, deep=f"{n} athletes x 2 sets",
                             broad=f"{2 * n} athletes x 1 set", **paired(broad, deep)))
    return pd.DataFrame(rows)  # diff = broad - deep


def depth(ath):
    rows = []
    for m in MODELS:
        for n in N_GRID:
            rows.append(dict(model=m, n_team=n, **paired(ath.loc[(m, n, 2, "new")], ath.loc[(m, n, 1, "new")])))
    return pd.DataFrame(rows)  # diff = k2 - k1


def gap(ath):
    rows = []
    for m in MODELS:
        for k in (1, 2):
            for n in N_GRID:
                rows.append(dict(model=m, k=k, n_team=n, **paired(ath.loc[(m, n, k, "recorded")], ath.loc[(m, n, k, "new")])))
    return pd.DataFrame(rows)  # diff = recorded - new


def n_threshold(curve, frac):
    """Smallest N (linear interpolation on N_GRID) where the gain from N=2 reaches `frac` of the gain to N=17."""
    y = np.asarray(curve, float)
    total = y[-1] - y[0]
    if total <= 0:
        return np.nan
    f = (y - y[0]) / total
    for i in range(1, len(y)):
        if f[i] >= frac:
            x0, x1, f0, f1 = N_GRID[i - 1], N_GRID[i], f[i - 1], f[i]
            return x0 + (frac - f0) * (x1 - x0) / (f1 - f0) if f1 > f0 else x1
    return np.nan


def power_law(n, a_inf, c, alpha):
    return a_inf - c * np.power(n, -alpha)


def fit_power(y):
    try:
        p, _ = curve_fit(power_law, np.array(N_GRID, float), y, p0=[y[-1] + 0.05, 1.0, 0.7],
                         bounds=([0, 0, 0.05], [1.0, 10, 5]), maxfev=20000)
        return p
    except (RuntimeError, ValueError):
        return [np.nan] * 3


def marginal_and_thresholds(ath, seed=0):
    marg, thr, pw = [], [], []
    rng = np.random.default_rng(seed)
    for m in MODELS:
        for k in (1, 2):
            A = pd.concat({n: ath.loc[(m, n, k, "new")] for n in N_GRID}, axis=1).dropna()  # athletes x N
            for i in range(len(N_GRID) - 1):
                n0, n1 = N_GRID[i], N_GRID[i + 1]
                s = paired(A[n1], A[n0])
                marg.append(dict(model=m, k=k, from_n=n0, to_n=n1, gain=s["diff"], gain_lo=s["ci_lo"], gain_hi=s["ci_hi"],
                                 gain_per_athlete=s["diff"] / (n1 - n0), p=s["p"]))
            curve = A.mean().to_numpy()
            boots = A.to_numpy()[rng.integers(0, len(A), size=(N_BOOT, len(A)))].mean(axis=1)  # N_BOOT x len(N_GRID)
            for frac in (0.90, 0.95):
                est = n_threshold(curve, frac)
                b = np.array([n_threshold(c, frac) for c in boots])
                thr.append(dict(model=m, k=k, fraction=frac, n_needed=est, ci_lo=np.nanpercentile(b, 2.5),
                                ci_hi=np.nanpercentile(b, 97.5), a_n2=curve[0], a_n17=curve[-1]))
            a_inf, c, alpha = fit_power(curve)
            bp = np.array([fit_power(cb) for cb in boots[:500]])
            n95 = lambda ai, cc, al: (cc / (0.05 * (ai - power_law(2, ai, cc, al)))) ** (1 / al) if al > 0 else np.nan  # noqa: E731
            est95 = n95(a_inf, c, alpha)
            b95 = np.array([n95(*row) for row in bp])
            pw.append(dict(model=m, k=k, a_inf=a_inf, a_inf_lo=np.nanpercentile(bp[:, 0], 2.5), a_inf_hi=np.nanpercentile(bp[:, 0], 97.5),
                           c=c, alpha=alpha, n95_of_asymptote=est95, n95_lo=np.nanpercentile(b95, 2.5), n95_hi=np.nanpercentile(b95, 97.5),
                           note="extrapolation beyond N=17; secondary"))
    return pd.DataFrame(marg), pd.DataFrame(thr), pd.DataFrame(pw)


def model_compare(ath):
    rows = []
    for a, b in (("cnn", "rf"), ("lstm", "rf"), ("cnn", "lstm")):
        for g in ("new", "recorded"):
            for k in (1, 2):
                for n in N_GRID:
                    rows.append(dict(comparison=f"{a} - {b}", group=g, k=k, n_team=n,
                                     **paired(ath.loc[(a, n, k, g)], ath.loc[(b, n, k, g)])))
    return pd.DataFrame(rows)


def confusable_pairs(thresh=0.95):
    s = pd.read_csv(ROOT / "results" / "pair_screen.csv")
    s = s[s.balanced_acc < thresh]
    return {frozenset((int(a), int(b))) for a, b in zip(s.class_a, s.class_b)}


def transfer():
    conf = confusable_pairs()
    rows = []
    for m in MODELS:
        p = pd.read_csv(T / f"06_battery_{m}_bouts.csv.gz")
        key = ["repeat", "n_team", "athlete", "segment_id"]
        none = p[p.condition == "none"][key + ["pred"]].rename(columns={"pred": "pred_none"})
        b = p[p.condition == "battery"].merge(none, on=key, how="left")
        assert b.pred_none.notna().all()
        cls = b.battery_classes.str.split(",").apply(lambda v: [int(x) for x in v])
        b = b[[t not in c for t, c in zip(b.true, cls)]].copy()  # only exercises NOT in the battery
        cls = b.battery_classes.str.split(",").apply(lambda v: [int(x) for x in v])
        b["same_family"] = [any(FAMILIES[t] & FAMILIES[c] for c in cs) for t, cs in zip(b.true, cls)]
        b["confusable"] = [any(frozenset((t, c)) in conf for c in cs) for t, cs in zip(b.true, cls)]
        b["change"] = (b.pred == b.true).astype(float) - (b.pred_none == b.true).astype(float)
        for n, bn in b.groupby("n_team"):
            for tag in ("same_family", "confusable"):
                a = bn.groupby([tag, "athlete"]).change.mean()
                sim, dis = a.loc[True], a.loc[False]
                s_sim, s_dis, s_diff = paired(sim, 0 * sim), paired(dis, 0 * dis), paired(sim, dis)
                rows.append(dict(model=m, n_team=n, similarity=tag, n_bouts_similar=int(bn[tag].sum()),
                                 n_bouts_dissimilar=int((~bn[tag]).sum()),
                                 change_similar=s_sim["diff"], similar_lo=s_sim["ci_lo"], similar_hi=s_sim["ci_hi"], p_similar=s_sim["p"],
                                 change_dissimilar=s_dis["diff"], p_dissimilar=s_dis["p"],
                                 similar_minus_dissimilar=s_diff["diff"], diff_lo=s_diff["ci_lo"], diff_hi=s_diff["ci_hi"],
                                 p_diff=s_diff["p"], n_athletes=s_diff["n"]))
    return pd.DataFrame(rows)


def fmt(df, cols):
    return df[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}" if abs(v) < 1e-3 and v != 0 else f"{v:.3f}")


def main():
    res = load_main()
    ath = athlete_means(res)

    eb = equal_budget(ath)
    dp = depth(ath)
    gp = gap(ath)
    mg, th, pw = marginal_and_thresholds(ath)
    mc = model_compare(ath)
    tr = transfer()
    for name, df in (("equal_budget", eb), ("depth", dp), ("gap", gp), ("marginal", mg), ("thresholds", th),
                     ("powerlaw", pw), ("model_compare", mc), ("transfer_family", tr)):
        df.to_csv(T / f"08_{name}.csv", index=False)

    print("A. Equal budget (diff = broad - deep, new athletes)")
    print(fmt(eb, ["model", "deep", "broad", "mean_b", "mean_a", "diff", "ci_lo", "ci_hi", "p", "n_positive", "n"]))
    print("\nB. Depth (diff = k2 - k1, new athletes)")
    print(fmt(dp[dp.n_team.isin([2, 8, 17])], ["model", "n_team", "diff", "ci_lo", "ci_hi", "p"]))
    print("\nC. Gap G(N) = recorded - new (k=2)")
    print(fmt(gp[gp.k == 2], ["model", "n_team", "diff", "ci_lo", "ci_hi", "p", "n"]))
    print("\nD1. Marginal gain per added athlete (k=1, new)")
    print(fmt(mg[mg.k == 1], ["model", "from_n", "to_n", "gain", "gain_lo", "gain_hi", "gain_per_athlete", "p"]))
    print("\nD2. Athletes needed to reach 90/95% of the gain observed from N=2 to N=17 (new)")
    print(fmt(th, ["model", "k", "fraction", "n_needed", "ci_lo", "ci_hi", "a_n2", "a_n17"]))
    print("\nD3. Power-law fit (SECONDARY, extrapolates beyond N=17)")
    print(fmt(pw, ["model", "k", "a_inf", "a_inf_lo", "a_inf_hi", "alpha", "n95_of_asymptote", "n95_lo", "n95_hi"]))
    print("\nE. Model comparisons, k=2")
    print(fmt(mc[(mc.k == 2) & mc.n_team.isin([2, 8, 17])], ["comparison", "group", "n_team", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi", "p"]))
    print("\nF. Transfer: accuracy change on non-battery exercises, similar vs dissimilar to a battery exercise")
    print(fmt(tr, ["model", "n_team", "similarity", "n_bouts_similar", "change_similar", "similar_lo", "similar_hi",
                   "change_dissimilar", "similar_minus_dissimilar", "p_diff", "n_athletes"]))


if __name__ == "__main__":
    main()
