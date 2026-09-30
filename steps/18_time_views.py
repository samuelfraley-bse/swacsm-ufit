"""Step 18: three time-based views (logistic regression, the main model; new athletes unless stated).

  A  fig12_budget_minutes   continuous budget curve: x = actual minutes of labeled lifting in each training run
                            (sum of the training sets' durations; rest excluded), one dot per repeat x condition,
                            per strategy (1 / 2 / 4 sets per athlete); trend = least-squares fit on log2(minutes).
                            Uses saved results (steps 16) + set durations; no training.
  B  fig13_days_since       does accuracy drift with time since the recording day? Accuracy per athlete x test
                            session vs calendar days after the athlete's day 1, recorded and new athletes,
                            8 athletes recorded x 1 set. Saved predictions; no training.
  C  fig14_realtime_fine    running call at 1 s resolution (2-20 s into the set) for 2, 8 and 17 athletes
                            recorded x 1 set; retrains logistic regression (same recipe/seeds as step 16).
                            After a set ends, its final call is carried forward.

    python steps/18_time_views.py [--jobs 24]
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

from src.experiment import repeat_plan, train_ids, train_ids_depth  # noqa: E402
from src.metrics import bootstrap_ci  # noqa: E402
from src.plots import INK, INK2, MUTED, SERIES, SURFACE, save  # noqa: E402
from src.team import load_manifest, test_bouts  # noqa: E402

T = ROOT / "outputs" / "tables"
BLUE, ORANGE = SERIES[0], SERIES[1]
RAMP3 = ["#86b6ef", "#2a78d6", "#104281"]
TIMES = list(range(2, 21))


def header(fig, text, sub, top_pad=62):
    h_in = fig.get_size_inches()[1]
    fig.text(0.01, 0.99, text, ha="left", va="top", fontsize=20, color=INK)
    fig.text(0.01, 0.99 - 30 / 72 / h_in, sub, ha="left", va="top", fontsize=13, color=INK2)
    return 0.99 - top_pad / 72 / h_in


# ------------------------------------------------------------------------------------------------ A
def view_a(man):
    dur = man.set_index("segment_id").duration_seconds
    res = pd.read_csv(T / "16_baselines_athletes.csv").query("model == 'logreg' and group == 'new'")
    deep = pd.read_csv(T / "16_deep_broad_baselines_athletes.csv").query("model == 'logreg' and depth == 4")
    rows = []
    for r in sorted(res.repeat.unique()):
        plan = repeat_plan(r, man)
        for (n, k), g in res[res.repeat == r].groupby(["n_team", "k"]):
            _, ids = train_ids(plan, n, k)
            rows.append(dict(repeat=r, n_team=n, depth=k, minutes=dur.loc[ids].sum() / 60, f1=g.macro_f1.mean()))
        for n, g in deep[deep.repeat == r].groupby("n_team"):
            _, ids = train_ids_depth(plan, man, n, 4)
            rows.append(dict(repeat=r, n_team=n, depth=4, minutes=dur.loc[ids].sum() / 60, f1=g.macro_f1.mean()))
    P = pd.DataFrame(rows)
    P.to_csv(T / "18_budget_minutes.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    labels = {1: "More athletes, 1 set each", 2: "2 sets per athlete", 4: "4 sets per athlete (2 days)"}
    fits = {}
    for depth, col in zip((4, 2, 1), RAMP3):
        d = P[P.depth == depth]
        ax.scatter(d.minutes, d.f1, s=22, color=col, alpha=0.45, edgecolor="none", zorder=2)
        b, a = np.polyfit(np.log2(d.minutes), d.f1, 1)
        xs = np.linspace(d.minutes.min(), d.minutes.max(), 100)
        ax.plot(xs, a + b * np.log2(xs), color=col, lw=4, zorder=4, label=labels[depth])
        fits[depth] = (a, b)
        ax.annotate(labels[depth].split(",")[0] if depth == 1 else labels[depth], (xs[-1], a + b * np.log2(xs[-1])),
                    xytext=(8, 0), textcoords="offset points", fontsize=12, color=col if depth != 4 else INK2, va="center")
    ax.set_xscale("log", base=2)
    ticks = [8, 15, 30, 60, 120]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel("Minutes of labeled lifting used for training (all athletes combined; rest excluded)")
    ax.set_ylabel("Macro-F1, new athletes (mean of 6 per repeat)")
    ax.set_ylim(0.45, 1.0)
    ax.set_xlim(P.minutes.min() * 0.9, P.minutes.max() * 1.6)
    h, l = ax.get_legend_handles_labels()
    top = header(fig, "Same minutes of recording: more athletes wins",
                 "Logistic regression. Each dot = one repeat (30 per condition); lines = fit on log2(minutes). "
                 "One athlete's full circuit ≈ 5 min of lifting.")
    fig.legend(h[::-1], l[::-1], loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.005, top + 0.01))
    fig.tight_layout(rect=(0, 0, 1, top - 0.05))
    save(fig, "fig12_budget_minutes")
    for depth, (a, b) in fits.items():
        print(f"A  depth {depth}: F1 = {a:.3f} + {b:.3f} * log2(minutes)  (per doubling of minutes: +{b:.3f})")
    for m in (20, 40, 80):
        print(f"A  at {m} min: " + ", ".join(f"{d} set(s): {fits[d][0] + fits[d][1] * np.log2(m):.3f}" for d in (1, 2, 4)))


# ------------------------------------------------------------------------------------------------ B
def view_b(man, n_team=8):
    b = pd.read_csv(T / "16_baselines_bouts.csv.gz").query(f"model == 'logreg' and k == 1 and n_team == {n_team}")
    info = man.set_index("segment_id")[["participant_id", "session_id", "session_date", "day_order"]]
    b = b.join(info, on="segment_id")
    day1 = man[man.day_order == 1].groupby("participant_id").session_date.first()
    b["days_since"] = (pd.to_datetime(b.session_date) - pd.to_datetime(b.participant_id.map(day1))).dt.days
    s = b.assign(ok=b.true == b.pred).groupby(["group", "athlete", "session_id", "days_since"]).ok.mean().reset_index()
    s.to_csv(T / "18_accuracy_by_days.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 7))
    rng = np.random.default_rng(0)
    for grp, col, lab in (("recorded", ORANGE, "Recorded athletes"), ("new", BLUE, "New athletes")):
        d = s[s.group == grp]
        ax.scatter(d.days_since + rng.uniform(-0.25, 0.25, len(d)), d.ok, s=40, color=col, alpha=0.55, edgecolor="none")
        b1, a1 = np.polyfit(d.days_since, d.ok, 1)
        xs = np.array([d.days_since.min(), d.days_since.max()])
        ax.plot(xs, a1 + b1 * xs, color=col, lw=4, label=f"{lab}: trend {100 * b1:+.2f} points per day")
        rho, p = spearmanr(d.days_since, d.ok)
        print(f"B  {grp}: {len(d)} athlete-sessions, days {d.days_since.min()}-{d.days_since.max()}, "
              f"slope {b1:+.4f}/day, Spearman rho {rho:+.2f} (p = {p:.2g})")
        # paired: first vs last test day per athlete (athletes with >= 2 test days)
        per = d.groupby(["athlete", "days_since"]).ok.mean().reset_index()
        pairs = per.groupby("athlete").filter(lambda g: g.days_since.nunique() >= 2).groupby("athlete")
        diff = pairs.apply(lambda g: g.sort_values("days_since").ok.iloc[-1] - g.sort_values("days_since").ok.iloc[0],
                           include_groups=False)
        if len(diff) > 5:
            print(f"B  {grp}: last - first test day, {len(diff)} athletes: {diff.mean():+.3f} (p = {wilcoxon(diff).pvalue:.2g})")
    ax.set_xlabel("Days since the recording day (day 1)")
    ax.set_ylabel("Share of sets named correctly (per athlete and session)")
    ax.set_ylim(0.3, 1.02)
    h, l = ax.get_legend_handles_labels()
    top = header(fig, "Does the tracker drift over time?",
                 f"Logistic regression trained on {n_team} athletes × 1 set (day 1). Each dot = one athlete's test session, "
                 "averaged over repeats.")
    fig.legend(h, l, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.005, top + 0.01))
    fig.tight_layout(rect=(0, 0, 1, top - 0.05))
    save(fig, "fig13_days_since")


# ------------------------------------------------------------------------------------------------ C
def one_realtime(r):
    from threadpoolctl import threadpool_limits
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from src.data import get_windows
    from src.realtime import calls_at_times, running_calls

    X, F, meta = get_windows()
    y = meta.cls.to_numpy()
    seg = meta.segment_id.to_numpy()
    man = load_manifest()
    plan = repeat_plan(r, man)
    tem = np.isin(seg, test_bouts(man, plan["new"]))
    out = []
    for n in (2, 8, 17):
        _, tr = train_ids(plan, n, 1)
        trm = np.isin(seg, tr)
        with threadpool_limits(1), warnings.catch_warnings():  # same recipe as step 16
            warnings.simplefilter("ignore")
            sc = StandardScaler().fit(F[trm])
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=r).fit(sc.transform(F[trm]), y[trm])
            proba = lr.predict_proba(sc.transform(F[tem]))
        out.append(calls_at_times(running_calls(proba, meta[tem]), times=TIMES).assign(repeat=r, n_team=n))
    return pd.concat(out, ignore_index=True)


def view_c(man, jobs):
    sets = pd.concat(Parallel(n_jobs=jobs)(delayed(one_realtime)(r) for r in range(30)), ignore_index=True)
    # check: end-of-set call must reproduce step 16 (logreg, new athletes, k=1)
    ref = pd.read_csv(T / "16_baselines_bouts.csv.gz").query("model == 'logreg' and group == 'new' and k == 1")
    end = sets[np.isinf(sets.t_s)].merge(ref[["repeat", "n_team", "segment_id", "pred"]], on=["repeat", "n_team", "segment_id"])
    assert len(end) == (np.isinf(sets.t_s)).sum() and (end.call == end.pred).all(), "end-of-set calls differ from step 16"
    print(f"C  end-of-set calls match step 16 for all {len(end)} sets")
    acc = sets.assign(ok=sets.call == sets.true).groupby(["n_team", "t_s", "athlete"]).ok.mean()
    rows = []
    for (n, t), g in acc.groupby(level=[0, 1]):
        m, lo, hi = bootstrap_ci(g.to_numpy())
        rows.append(dict(n_team=n, t_s=t, accuracy=m, ci_lo=lo, ci_hi=hi))
    S = pd.DataFrame(rows)
    S.to_csv(T / "18_realtime_fine.csv", index=False)
    running = sets[np.isinf(sets.t_s)].set_duration_s
    med = running.median()

    fig, ax = plt.subplots(figsize=(13, 7))
    for n, col in zip((2, 8, 17), RAMP3):
        d = S[(S.n_team == n) & np.isfinite(S.t_s)].sort_values("t_s")
        ax.fill_between(d.t_s, d.ci_lo, d.ci_hi, color=col, alpha=0.2, lw=0)
        ax.plot(d.t_s, d.accuracy, color=col, lw=3, label=f"{n} athletes recorded")
        e = S[(S.n_team == n) & np.isinf(S.t_s)].accuracy.iloc[0]
        ax.annotate(f"{n} recorded (end of set {e:.2f})", (20, d.accuracy.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    fontsize=12, color=INK2, va="center")
    still = [(running > t).mean() for t in TIMES]
    ax.axvline(med, color=MUTED, lw=1.2, ls=(0, (2, 3)))
    ax.text(med + 0.2, 0.52, f"median set ends\n({med:.0f} s)", fontsize=11, color=INK2)
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ax.set_xlim(1.5, 26)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Seconds into the set (a new 2 s window every second; ≈1 rep every 2 s)")
    ax.set_ylabel("Share of sets named correctly, new athletes")
    top = header(fig, "The tracker's call is set within the first rep or two",
                 "Logistic regression, 1 set per exercise; running vote over windows so far; set start given; "
                 "finished sets keep their final call. 95% CI over 23 athletes.")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, "fig14_realtime_fine")
    for n in (2, 8, 17):
        d = S[S.n_team == n].set_index("t_s").accuracy
        print(f"C  N={n}: 2 s {d[2]:.3f}, 4 s {d[4]:.3f}, 10 s {d[10]:.3f}, end {d[np.inf]:.3f}")
    print("C  share of sets still running at 2/10/20 s:", [round(still[i], 2) for i in (0, 8, 18)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    man = load_manifest()
    view_a(man)
    view_b(man)
    view_c(man, args.jobs)


if __name__ == "__main__":
    main()
