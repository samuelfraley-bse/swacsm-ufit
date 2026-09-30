"""Step 14: poster figures from saved results (no training). 300 dpi PNG + PDF in figs/.

  fig1_learning_curves   new vs recorded athletes by number recorded; one panel per model
  fig2_budget            same budget spent deep / middle / broad (step 12); RF and CNN
  fig3_shortcuts         forest plot of low-data shortcuts vs "record 2 more athletes"
  fig4_posture_motion    what the sensor uses (step 07): new-athlete F1 and recorded-new gap by signal
  fig5_realtime          accuracy by seconds into the set (step 13); RF and CNN

Colors (reference palette, validated slots): new athletes = blue, recorded = orange; ordinal comparisons use
one blue ramp (light -> dark). Text stays in ink colors.

    python steps/14_figures.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.metrics import bootstrap_ci  # noqa: E402
from src.plots import AXIS, INK, INK2, MUTED, SERIES, SURFACE, save  # noqa: E402

T = ROOT / "outputs" / "tables"
BLUE, ORANGE = SERIES[0], SERIES[1]
RAMP3 = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal steps 250 / 450 / 650 (lightest clears 2:1 on light)
MODEL_NAME = {"rf": "Random forest", "cnn": "CNN", "lstm": "LSTM"}
N_GRID = [2, 4, 6, 8, 12, 16, 17]


def header(fig, text, sub, handles=None, labels=None, ncol=3):
    """Title, subtitle and (optional) legend stacked top-left; returns the top of the plotting area."""
    h_in = fig.get_size_inches()[1]
    line = lambda pts: pts / 72 / h_in  # noqa: E731  (points -> figure fraction)
    y = 0.99
    fig.text(0.01, y, text, ha="left", va="top", fontsize=20, color=INK)
    y -= line(30)
    fig.text(0.01, y, sub, ha="left", va="top", fontsize=14, color=INK2)
    y -= line(24)
    if handles:
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, y), ncol=ncol, frameon=False)
        y -= line(30)
    return y - line(6)


def fig1():
    s = pd.concat([pd.read_csv(T / "04_rf_summary.csv"), pd.read_csv(T / "05_cnn_summary.csv"),
                   pd.read_csv(T / "05_lstm_summary.csv")])
    s = s[s.k == 1]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
    for ax, m in zip(axes, ["rf", "cnn", "lstm"]):
        for grp, col, lab in (("recorded", ORANGE, "Recorded athletes"), ("new", BLUE, "New (never-recorded) athletes")):
            d = s[(s.model == m) & (s.group == grp)].sort_values("n_team")
            ax.fill_between(d.n_team, d.ci_lo, d.ci_hi, color=col, alpha=0.18, lw=0)
            ax.plot(d.n_team, d["mean"], color=col, lw=2.5, marker="o", ms=8, mec=SURFACE, mew=2, label=lab)
            ax.annotate(f"{d['mean'].iloc[-1]:.2f}", (d.n_team.iloc[-1], d["mean"].iloc[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=13, color=INK2)
        ax.set_title(MODEL_NAME[m], loc="left", fontsize=17, color=INK)
        ax.set_xticks([2, 4, 6, 8, 12, 17])
        ax.set_xlim(1, 19.5)
        ax.set_xlabel("Athletes recorded (1 set per exercise)")
    axes[0].set_ylabel("Macro-F1 (whole sets)")
    axes[0].set_ylim(0.4, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    top = header(fig, "Recording more athletes improves recognition of new athletes",
                 "The gap to recorded athletes shrinks but never closes. Mean over 23 athletes, 95% CI.",
                 h[::-1], l[::-1], ncol=2)
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, "fig1_learning_curves")


def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    labels = ["Few athletes, many sets", "In between", "Many athletes, 1 set"]
    for ax, m in zip(axes, ["rf", "cnn"]):
        r = pd.read_csv(T / f"12_deep_broad_{m}_athletes.csv")
        ath = r.groupby(["budget", "n_team", "depth", "athlete"]).macro_f1.mean()
        budgets = sorted(r.budget.unique())
        w = 0.26
        for bi, b in enumerate(budgets):
            splits = sorted(r[r.budget == b][["n_team", "depth"]].drop_duplicates().itertuples(index=False), key=lambda x: x.n_team)
            for si, (n, d) in enumerate(splits):
                mean, lo, hi = bootstrap_ci(ath.loc[(b, n, d)].to_numpy())
                x = bi + (si - 1) * w
                ax.bar(x, mean, w, color=RAMP3[si], edgecolor=SURFACE, lw=2, label=labels[si] if bi == 0 else None)
                ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], color=INK, lw=1.5, capsize=4)
                ax.text(x, 0.02, f"{n}×{d}", ha="center", va="bottom", fontsize=12,
                        color="#ffffff" if si > 0 else INK)
                ax.text(x, hi + 0.012, f"{mean:.2f}", ha="center", va="bottom", fontsize=12, color=INK2)
        ax.set_xticks(range(len(budgets)), [f"{b} athlete-sets\nper exercise" for b in budgets])
        ax.set_title(MODEL_NAME[m], loc="left", fontsize=17, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1.0)  # bars start at zero
    axes[0].set_ylabel("Macro-F1, new athletes")
    h, l = axes[0].get_legend_handles_labels()
    top = header(fig, "Same recording budget: spread it across more athletes",
                 "Labels show athletes × sets per exercise; all 15 exercises in every condition. 95% CI over 23 athletes.", h, l)
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, "fig2_budget")


def fig3():
    rows = []  # (section, label, detail, mean, lo, hi, is_reference)
    mg = pd.read_csv(T / "08_marginal.csv").query("model == 'rf' and k == 1 and from_n == 2 and to_n == 4").iloc[0]
    rows.append(("Reference", "Record 2 more athletes", "RF, 2 → 4 athletes", mg.gain, mg.gain_lo, mg.gain_hi, True))
    sc = pd.read_csv(T / "06_battery_cnn_scores.csv").query("n_team == 17")
    a = sc.groupby("athlete")[["f1_none", "f1_battery", "f1_all15"]].mean()
    for lab, col, ref in (("Record new athlete on all 15 exercises", "f1_all15", True),
                          ("Record new athlete on 2 exercises", "f1_battery", False)):
        mm, lo, hi = bootstrap_ci((a[col] - a.f1_none).to_numpy())
        rows.append(("Record the new athlete", lab, "CNN, 17 recorded; scored on their other exercises", mm, lo, hi, ref))
    tg = pd.read_csv(T / "09_targeted_summary.csv").query("metric == 'macro_f1' and n_team == 10")
    for comp, lab, ref in (("press - base", "2nd set of the 3 presses", False), ("broad - base", "Same sets spent on 20% more athletes", True)):
        r = tg[tg.comparison == comp].iloc[0]
        rows.append(("Extra sets", lab, "RF, 10 athletes", r["diff"], r.ci_lo, r.ci_hi, ref))
    ab = pd.read_csv(T / "06b_cnn_summary.csv").query("condition == 'adabn'")
    for r in ab.itertuples():
        rows.append(("No labels needed", "Label-free calibration (1 unlabeled session)", f"CNN, {r.n_team} recorded", r.gain_vs_none, r.gain_lo, r.gain_hi, False))
    ss = pd.read_csv(T / "10_ssl_summary.csv").query("condition == 'ssl_team'")
    for r in ss.itertuples():
        rows.append(("No labels needed", "Self-training on unlabeled teammates", f"RF, {r.n_labeled} labeled", r.diff, r.ci_lo, r.ci_hi, False))

    fig, ax = plt.subplots(figsize=(14, 8.5))
    y = 0
    yt, yl = [], []
    last = None
    for sec, lab, det, mm, lo, hi, ref in rows:
        if sec != last:
            y -= 0.6
            ax.text(-0.075, y, sec, fontsize=14, color=INK, fontweight="bold", va="center")
            y -= 0.8
            last = sec
        col = MUTED if ref else BLUE
        ax.plot([lo, hi], [y, y], color=col, lw=3, solid_capstyle="round")
        ax.plot(mm, y, "o", color=col, ms=11, mec=SURFACE, mew=2)
        ax.text(hi + 0.004, y, f"{mm:+.3f}", va="center", fontsize=12, color=INK2)
        yt.append(y)
        yl.append(f"{lab}\n{det}")
        y -= 1
    ax.axvline(0, color=AXIS, lw=1.5)
    ax.set_yticks(yt, yl, fontsize=12)
    ax.set_ylim(y + 0.3, 0.2)
    ax.set_xlim(-0.08, 0.16)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in macro-F1 for new athletes (95% CI)")
    ax.text(0.155, y + 0.6, "gray = reference options", ha="right", fontsize=12, color=MUTED)
    top = header(fig, "Shortcuts are worth about one extra athlete at best",
                 "Blue: low-cost shortcuts. Gray: what real recording buys. New athletes; 95% CI over 23 athletes.")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, "fig3_shortcuts")


def fig4():
    s = pd.read_csv(T / "07_rf_summary.csv").query("n_team == 17").set_index("variant")
    order = ["full", "gravity", "motion", "magnitude"]
    names = ["Full signal", "Posture only\n(gravity)", "Movement only\n(gravity removed)", "Total acceleration\nonly (no orientation)"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
    for ax, (col, lo, hi, lab, c) in zip(axes, (("new_f1", "new_lo", "new_hi", "Macro-F1, new athletes", BLUE),
                                               ("gap", "gap_lo", "gap_hi", "Recorded − new gap (lower = more shared)", ORANGE))):
        v = s.loc[order]
        x = np.arange(len(order))
        ax.bar(x, v[col], 0.6, color=c, edgecolor=SURFACE, lw=2)
        ax.errorbar(x, v[col], yerr=[v[col] - v[lo], v[hi] - v[col]], fmt="none", color=INK, lw=1.5, capsize=4)
        for xi, (m, h) in enumerate(zip(v[col], v[hi])):
            ax.text(xi, h + 0.01, f"{m:.2f}", ha="center", va="bottom", fontsize=13, color=INK2)
        ax.set_xticks(x, names, fontsize=13)
        ax.set_title(lab, loc="left", fontsize=16, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1.0)
    axes[1].set_ylim(0, 0.25)
    top = header(fig, "Wrist posture is shared across people; movement is personal",
                 "Random forest, 17 athletes recorded. 95% CI over 23 athletes.")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, "fig4_posture_motion")


def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    END_X = 18  # median set length (s); plotted position for "end of set"
    for ax, m in zip(axes, ["rf", "cnn"]):
        s = pd.read_csv(T / f"13_realtime_{m}_summary.csv")
        s["x"] = s.t_s.replace(np.inf, END_X)
        for c, n in zip(RAMP3, [2, 8, 17]):
            d = s[s.n_team == n].sort_values("x")
            ax.fill_between(d.x, d.ci_lo, d.ci_hi, color=c, alpha=0.2, lw=0)
            ax.plot(d.x, d.accuracy, color=c, lw=2.5, marker="o", ms=8, mec=SURFACE, mew=2, label=f"{n} athletes recorded")
            ax.annotate(f"{n} recorded", (END_X, d.accuracy.iloc[-1]), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=12, color=INK2)
        ax.set_xticks([2, 4, 6, 8, 10, 15, END_X], ["2", "4", "6", "8", "10", "15", "end"])
        ax.set_xlim(0.5, 23)
        ax.set_xlabel("Seconds into the set (≈ 1 rep every 2 s)")
        ax.set_title(MODEL_NAME[m], loc="left", fontsize=17, color=INK)
    axes[0].set_ylabel("Sets named correctly, new athletes")
    axes[0].set_ylim(0.5, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    top = header(fig, "Once a set starts, the tracker names it within about one rep",
                 "Set start given (pre-segmented sets). 95% CI over 23 athletes.", h, l)
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, "fig5_realtime")


def _athlete_ci(df, value="macro_f1"):
    """Mean and 95% CI over athletes (each athlete averaged over repeats first)."""
    return bootstrap_ci(df.groupby("athlete")[value].mean().to_numpy())


def fig6():
    """Everything on one axis: new-athlete macro-F1 vs recording budget (sets per exercise across athletes)."""
    AQUA = SERIES[2]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), sharey=True)
    for ax, m in zip(axes, ["rf", "cnn"]):
        summ = pd.read_csv(T / ("04_rf_summary.csv" if m == "rf" else "05_cnn_summary.csv"))
        summ = summ[summ.group == "new"]
        lines = {}
        # 4 sets per athlete (step 12; sets 3-4 from a second day)
        r12 = pd.read_csv(T / f"12_deep_broad_{m}_athletes.csv").query("depth == 4")
        d4 = [(n * 4, *_athlete_ci(g)) for n, g in r12.groupby("n_team")]
        lines["4 sets per athlete"] = (pd.DataFrame(d4, columns=["x", "mean", "lo", "hi"]), RAMP3[0])
        d2 = summ[summ.k == 2].assign(x=lambda d: d.n_team * 2)
        lines["2 sets per athlete"] = (d2.rename(columns={"ci_lo": "lo", "ci_hi": "hi"})[["x", "mean", "lo", "hi"]], RAMP3[1])
        d1 = summ[summ.k == 1].assign(x=lambda d: d.n_team)
        lines["More athletes, 1 set each"] = (d1.rename(columns={"ci_lo": "lo", "ci_hi": "hi"})[["x", "mean", "lo", "hi"]], RAMP3[2])
        end_labels = {"4 sets per athlete": "4 sets each", "2 sets per athlete": "2 sets each",
                      "More athletes, 1 set each": "1 set each"}
        for lab, (d, c) in lines.items():
            d = d.sort_values("x")
            ax.fill_between(d.x, d.lo, d.hi, color=c, alpha=0.15, lw=0)
            ax.plot(d.x, d["mean"], color=c, lw=2.5, marker="o", ms=7, mec=SURFACE, mew=2, label=lab)
            ax.annotate(end_labels[lab], (d.x.iloc[-1], d["mean"].iloc[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=12, color=INK2)
        if m == "rf":  # targeted depth (step 09): 1 set of everything + a 2nd set of the 3 presses = 18/15 sets per exercise
            tg = pd.read_csv(T / "09_targeted_athletes.csv").query("group == 'new' and condition == 'press'")
            for n, g in tg.groupby("n_team"):
                mm, lo, hi = _athlete_ci(g)
                x = n * 18 / 15
                ax.errorbar(x, mm, yerr=[[mm - lo], [hi - mm]], color=AQUA, lw=1.5, capsize=3, zorder=5)
                ax.plot(x, mm, "^", color=AQUA, ms=12, mec=SURFACE, mew=2, zorder=6,
                        label="Extra sets of the 3 presses" if n == 5 else None)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
        ax.set_xlim(1.7, 40)
        ax.set_xlabel("Recording budget (sets per exercise, all athletes combined)")
        ax.set_title(MODEL_NAME[m], loc="left", fontsize=17, color=INK)
    axes[0].set_ylabel("Macro-F1, new athletes")
    axes[0].set_ylim(0.4, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    order = [2, 1, 0, 3]  # 1 set, 2 sets, 4 sets, presses
    top = header(fig, "What a recording budget buys: spend it on more athletes",
                 "New athletes. The 1-set line stops at 17 (no more athletes to record); the 4-set line uses two recording days. 95% CI.",
                 [h[i] for i in order], [l[i] for i in order], ncol=4)
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, "fig6_budget_curves")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
        print("saved", f.__name__)
