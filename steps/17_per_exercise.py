"""Step 17: which exercises need more athletes? Per-exercise recall for new athletes vs athletes recorded.

From saved set-level predictions (no training): logistic regression (step 16), RF (step 04), CNN (step 05);
new athletes, 1 set per exercise. Recall per exercise = share of that exercise's sets named correctly,
averaged per athlete over repeats, then over athletes (95% bootstrap CI over athletes).

Groups (thresholds chosen after seeing the logistic-regression numbers, then applied unchanged to RF and CNN):
  easy                 recall >= 0.90 with 4 athletes recorded
  stuck                recall < 0.80 even with 17 athletes recorded
  needs more athletes  everything else

Figures: fig10_exercise_groups (thin line per exercise, bold group mean; logistic regression),
         fig11_exercise_panels (small multiples, one panel per exercise; all three models).

    python steps/17_per_exercise.py
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

from src import config as C  # noqa: E402
from src.metrics import bootstrap_ci  # noqa: E402
from src.plots import INK, INK2, MUTED, SERIES, SURFACE, pretty, save  # noqa: E402

T = ROOT / "outputs" / "tables"
N_GRID = [2, 4, 6, 8, 12, 16, 17]
FILES = {"logreg": ("16_baselines_bouts.csv.gz", "Logistic regression"), "rf": ("04_rf_bouts.csv.gz", "Random forest"),
         "cnn": ("05_cnn_bouts.csv.gz", "CNN")}
GROUPS = {"easy": ("Easy", SERIES[2]), "more": ("Needs more athletes", SERIES[0]), "stuck": ("Stuck", SERIES[1])}
EASY_AT, EASY_THR, STUCK_THR = 4, 0.90, 0.80


def recall_table():
    rows = []
    for m, (f, _) in FILES.items():
        b = pd.read_csv(T / f)
        if "model" in b.columns:
            b = b[b.model == m]
        b = b[(b.group == "new") & (b.k == 1)]
        ath = b.assign(ok=b.true == b.pred).groupby(["true", "n_team", "athlete"]).ok.mean()
        for (c, n), g in ath.groupby(level=[0, 1]):
            mm, lo, hi = bootstrap_ci(g.to_numpy())
            rows.append(dict(model=m, cls=c, exercise=pretty(C.CLASS_NAMES[c]), n_team=n, recall=mm, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def assign_groups(R):
    out = []
    for (m, c), g in R.groupby(["model", "cls"]):
        r = g.set_index("n_team").recall
        grp = "easy" if r[EASY_AT] >= EASY_THR else ("stuck" if r[17] < STUCK_THR else "more")
        out.append(dict(model=m, cls=c, exercise=g.exercise.iloc[0], group=grp, recall_2=r[2], recall_4=r[4],
                        recall_17=r[17], gain_2_17=r[17] - r[2]))
    return pd.DataFrame(out)


def fig10(R, G, model="logreg"):
    fig, ax = plt.subplots(figsize=(13, 8))
    Rm, Gm = R[R.model == model], G[G.model == model].set_index("cls")
    ypos = {}
    for key, (lab, col) in GROUPS.items():
        members = Gm.index[Gm.group == key]
        for c in members:
            d = Rm[Rm.cls == c].sort_values("n_team")
            ax.plot(d.n_team, d.recall, color=col, lw=1.4, alpha=0.45)
        mean = Rm[Rm.cls.isin(members)].groupby("n_team").recall.mean().reindex(N_GRID)
        ax.plot(mean.index, mean.values, color=col, lw=4.5, marker="o", ms=10, mec=SURFACE, mew=2, zorder=5)
        ypos[key] = (mean.iloc[-1], lab, col, ", ".join(sorted(Gm.loc[members, "exercise"])))
    # group labels at the right, with member lists
    for key, (y, lab, col, names) in ypos.items():
        ax.annotate(f"{lab}", (17, y), xytext=(14, 8), textcoords="offset points", fontsize=15, color=col,
                    fontweight="bold", va="center")
        ax.annotate(names, (17, y), xytext=(14, -12), textcoords="offset points", fontsize=11, color=INK2, va="center",
                    wrap=True)
    ax.axhline(0.9, color=MUTED, lw=1, ls=(0, (2, 3)))
    ax.set_xticks([2, 4, 6, 8, 12, 17])
    ax.set_xlim(1.2, 30)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Athletes recorded (1 set per exercise)")
    ax.set_ylabel("Share of sets named correctly, new athletes")
    for s in ("right",):
        ax.spines[s].set_visible(False)
    h_in = fig.get_size_inches()[1]
    fig.text(0.01, 0.99, "Which lifts need more athletes?", ha="left", va="top", fontsize=20, color=INK)
    fig.text(0.01, 0.99 - 30 / 72 / h_in,
             f"{FILES[model][1]}; thin line = one exercise, bold = group mean. Easy: ≥0.90 with 4 athletes; "
             "stuck: <0.80 even with 17.", ha="left", va="top", fontsize=13, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.99 - 62 / 72 / h_in))
    save(fig, "fig10_exercise_groups")


def fig11(R, G):
    order = G[G.model == "logreg"].assign(o=lambda d: d.group.map({"easy": 0, "more": 1, "stuck": 2})) \
        .sort_values(["o", "recall_17"], ascending=[True, False])
    fig, axes = plt.subplots(3, 6, figsize=(20, 10.5), sharex=True, sharey=True)
    rows = {"easy": 0, "more": 1, "stuck": 2}
    used = {0: 0, 1: 0, 2: 0}
    mstyle = {"logreg": (INK, "-", 2.6), "rf": (MUTED, (0, (4, 2)), 1.8), "cnn": (MUTED, (0, (1, 1.5)), 1.8)}
    for r in order.itertuples():
        i = rows[r.group]
        ax = axes[i, used[i]]
        used[i] += 1
        for m, (col, ls, lw) in mstyle.items():
            d = R[(R.model == m) & (R.cls == r.cls)].sort_values("n_team")
            if m == "logreg":
                ax.fill_between(d.n_team, d.ci_lo, d.ci_hi, color=GROUPS[r.group][1], alpha=0.2, lw=0)
            ax.plot(d.n_team, d.recall, color=GROUPS[r.group][1] if m == "logreg" else col, ls=ls, lw=lw,
                    label=FILES[m][1])
        ax.axhline(0.9, color=MUTED, lw=0.8, ls=(0, (2, 3)))
        ax.set_title(r.exercise, loc="left", fontsize=13, color=INK)
        ax.set_xticks([2, 8, 17])
    for i in range(3):
        for j in range(used[i], 6):
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(GROUPS[list(rows)[i]][0], fontsize=14, color=GROUPS[list(rows)[i]][1])
    axes[0, 0].set_ylim(0.2, 1.02)
    from matplotlib.lines import Line2D

    h = [Line2D([], [], color=INK, lw=2.6), Line2D([], [], color=MUTED, ls=(0, (4, 2)), lw=1.8),
         Line2D([], [], color=MUTED, ls=(0, (1, 1.5)), lw=1.8)]
    l = ["Logistic regression (colored by group)", "Random forest", "CNN"]
    h_in = fig.get_size_inches()[1]
    fig.text(0.01, 0.99, "Recall by exercise, new athletes", ha="left", va="top", fontsize=20, color=INK)
    fig.text(0.01, 0.99 - 30 / 72 / h_in, "x = athletes recorded (1 set per exercise); y = share of sets named correctly; "
             "band = 95% CI (logistic regression).", ha="left", va="top", fontsize=13, color=INK2)
    fig.legend(h, l, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.99, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.99 - 60 / 72 / h_in))
    save(fig, "fig11_exercise_panels")


def main():
    R = recall_table()
    G = assign_groups(R)
    R.to_csv(T / "17_per_exercise_recall.csv", index=False)
    G.to_csv(T / "17_exercise_groups.csv", index=False)
    piv = G.pivot(index="exercise", columns="model", values="group")[["logreg", "rf", "cnn"]]
    piv["agree"] = piv.nunique(axis=1) == 1
    print(piv.sort_values("logreg").to_string())
    print(f"\nall three models agree on {int(piv.agree.sum())} of 15 exercises")
    print(G[G.model == "logreg"].sort_values("recall_17")[["exercise", "group", "recall_2", "recall_4", "recall_17", "gain_2_17"]]
          .round(2).to_string(index=False))
    fig10(R, G)
    fig11(R, G)


if __name__ == "__main__":
    main()
