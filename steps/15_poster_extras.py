"""Step 15: poster extras from saved results / cached signals (no training).

  fig7_what_the_watch_sees   (a) raw wrist signal of two athletes doing a back squat (athletes chosen with a fixed
                             seed), split into posture (gravity, 0.3 Hz) and movement; (b) every pool athlete's
                             average wrist angle for 3 lifts (objective: no hand-picking)
  fig8_individual_athletes   every never-recorded athlete's curve (thin) + the mean (bold); RF and CNN, 1 set
  fig9_confusion             which lifts are mistaken for which: CNN, 8 athletes x 1 set, new athletes
  regression                 macro-F1 ~ athlete + log2(athletes recorded) + log2(sets per athlete), new athletes,
                             steps 04/05 (1-2 sets) + step 12 (4 sets); cluster bootstrap over athletes
                             -> outputs/tables/15_regression.csv

    python steps/15_poster_extras.py
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src import config as C  # noqa: E402
from src.load import load_all  # noqa: E402
from src.plots import BLUE_RAMP, INK, INK2, MUTED, SERIES, SURFACE, pretty, save  # noqa: E402
from src.screen import FAMILY_ORDER  # noqa: E402
from src.team import load_manifest, load_pool  # noqa: E402
from src.variants import gravity_component  # noqa: E402

T = ROOT / "outputs" / "tables"
BLUE = SERIES[0]
SEED = 7


def header(fig, text, sub):
    h_in = fig.get_size_inches()[1]
    fig.text(0.01, 0.99, text, ha="left", va="top", fontsize=20, color=INK)
    fig.text(0.01, 0.99 - 30 / 72 / h_in, sub, ha="left", va="top", fontsize=14, color=INK2)
    return 0.99 - 60 / 72 / h_in


# ---------------------------------------------------------------------------------------------------- fig 7
FEATURES = [  # (column, label, group)
    ("ux", "Wrist orientation, axis x", "Posture"),
    ("uz", "Wrist orientation, axis z", "Posture"),
    ("uy", "Wrist orientation, axis y", "Posture"),
    ("posture_change", "Orientation change within a set", "Posture"),
    ("intensity", "Movement intensity", "Movement"),
    ("tempo", "Movement tempo", "Movement"),
]


def set_features(seg, signals, pool):
    """One row per set (pool athletes): posture = mean unit gravity vector (0.3 Hz) + its within-set spread (deg);
    movement = RMS of |signal - gravity| (g) and its dominant frequency in 0.2-3 Hz (tempo)."""
    rows = []
    for r, s in zip(seg.itertuples(), signals):
        if r.athlete not in pool:
            continue
        g = gravity_component(s)
        u = g / np.linalg.norm(g, axis=1, keepdims=True)
        um = u.mean(0) / np.linalg.norm(u.mean(0))
        mot = np.linalg.norm(s - g, axis=1) / 9.81
        f = np.fft.rfftfreq(len(mot), 1 / C.FS)
        P = np.abs(np.fft.rfft(mot - mot.mean())) ** 2
        band = (f >= 0.2) & (f <= 3)
        rows.append(dict(athlete=r.athlete, cls=r.cls, segment_id=r.segment_id, ux=u[:, 0].mean(), uy=u[:, 1].mean(),
                         uz=u[:, 2].mean(), posture_change=np.degrees(np.arccos(np.clip(u @ um, -1, 1))).std(),
                         intensity=np.sqrt((mot ** 2).mean()), tempo=f[band][P[band].argmax()]))
    return pd.DataFrame(rows)


def decompose(D):
    """Share of each feature's variance: between exercises / between athletes within an exercise / set-to-set."""
    out = []
    for col, lab, grp in FEATURES:
        y = D[col]
        mc = D.groupby("cls")[col].transform("mean")
        mac = D.groupby(["cls", "athlete"])[col].transform("mean")
        tot = ((y - y.mean()) ** 2).sum()
        out.append(dict(feature=lab, group=grp, exercise=((mc - y.mean()) ** 2).sum() / tot,
                        athlete=((mac - mc) ** 2).sum() / tot, set_to_set=((y - mac) ** 2).sum() / tot))
    return pd.DataFrame(out)


def fig7():
    seg, signals = load_all()
    sig = dict(zip(seg.segment_id, signals))
    man = load_manifest()
    pool = load_pool()
    D = set_features(seg, signals, set(pool))
    V = decompose(D)
    V.to_csv(T / "15_variance_decomposition.csv", index=False)
    print(V.round(3).to_string(index=False), f"\n({len(D)} sets, {D.athlete.nunique()} athletes)")

    # Raw signal only (illustration, no claims): one athlete (fixed seed), first day-1 set of four lifts, same scale.
    pid = np.random.default_rng(SEED).choice(pool)
    lifts = [(6, "Deadlift"), (9, "Arm curl"), (4, "Bench press"), (10, "Military press")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.4), sharey=True)
    for ax, (c, name) in zip(axes, lifts):
        b = man[(man.participant_id == pid) & (man.exercise_class == c) & (man.day_order == 1)] \
            .sort_values(["session_order", "set_order_in_session"]).iloc[0]
        s = sig[b.segment_id][: int(8 * C.FS)] / 9.81
        t = np.arange(len(s)) / C.FS
        for k, (col, nm) in enumerate(zip(SERIES[:3], "xyz")):
            ax.plot(t, s[:, k], color=col, lw=2.0, label=f"{nm}-axis")
        ax.set_title(name, loc="left", fontsize=16, color=INK)
        ax.set_xlabel("Seconds")
        ax.set_xticks([0, 2, 4, 6, 8])
    axes[0].set_ylabel("Wrist acceleration (g)")
    h, l = axes[0].get_legend_handles_labels()
    top = header(fig, "What the watch sees",
                 "One athlete (chosen at random), first 8 s of one set of each lift. The two presses look alike, and are the most confused.")
    fig.legend(h, l, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.99, 0.99))
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, "fig7_what_the_watch_sees")
    return pid


# ---------------------------------------------------------------------------------------------------- fig 8
def fig8():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, (m, f) in zip(axes, [("Random forest", "04_rf_athletes.csv"), ("CNN", "05_cnn_athletes.csv")]):
        r = pd.read_csv(T / f).query("group == 'new' and k == 1")
        a = r.groupby(["athlete", "n_team"]).macro_f1.mean().unstack("n_team")
        worst = a.iloc[:, -1].idxmin()
        for pid, row in a.iterrows():
            ax.plot(row.index, row.values, color=MUTED if pid != worst else INK, lw=1.0 if pid != worst else 2.0,
                    alpha=0.55 if pid != worst else 0.9)
        ax.plot(a.columns, a.mean(), color=BLUE, lw=4, marker="o", ms=9, mec=SURFACE, mew=2, zorder=5, label="Mean")
        ax.annotate("mean", (a.columns[-1], a.mean().iloc[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=13, color=BLUE)
        ax.annotate("lowest athlete", (a.columns[-1], a.loc[worst].iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=12, color=INK)
        ax.set_xticks([2, 4, 6, 8, 12, 17])
        ax.set_xlim(1, 21)
        ax.set_xlabel("Athletes recorded (1 set per exercise)")
        ax.set_title(m, loc="left", fontsize=17, color=INK)
        lo8 = (a[8] < 0.8).sum()
        ax.text(0.98, 0.04, f"With 8 recorded: {lo8} of {len(a)} new athletes below 0.80", transform=ax.transAxes,
                ha="right", fontsize=12, color=INK2)
    axes[0].set_ylabel("Macro-F1, each new athlete")
    axes[0].set_ylim(0.2, 1.0)
    top = header(fig, "The tracker works for most athletes, not all",
                 "Each thin line is one never-recorded athlete (average over repeats); bold = mean of 23 athletes.")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, "fig8_individual_athletes")


# ---------------------------------------------------------------------------------------------------- fig 9
def fig9():
    b = pd.read_csv(T / "05_cnn_bouts.csv.gz").query("group == 'new' and n_team == 8 and k == 1")
    M = pd.crosstab(b.true, b.pred).reindex(index=range(15), columns=range(15), fill_value=0)
    M = M.loc[FAMILY_ORDER, FAMILY_ORDER]
    P = 100 * M.div(M.sum(axis=1), axis=0)
    cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff"] + BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(13, 11.5))
    ax.imshow(P.to_numpy(), cmap=cmap, vmin=0, vmax=100)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    names = [pretty(C.CLASS_NAMES[c]) for c in FAMILY_ORDER]
    ax.set_xticks(range(15), names, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(range(15), names, fontsize=12)
    ax.tick_params(length=0)
    for i in range(15):
        for j in range(15):
            v = P.iat[i, j]
            if v >= 1:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=11, color="#ffffff" if v > 55 else INK)
    k = [FAMILY_ORDER.index(c) for c in (7, 10, 4)]
    lo, hi = min(k) - 0.5, max(k) + 0.5
    ax.add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo, fill=False, ec=INK, lw=2.5))
    ax.set_xlabel("Tracker's call", fontsize=15)
    ax.set_ylabel("Actual exercise", fontsize=15)
    top = header(fig, "Which lifts get mixed up?",
                 "% of each exercise's sets given each label. New athletes; CNN trained on 8 athletes × 1 set. Box: the presses.")
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, "fig9_confusion")
    P.round(1).to_csv(T / "15_confusion_cnn_n8_k1.csv")


# ---------------------------------------------------------------------------------------------------- regression
def regression(n_boot=2000):
    out = []
    rng = np.random.default_rng(0)
    for m, f04, f12 in (("rf", "04_rf_athletes.csv", "12_deep_broad_rf_athletes.csv"),
                        ("cnn", "05_cnn_athletes.csv", "12_deep_broad_cnn_athletes.csv")):
        a = pd.read_csv(T / f04).query("group == 'new'")[["athlete", "repeat", "n_team", "k", "macro_f1"]]
        d = pd.read_csv(T / f12).query("depth == 4")[["athlete", "repeat", "n_team", "depth", "macro_f1"]].rename(columns={"depth": "k"})
        df = pd.concat([a, d]).groupby(["athlete", "n_team", "k"]).macro_f1.mean().reset_index()
        df["la"], df["ls"] = np.log2(df.n_team), np.log2(df.k)
        athletes = df.athlete.unique()

        def fit(sub):
            # athlete intercepts (within-athlete) + two slopes
            X = pd.get_dummies(sub.athlete).astype(float)
            X["la"], X["ls"] = sub.la.to_numpy(), sub.ls.to_numpy()
            beta, *_ = np.linalg.lstsq(X.to_numpy(), sub.macro_f1.to_numpy(), rcond=None)
            pred = X.to_numpy() @ beta
            ss_res = ((sub.macro_f1 - pred) ** 2).sum()
            ss_tot = ((sub.macro_f1 - sub.groupby("athlete").macro_f1.transform("mean")) ** 2).sum()
            return beta[-2], beta[-1], 1 - ss_res / ss_tot

        ba, bs, r2 = fit(df)
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(athletes, len(athletes), replace=True)
            sub = pd.concat([df[df.athlete == p].assign(athlete=f"{p}#{i}") for i, p in enumerate(pick)])
            boots.append(fit(sub)[:2])
        B = np.array(boots)
        ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))  # noqa: E731
        out.append(dict(model=m, per_doubling_athletes=ba, athletes_lo=ci(B[:, 0])[0], athletes_hi=ci(B[:, 0])[1],
                        per_doubling_sets=bs, sets_lo=ci(B[:, 1])[0], sets_hi=ci(B[:, 1])[1],
                        shift_one_doubling_sets_to_athletes=ba - bs, shift_lo=ci(B[:, 0] - B[:, 1])[0],
                        shift_hi=ci(B[:, 0] - B[:, 1])[1], ratio=ba / bs, ratio_lo=ci(B[:, 0] / B[:, 1])[0],
                        ratio_hi=ci(B[:, 0] / B[:, 1])[1], within_athlete_r2=r2, n_rows=len(df)))
    res = pd.DataFrame(out)
    res.to_csv(T / "15_regression.csv", index=False)
    print(res.round(3).T.to_string())


if __name__ == "__main__":
    print("fig7 athletes:", fig7())
    fig8()
    fig9()
    regression()
