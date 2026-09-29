"""Pair screen (plan section 4): athlete-grouped 5-fold CV, features + RF, full data, all eligible pairs.

    python -m src.screen
"""
import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from . import config as C
from .data import get_windows
from .plots import pair_heatmap, save
from .splits import assert_disjoint
from .utils import run_info

# Movement families, used only to prefer interpretable pairs within a difficulty band.
FAMILIES = {
    0: {"squat"}, 8: {"squat"}, 2: {"squat"},
    5: {"hinge"}, 6: {"hinge"}, 11: {"hinge", "pull"},
    7: {"overhead"}, 10: {"overhead"}, 14: {"overhead", "arm"},
    1: {"push"}, 4: {"push"},
    9: {"arm", "pull"},
    3: {"cardio"}, 12: {"cardio"},
    13: {"core"},
}
# Heatmap order: squat pattern, hinge/pull, overhead, horizontal push, arm, cardio, core
FAMILY_ORDER = [0, 8, 2, 5, 6, 11, 7, 10, 4, 1, 14, 9, 3, 12, 13]


def screen_pair(F, meta, a, b, seed=0):
    m = meta.cls.isin([a, b]).to_numpy()
    X, y, g = F[m], (meta.cls.to_numpy()[m] == b).astype(int), meta.athlete.to_numpy()[m]
    sub = meta[m].reset_index(drop=True)
    pred = np.empty_like(y)
    ba, f1 = [], []
    for tr, te in GroupKFold(n_splits=C.SCREEN_FOLDS).split(X, y, g):
        assert_disjoint(sub, tr, te)
        rf = RandomForestClassifier(C.RF_TREES, class_weight="balanced", n_jobs=1, random_state=seed)
        rf.fit(X[tr], y[tr])
        p = rf.predict(X[te])
        pred[te] = p
        ba.append(balanced_accuracy_score(y[te], p))
        f1.append(f1_score(y[te], p, average="macro"))
    return dict(class_a=a, class_b=b, name_a=C.CLASS_NAMES[a], name_b=C.CLASS_NAMES[b],
                balanced_acc=float(np.mean(ba)), balanced_acc_sd=float(np.std(ba)), macro_f1=float(np.mean(f1)),
                balanced_acc_pooled=balanced_accuracy_score(y, pred),
                n_windows_a=int((y == 0).sum()), n_windows_b=int((y == 1).sum()),
                n_athletes_a=sub[sub.cls == a].athlete.nunique(), n_athletes_b=sub[sub.cls == b].athlete.nunique(),
                same_family=bool(FAMILIES[a] & FAMILIES[b]))


def select_pairs(res):
    lines = ["# Pair selection", "",
             "Selected automatically from `results/pair_screen.csv` (full-data balanced accuracy, athlete-grouped 5-fold CV, RF).",
             "Rule: in each band, prefer same-movement-family pairs, then closeness to the band centre, then more windows in the smaller class.",
             "(All 15 classes have all 35 athletes, so athlete coverage does not discriminate between pairs.)",
             "Pair difficulty is descriptive (used to stratify the learning curves), not a model-selection step;",
             "learning curves are evaluated on separately held-out athletes in every repeat.", ""]
    chosen = {}
    for band, (lo, hi) in C.DIFFICULTY_BANDS.items():
        centre = (lo + min(hi, 1.0)) / 2
        allp = res.copy()
        allp["in_band"] = (allp.balanced_acc >= lo) & (allp.balanced_acc < hi)
        allp["min_windows"] = allp[["n_windows_a", "n_windows_b"]].min(axis=1)
        allp["dist"] = (allp.balanced_acc - centre).abs()
        cand = allp[allp.in_band].sort_values(["same_family", "dist", "min_windows"], ascending=[False, True, False])
        lines.append(f"## {band} (balanced accuracy {lo:.2f}-{min(hi, 1):.2f}, centre {centre:.3f}): "
                     f"{len(cand)} candidate pairs")
        if len(cand) < 3:  # flag and fill alternates with the nearest out-of-band pairs
            lines.append(f"Fewer than 3 pairs in band; alternates below include the nearest pairs outside it.")
            near = allp[~allp.in_band].assign(d=lambda d: np.minimum((d.balanced_acc - lo).abs(), (d.balanced_acc - hi).abs()))
            cand = pd.concat([cand, near.sort_values("d")])
        if cand.empty:
            lines.append("No pair in band.\n")
            continue
        for k, r in enumerate(cand.head(3).itertuples()):
            tag = "**Selected**" if k == 0 else f"Alternate {k}"
            where = "" if r.in_band else " [outside band]"
            lines.append(f"- {tag}: {r.name_a} vs {r.name_b}, balanced acc {r.balanced_acc:.3f} "
                         f"(SD over folds {r.balanced_acc_sd:.3f}), same family={r.same_family}, "
                         f"windows {r.n_windows_a}/{r.n_windows_b}{where}")
        top = cand.iloc[0]
        chosen[band] = (int(top.class_a), int(top.class_b))
        lines.append("")
    return chosen, "\n".join(lines)


def main():
    X, F, meta = get_windows()
    counts = meta.groupby("cls").athlete.nunique()
    eligible = sorted(counts[counts >= C.MIN_ATHLETES].index)
    pairs = list(itertools.combinations(eligible, 2))
    print(f"{len(meta)} windows, {len(eligible)} eligible classes, {len(pairs)} pairs")
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(screen_pair)(F, meta, a, b) for a, b in pairs)
    res = pd.DataFrame(rows).sort_values("balanced_acc").reset_index(drop=True)
    for k, v in run_info(seed=0).items():
        res[k] = v
    C.RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(C.RESULTS_DIR / "pair_screen.csv", index=False)
    chosen, md = select_pairs(res)
    (C.RESULTS_DIR / "pair_selection.md").write_text(md, encoding="utf-8")
    pd.DataFrame([dict(band=k, class_a=a, class_b=b) for k, (a, b) in chosen.items()]).to_csv(
        C.RESULTS_DIR / "selected_pairs.csv", index=False)
    save(pair_heatmap(res, order=FAMILY_ORDER, highlight=list(chosen.values())), "pair_screen")
    print(res[["name_a", "name_b", "balanced_acc", "macro_f1"]].to_string())
    print(md)


if __name__ == "__main__":
    main()
