"""Self-training (pseudo-labeling) with a random forest, at the set (bout) level.

Fixed recipe: fit on labeled windows; predict unlabeled windows; average window probabilities within each
set; the set's pseudo-label is the argmax and its confidence the max. Keep the most confident FRACTIONS[i]
of sets *per predicted class* (so the model cannot drift toward classes it already favors), add all their
windows with the pseudo-label, refit. Each round re-predicts every unlabeled set with the latest model.
"""
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FRACTIONS = (0.5, 0.8)


def _fit(F, y, mask, seed, trees):
    rf = RandomForestClassifier(trees, class_weight="balanced", n_jobs=1, random_state=seed)
    return rf.fit(F[mask], y[mask])


def self_train(F, y, seg, lab_mask, unl_mask, seed, trees=300, fractions=FRACTIONS):
    """y is only read at labeled windows (and, for diagnostics, to score pseudo-labels).

    Returns (final model, list of per-round diagnostics).
    """
    rf = _fit(F, y, lab_mask, seed, trees)
    unl_idx = np.flatnonzero(unl_mask)
    diag = []
    for rnd, frac in enumerate(fractions, 1):
        P = rf.predict_proba(F[unl_idx])
        df = pd.DataFrame(P, columns=rf.classes_)
        df["seg"] = seg[unl_idx]
        bout = df.groupby("seg").mean()
        pred = bout.to_numpy().argmax(1)
        conf = bout.to_numpy().max(1)
        b = pd.DataFrame(dict(seg=bout.index, pred=rf.classes_[pred], conf=conf))
        keep = []
        for c, g in b.groupby("pred"):
            keep.append(g.nlargest(math.ceil(frac * len(g)), "conf"))
        keep = pd.concat(keep)
        # training set = labeled windows + windows of kept sets with their pseudo-labels
        y_train = y.copy()
        pseudo_mask = np.isin(seg, keep.seg.to_numpy())
        y_train[pseudo_mask] = pd.Series(keep.pred.to_numpy(), index=keep.seg).reindex(seg[pseudo_mask]).to_numpy()
        rf = _fit(F, y_train, lab_mask | pseudo_mask, seed, trees)
        truth = pd.Series(y[unl_idx], index=seg[unl_idx]).groupby(level=0).first()
        diag.append(dict(round=rnd, fraction=frac, n_unlabeled_sets=len(b), n_pseudo_sets=len(keep),
                         pseudo_acc_kept=float((keep.pred.to_numpy() == truth.loc[keep.seg].to_numpy()).mean()),
                         pseudo_acc_all=float((b.pred.to_numpy() == truth.loc[b.seg].to_numpy()).mean())))
    return rf, diag
