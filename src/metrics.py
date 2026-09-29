"""Bout-level scoring. Participants are the unit: metrics are computed per test athlete."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

N_CLASSES = 15
LABELS = list(range(N_CLASSES))


def vote_bouts(proba, meta):
    """Majority vote of window predictions within each bout; ties broken by mean probability.

    proba: (n_windows, 15) with columns in class order 0..14. meta: window meta (segment_id, athlete, cls).
    Returns one row per bout with true and predicted class.
    """
    votes = np.zeros_like(proba)
    votes[np.arange(len(proba)), proba.argmax(1)] = 1
    df = pd.DataFrame(votes + 1e-3 * proba, columns=LABELS)
    df["segment_id"] = meta.segment_id.to_numpy()
    agg = df.groupby("segment_id", sort=False)[LABELS].sum()
    first = meta.groupby("segment_id", sort=False)[["athlete", "cls"]].first()
    out = first.loc[agg.index].reset_index()
    out["pred"] = agg.to_numpy().argmax(1)
    out["n_windows"] = meta.groupby("segment_id", sort=False).size().loc[agg.index].to_numpy()
    return out.rename(columns={"cls": "true"})


def per_athlete(bouts):
    """Macro-F1 (over all 15 classes), balanced accuracy and accuracy for each test athlete."""
    rows = []
    for a, g in bouts.groupby("athlete"):
        rows.append(dict(athlete=a, n_bouts=len(g),
                         macro_f1=f1_score(g.true, g.pred, labels=LABELS, average="macro", zero_division=0),
                         balanced_acc=balanced_accuracy_score(g.true, g.pred),
                         accuracy=accuracy_score(g.true, g.pred)))
    return pd.DataFrame(rows)


def bootstrap_ci(x, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean, resampling units (athletes)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
