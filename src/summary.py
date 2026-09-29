"""Athlete-level summaries: average each athlete over repeats, then bootstrap over athletes."""
import pandas as pd

from .metrics import bootstrap_ci


def summarize(res, metric="macro_f1", by=("model", "n_team", "k", "group")):
    by = list(by)
    ath = res.groupby(by + ["athlete"])[metric].agg(["mean", "size"]).reset_index()
    out = []
    for key, g in ath.groupby(by):
        m, lo, hi = bootstrap_ci(g["mean"].to_numpy())
        out.append(dict(zip(by, key), mean=m, ci_lo=lo, ci_hi=hi, median=g["mean"].median(),
                        n_athletes=len(g), n_athlete_repeats=int(g["size"].sum())))
    return pd.DataFrame(out)
