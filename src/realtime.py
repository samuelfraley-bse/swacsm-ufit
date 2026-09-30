"""Real-time recognition: the tracker's running call as a set unfolds.

Windows are 2 s long with a new one every 1 s. At time t into a set, the running call is the same vote
used for whole sets (src.metrics.vote_bouts: one vote per window, ties broken by mean probability) over the
windows that have ended by t. At the end of the set it equals the whole-set call exactly.
"""
import numpy as np
import pandas as pd

from . import config as C

TIMES_S = [2, 3, 4, 6, 8, 10, 15]


def running_calls(proba, meta):
    """Return a DataFrame with one row per window: segment_id, athlete, true, end_s, running call."""
    df = meta[["segment_id", "athlete", "cls", "t0_s", "rep_rate_hz"]].reset_index(drop=True).copy()
    votes = np.zeros_like(proba)
    votes[np.arange(len(proba)), proba.argmax(1)] = 1
    score = votes + 1e-3 * proba
    df["end_s"] = df.t0_s + C.WINDOW_S
    order = np.lexsort((df.t0_s.to_numpy(), df.segment_id.to_numpy()))
    df, score = df.iloc[order].reset_index(drop=True), score[order]
    calls = np.empty(len(df), dtype=int)
    for _, idx in df.groupby("segment_id", sort=False).indices.items():
        calls[idx] = np.cumsum(score[idx], axis=0).argmax(1)
    df["call"] = calls
    return df.rename(columns={"cls": "true"})


def calls_at_times(run, times=TIMES_S):
    """One row per (set, t): the call using windows ended by t (the final call if the set ended before t)."""
    rows = []
    for seg, g in run.groupby("segment_id", sort=False):
        ends, calls = g.end_s.to_numpy(), g.call.to_numpy()
        base = dict(segment_id=seg, athlete=g.athlete.iloc[0], true=g.true.iloc[0], rep_rate_hz=g.rep_rate_hz.iloc[0],
                    set_duration_s=ends[-1])
        for t in times:
            k = np.searchsorted(ends, t, side="right")  # number of windows ended by t
            rows.append(dict(base, t_s=t, call=calls[max(k, 1) - 1]))
        rows.append(dict(base, t_s=np.inf, call=calls[-1]))  # end of set
        # time to a stable correct call: first window after which every running call is correct
        ok = calls == base["true"]
        if ok[-1]:
            first_bad_from_end = np.flatnonzero(~ok)
            j = first_bad_from_end[-1] + 1 if len(first_bad_from_end) else 0
            stable = ends[j]
        else:
            stable = np.inf
        rows[-1]["stable_s"] = stable
    return pd.DataFrame(rows)
