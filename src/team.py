"""New-team design: pick who is recorded, which day-1 sets they contribute, and the later-day test bouts.

Training data only ever comes from day 1 of recorded athletes; test data only from later days.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path("outputs/tables")


def load_manifest():
    m = pd.read_csv(TABLES / "manifest.csv")
    m = m[m.include].copy()
    m["segment_id"] = m.file_path.str.rsplit("/", n=1).str[-1].str.removesuffix(".csv")
    return m


def load_pool():
    p = pd.read_csv(TABLES / "audit_participants.csv", index_col=0)
    return sorted(p.index[p.in_pool])


def split_new(pool, n_new, rng):
    """Hold out n_new athletes as never recorded; return (team_candidates, new)."""
    a = np.array(pool)
    rng.shuffle(a)
    return sorted(a[n_new:]), sorted(a[:n_new])


def train_bouts(manifest, team, k, rng, classes=None):
    """Day-1 bouts of the team: k sets per (athlete, exercise), chosen at random. Optional class subset."""
    d1 = manifest[(manifest.day_order == 1) & manifest.participant_id.isin(team)]
    if classes is not None:
        d1 = d1[d1.exercise_class.isin(classes)]
    picked = []
    for _, g in d1.groupby(["participant_id", "exercise_class"], sort=True):
        if len(g) < k:
            raise ValueError(f"only {len(g)} day-1 sets for {g.participant_id.iloc[0]} class {g.exercise_class.iloc[0]}, need {k}")
        picked += list(rng.choice(g.segment_id.to_numpy(), size=k, replace=False))
    return sorted(picked)


def test_bouts(manifest, athletes):
    """All later-day bouts of the given athletes."""
    t = manifest[(manifest.day_order > 1) & manifest.participant_id.isin(athletes)]
    return sorted(t.segment_id)


def assert_no_leak(manifest, train_ids, test_ids, new):
    tr = manifest[manifest.segment_id.isin(train_ids)]
    te = manifest[manifest.segment_id.isin(test_ids)]
    assert not set(train_ids) & set(test_ids), "bout used for both training and testing"
    assert (tr.day_order == 1).all() and (te.day_order > 1).all(), "day split violated"
    assert not set(tr.participant_id) & set(new), "a 'new' athlete appears in training"
