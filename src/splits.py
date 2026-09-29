"""Athlete-grouped splits. Every generated split is checked for athlete overlap."""
import numpy as np


def assert_disjoint(meta, train_idx, test_idx):
    tr = set(meta.athlete.values[train_idx])
    te = set(meta.athlete.values[test_idx])
    overlap = tr & te
    assert not overlap, f"athlete overlap between train and test: {sorted(overlap)}"


def athlete_split(athletes, test_frac, rng):
    """Split a list of athlete ids into (train_pool, test) arrays."""
    a = np.array(sorted(set(athletes)))
    rng.shuffle(a)
    n_test = max(1, int(round(len(a) * test_frac)))
    return np.sort(a[n_test:]), np.sort(a[:n_test])
