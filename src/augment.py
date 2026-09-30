"""Simulated athletes: physically motivated variations of a recorded set (bout), applied before windowing.

- tempo:      the whole set is re-timed by a factor in TEMPO (faster / slower lifter)
- intensity:  the movement component (signal minus its 0.3 Hz gravity component) is scaled by a factor in
              INTENSITY; gravity is left unchanged (a more / less vigorous lifter, same posture)
- placement:  optional small random 3-D rotation of the whole signal (watch worn slightly differently)
"""
import numpy as np
import pandas as pd

from . import config as C
from .variants import gravity_component
from .windows import make_windows

TEMPO = (0.8, 1.2)
INTENSITY = (0.8, 1.2)
ROTATION_DEG = 15.0


def random_rotation(rng, max_deg):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K  # Rodrigues


def simulate(sig, rng, rotate=False):
    """sig: (n, 3) bout at C.FS. Returns a simulated bout (m, 3)."""
    g = gravity_component(sig)
    motion = sig - g
    speed = rng.uniform(*TEMPO)
    n = len(sig)
    m = max(2, int(round(n / speed)))
    t_old, t_new = np.arange(n), np.linspace(0, n - 1, m)
    warp = lambda a: np.column_stack([np.interp(t_new, t_old, a[:, k]) for k in range(3)])  # noqa: E731
    out = warp(g) + rng.uniform(*INTENSITY) * warp(motion)
    if rotate:
        out = out @ random_rotation(rng, ROTATION_DEG).T
    return out


def simulated_windows(seg_df, signals_by_id, bout_ids, n_copies, seed, rotate=False):
    """Make n_copies simulated versions of each bout and window them like the real data.

    Returns (X (n,4,T), labels (n,)). Deterministic given seed.
    """
    rng = np.random.default_rng(seed)
    rows, sigs = [], []
    info = seg_df.set_index("segment_id")
    for b in bout_ids:
        for j in range(n_copies):
            sigs.append(simulate(signals_by_id[b], rng, rotate))
            r = info.loc[b]
            rows.append(dict(athlete=r.athlete, cls=r.cls, segment_id=f"{b}#sim{j}", session_id=r.session_id,
                             rep_rate_hz=r.rep_rate_hz))
    X, meta = make_windows(pd.DataFrame(rows), sigs)
    return X, meta.cls.to_numpy()
