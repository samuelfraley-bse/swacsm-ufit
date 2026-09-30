"""Simulated athletes: physically motivated variations of a recorded set (bout), applied before windowing.
Also `um_windows`: the published window-level recipe of Um et al. (ICMI 2017), ported from their official code
(github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data) with numpy Generators for reproducibility.

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


def _um_timewarp(x, rng, sigma=0.2, knot=4):
    """Um et al. 2017 DA_TimeWarp: smooth random time distortion, independently per axis. x: (T, 3)."""
    from scipy.interpolate import CubicSpline

    T = len(x)
    xx = np.arange(0, T, (T - 1) / (knot + 1))
    out = np.empty_like(x)
    for k in range(3):
        yy = rng.normal(1.0, sigma, size=len(xx))
        tt = np.cumsum(CubicSpline(xx, yy)(np.arange(T)))
        tt *= (T - 1) / tt[-1]
        out[:, k] = np.interp(np.arange(T), tt, x[:, k])
    return out


def _um_permutation(x, rng, n_perm=4, min_seg=10):
    """Um et al. 2017 DA_Permutation: cut into n_perm segments (each > min_seg samples) and shuffle them."""
    T = len(x)
    while True:
        segs = np.zeros(n_perm + 1, dtype=int)
        segs[1:-1] = np.sort(rng.integers(min_seg, T - min_seg, n_perm - 1))
        segs[-1] = T
        if np.min(np.diff(segs)) > min_seg:
            break
    order = rng.permutation(n_perm)
    return np.concatenate([x[segs[i]:segs[i + 1]] for i in order])


def _um_rotation(x, rng):
    """Um et al. 2017 DA_Rotation: random axis, angle uniform in [-pi, pi]."""
    axis = rng.uniform(-1, 1, size=3)
    axis /= np.linalg.norm(axis)
    ang = rng.uniform(-np.pi, np.pi)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    return x @ R


def um_windows(Xw, n_copies, seed):
    """Um et al. (ICMI 2017) best recipe, Rot+Perm+TimeW, applied to windows with their default parameters
    (applied in the order time-warp -> permutation -> rotation). Xw: (n, 4, T) windows (x, y, z, |a|).
    Returns (n * n_copies, 4, T) with the magnitude channel recomputed."""
    rng = np.random.default_rng(seed)
    out = np.empty((len(Xw) * n_copies,) + Xw.shape[1:], dtype=np.float32)
    i = 0
    for w in Xw:
        xyz = w[:3].T.astype(float)
        for _ in range(n_copies):
            a = _um_rotation(_um_permutation(_um_timewarp(xyz, rng), rng), rng)
            out[i, :3] = a.T
            out[i, 3] = np.linalg.norm(a, axis=1)
            i += 1
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
