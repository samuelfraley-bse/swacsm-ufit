"""Signal variants for step 07 ("what does the sensor use?").

Filtering is done on whole resampled bouts before windowing, so windows line up exactly with the
default windows (same bouts, same start times).
- full:      x, y, z, |a| (default)
- gravity:   0.3 Hz low-pass of each axis = wrist orientation relative to gravity (posture)
- motion:    signal minus its gravity component = movement dynamics
- magnitude: |a| only (1 channel), independent of how the watch is rotated on the wrist
For gravity and motion, the 4th channel is the magnitude of that component.
"""
import pickle

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import config as C
from .features import extract
from .load import load_all
from .windows import make_windows

GRAVITY_HZ = 0.3
VARIANTS = ["full", "gravity", "motion", "magnitude"]


def gravity_component(s, fs=C.FS, cutoff=GRAVITY_HZ):
    sos = butter(2, cutoff, fs=fs, output="sos")
    return sosfiltfilt(sos, s, axis=0)


def get_variant_windows(variant, use_cache=True):
    """Return (X, F, meta) for a variant; meta is identical to the default windows' meta."""
    from .data import get_windows

    if variant == "full":
        return get_windows()
    C.CACHE_DIR.mkdir(exist_ok=True)
    cache = C.CACHE_DIR / f"windows_variant_{variant}_fs{int(C.FS)}_w{C.WINDOW_S:g}_o{C.OVERLAP:g}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    X0, _, meta0 = get_windows()
    if variant == "magnitude":
        X = X0[:, 3:4].copy()
        meta = meta0
    else:
        seg, signals = load_all()
        g = [gravity_component(s) for s in signals]
        sig = g if variant == "gravity" else [s - gs for s, gs in zip(signals, g)]
        X, meta = make_windows(seg, sig)
        assert meta[["segment_id", "t0_s"]].equals(meta0[["segment_id", "t0_s"]]), "windows misaligned"
    F = extract(X)
    assert np.isfinite(F).all()
    out = (X, F, meta)
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out
