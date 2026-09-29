"""Build (and cache) the window arrays + features used by every experiment."""
import pickle

import numpy as np

from . import config as C
from .features import extract
from .load import load_all
from .windows import make_windows


def get_windows(window_s=C.WINDOW_S, overlap=C.OVERLAP, edge_trim_s=C.EDGE_TRIM_S, use_cache=True):
    """Return X (n,4,T) float32, F (n,59) float32, meta DataFrame."""
    C.CACHE_DIR.mkdir(exist_ok=True)
    cache = C.CACHE_DIR / f"windows_fs{int(C.FS)}_w{window_s:g}_o{overlap:g}_t{edge_trim_s:g}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    seg, signals = load_all()
    X, meta = make_windows(seg, signals, window_s, overlap, edge_trim_s)
    F = extract(X)
    assert np.isfinite(F).all()
    out = (X, F, meta)
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out
