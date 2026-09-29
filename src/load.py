"""Load uLift workout segments and resample them to a uniform rate.

BLE arrival timestamps come in bursts (~35% of rows share a timestamp with a distinct sample), so we
keep every sample, assume even spacing between the first and last arrival time, low-pass, then
interpolate to config.FS. Units are m/s^2 (see STATUS.md).
"""
import pickle
import re

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from . import config as C

SEG_RE = re.compile(r"^(?P<nick>[^_]+)_(?P<year>\d{4})_(?P<date>\d{4})_(?P<time>\d{6})_segment(?P<num>\d+)_(?P<type>\d{2})$")


def list_segments(root=C.DATA_ROOT):
    """Return a DataFrame describing every workout segment file (rest/whole files excluded)."""
    rows = []
    for csv in sorted(root.rglob("*.csv")):
        m = SEG_RE.match(csv.stem)
        if not m:
            continue  # rest and whole-session files
        info = csv.with_suffix(".info").read_text().strip().splitlines()
        cls, name, reps = info[1].split(",")[:3]
        assert int(cls) == int(m["type"]), csv
        rows.append(dict(
            segment_id=csv.stem, athlete=csv.parent.name,
            session_id=f"{m['nick']}_{m['year']}_{m['date']}_{m['time']}",
            cls=int(cls), cls_name=name, reps=int(reps), path=csv.as_posix(),
        ))
    return pd.DataFrame(rows)


def resample_segment(t_ms, xyz, fs=C.FS, lowpass=C.LOWPASS_HZ):
    """Evenly re-time samples over [t0, t_end], low-pass, and linearly interpolate to fs. Returns (n, 3)."""
    n = len(t_ms)
    dur = (t_ms[-1] - t_ms[0]) / 1000.0
    if n < 16 or dur <= 0:
        return np.empty((0, 3))
    fs_raw = (n - 1) / dur
    x = xyz.astype(float)
    if lowpass < fs_raw / 2:
        sos = butter(4, lowpass, fs=fs_raw, output="sos")
        x = sosfiltfilt(sos, x, axis=0)
    t_raw = np.arange(n) / fs_raw
    t_new = np.arange(0, t_raw[-1] + 1e-9, 1.0 / fs)
    return np.column_stack([np.interp(t_new, t_raw, x[:, k]) for k in range(3)])


def load_all(use_cache=True):
    """Return (segments DataFrame with duration/rep_rate, list of (n,3) arrays at config.FS)."""
    C.CACHE_DIR.mkdir(exist_ok=True)
    cache = C.CACHE_DIR / f"segments_fs{int(C.FS)}_lp{C.LOWPASS_HZ:g}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    seg = list_segments()
    seg = seg[seg.reps >= C.MIN_REPS].reset_index(drop=True)
    signals, n_raw, fs_raw = [], [], []
    for p in seg.path:
        df = pd.read_csv(p, header=None, names=["row", "t", "x", "y", "z"])
        df = df[df.row == 1]
        t = df.t.to_numpy(np.int64)
        signals.append(resample_segment(t, df[["x", "y", "z"]].to_numpy()))
        n_raw.append(len(df))
        fs_raw.append((len(df) - 1) / ((t[-1] - t[0]) / 1000.0))
    seg["n_raw"] = n_raw
    seg["fs_raw"] = fs_raw
    seg["duration_s"] = [len(s) / C.FS for s in signals]
    seg["rep_rate_hz"] = seg.reps / seg.duration_s
    out = (seg, signals)
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out
