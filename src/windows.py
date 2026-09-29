"""Cut resampled segments into fixed-length windows with a magnitude channel."""
import numpy as np
import pandas as pd

from . import config as C


def add_magnitude(xyz):
    return np.column_stack([xyz, np.linalg.norm(xyz, axis=1)])


def make_windows(seg, signals, window_s=C.WINDOW_S, overlap=C.OVERLAP, edge_trim_s=C.EDGE_TRIM_S, fs=C.FS):
    """Return X (n_windows, 4, T) float32 and meta DataFrame (one row per window).

    Windows never cross segment boundaries. Segments shorter than one window yield none (logged in
    the returned meta.attrs['short_segments']).
    """
    T = int(round(window_s * fs))
    step = max(1, int(round(T * (1 - overlap))))
    trim = int(round(edge_trim_s * fs))
    Xs, rows, short = [], [], []
    for i, s in enumerate(signals):
        s = s[trim: len(s) - trim] if trim else s
        if len(s) < T:
            short.append(seg.segment_id.iloc[i])
            continue
        s4 = add_magnitude(s)
        starts = np.arange(0, len(s4) - T + 1, step)
        Xs.append(np.stack([s4[a:a + T].T for a in starts]))
        r = seg.iloc[i]
        for a in starts:
            rows.append((r.athlete, r.cls, r.segment_id, r.session_id, r.rep_rate_hz, a / fs))
    X = np.concatenate(Xs).astype(np.float32)
    meta = pd.DataFrame(rows, columns=["athlete", "cls", "segment_id", "session_id", "rep_rate_hz", "t0_s"])
    meta.attrs["short_segments"] = short
    meta.attrs["window_s"] = window_s
    return X, meta
