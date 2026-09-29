"""Hand-crafted per-window features (vectorized over windows).

Per channel (x, y, z, |a|): mean, std, min, max, range, IQR, RMS, skewness, kurtosis, zero-crossings
(of the mean-removed signal), mean absolute jerk, dominant frequency, its power, spectral entropy.
Plus inter-axis correlations (xy, xz, yz). 4*14 + 3 = 59 features.
"""
import numpy as np
from scipy.stats import kurtosis, skew

from . import config as C

STATS = ["mean", "std", "min", "max", "range", "iqr", "rms", "skew", "kurt", "zc", "jerk", "domf", "domp", "spent"]
CHANNELS = ["x", "y", "z", "mag"]
FEATURE_NAMES = [f"{c}_{s}" for c in CHANNELS for s in STATS] + ["corr_xy", "corr_xz", "corr_yz"]


def _corr(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return np.where(den > 0, (a * b).sum(-1) / np.maximum(den, 1e-12), 0.0)


def extract(X, fs=C.FS):
    """X: (n, 4, T) -> (n, 59) float32."""
    X = X.astype(np.float64)
    n, ch, T = X.shape
    mean = X.mean(-1)
    std = X.std(-1)
    mn, mx = X.min(-1), X.max(-1)
    q75, q25 = np.percentile(X, [75, 25], axis=-1)
    rms = np.sqrt((X ** 2).mean(-1))
    with np.errstate(all="ignore"):
        sk = np.nan_to_num(skew(X, axis=-1))
        ku = np.nan_to_num(kurtosis(X, axis=-1))
    Xc = X - mean[..., None]
    zc = (np.diff(np.signbit(Xc), axis=-1) != 0).sum(-1) / T
    jerk = np.abs(np.diff(X, axis=-1)).mean(-1) * fs
    P = np.abs(np.fft.rfft(Xc, axis=-1)) ** 2
    P = P[..., 1:]  # drop DC
    freqs = np.fft.rfftfreq(T, 1 / fs)[1:]
    k = P.argmax(-1)
    domf = freqs[k]
    Psum = P.sum(-1)
    domp = np.take_along_axis(P, k[..., None], -1)[..., 0] / np.maximum(Psum, 1e-12)
    p = P / np.maximum(Psum[..., None], 1e-12)
    spent = -(p * np.log(np.maximum(p, 1e-12))).sum(-1) / np.log(P.shape[-1])
    per = np.stack([mean, std, mn, mx, mx - mn, q75 - q25, rms, sk, ku, zc, jerk, domf, domp, spent], -1)
    per = per.reshape(n, ch * len(STATS))
    corr = np.column_stack([_corr(X[:, 0], X[:, 1]), _corr(X[:, 0], X[:, 2]), _corr(X[:, 1], X[:, 2])])
    return np.hstack([per, corr]).astype(np.float32)
