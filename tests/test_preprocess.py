import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.features import FEATURE_NAMES, extract
from src.load import resample_segment
from src.splits import assert_disjoint, athlete_split
from src.windows import make_windows


def test_resample_rate_and_frequency():
    # 95 Hz bursty arrival, 1 Hz sine: output should be 30 Hz and preserve the 1 Hz oscillation.
    n, fs_raw = 950, 95.0
    t = np.repeat(np.arange(0, n, 3) * (1000 / fs_raw), 3)[:n].astype(np.int64)  # bursts of 3 identical stamps
    tt = np.arange(n) / fs_raw
    xyz = np.column_stack([np.sin(2 * np.pi * tt), np.zeros(n), np.full(n, 9.81)])
    out = resample_segment(t, xyz)
    assert abs(len(out) - int(t[-1] / 1000 * C.FS) - 1) <= 1
    f = np.fft.rfftfreq(len(out), 1 / C.FS)[np.abs(np.fft.rfft(out[:, 0])).argmax()]
    assert abs(f - 1.0) < 0.15
    assert np.allclose(out[:, 2], 9.81, atol=1e-6)


def test_windows_do_not_cross_segments():
    seg = pd.DataFrame(dict(athlete=["a", "b"], cls=[0, 1], segment_id=["s1", "s2"], session_id=["x", "y"],
                            rep_rate_hz=[0.5, 0.5]))
    sig = [np.random.randn(95, 3), np.random.randn(40, 3)]  # 3.17 s and 1.33 s at 30 Hz
    X, meta = make_windows(seg, sig, window_s=2.0, overlap=0.5)
    T = int(2.0 * C.FS)
    assert X.shape[1:] == (4, T)
    assert (meta.segment_id == "s1").sum() == len(range(0, 95 - T + 1, T // 2))
    assert meta.attrs["short_segments"] == ["s2"]
    assert np.allclose(X[:, 3], np.linalg.norm(X[:, :3], axis=1), atol=1e-5)


def test_features_shape_and_dominant_frequency():
    T = int(C.WINDOW_S * C.FS)
    t = np.arange(T) / C.FS
    X = np.zeros((2, 4, T))
    X[0, 0] = np.sin(2 * np.pi * 2.0 * t)
    X[1, 0] = np.sin(2 * np.pi * 5.0 * t)
    F = extract(X)
    assert F.shape == (2, len(FEATURE_NAMES))
    k = FEATURE_NAMES.index("x_domf")
    assert np.allclose(F[:, k], [2.0, 5.0], atol=0.5)
    assert np.isfinite(F).all()


def test_athlete_split_disjoint():
    rng = np.random.default_rng(0)
    ath = [f"s{i}" for i in range(35)]
    tr, te = athlete_split(ath, 0.3, rng)
    assert not set(tr) & set(te) and len(tr) + len(te) == 35 and len(te) == 10
    meta = pd.DataFrame(dict(athlete=["a", "a", "b"]))
    with pytest.raises(AssertionError):
        assert_disjoint(meta, np.array([0]), np.array([1]))
