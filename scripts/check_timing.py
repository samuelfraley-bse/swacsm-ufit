"""Timing check (plan section 1, item 2): are same-timestamp rows distinct samples, and how far do
evenly-spaced sample times deviate from BLE arrival times?"""
from pathlib import Path

import numpy as np
import pandas as pd

A = pd.read_csv("results/audit_segments.csv")
A = A[A.reps > 1]
rng = np.random.default_rng(0)
out = []
for f, folder in A.sample(400, random_state=0)[["file", "folder"]].itertuples(index=False):
    df = pd.read_csv(Path("data/uLift-dataset/uLift_sensor_dataset") / folder / f"{f}.csv", header=None,
                     names=["row", "t", "x", "y", "z"])
    t = df.t.to_numpy(float)
    v = df[["x", "y", "z"]].to_numpy()
    same_t = np.diff(t) == 0
    same_v = np.all(np.diff(v, axis=0) == 0, axis=1)
    lin = np.linspace(t[0], t[-1], len(t))
    dev = np.abs(lin - t)
    out.append(dict(frac_same_t=same_t.mean(), frac_same_t_and_v=(same_t & same_v).mean(),
                    frac_same_v_any=same_v.mean(), dev_ms_max=dev.max(), dev_ms_p95=np.percentile(dev, 95)))
o = pd.DataFrame(out)
print(o.describe(percentiles=[.5, .95, .99]).round(4).T.to_string())
