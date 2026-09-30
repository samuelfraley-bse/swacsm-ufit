"""Coach's-rules baseline: each exercise has a typical profile on four plain-language measures of a whole set,
and a new set is called whichever exercise it most resembles. No machine learning beyond averaging.

Measures (per set, from the 30 Hz signal; gravity = 0.3 Hz low-pass):
  forearm_direction  mean of the gravity direction along the sensor's x-axis (about +1 with the arms hanging,
                     about -1 with the hands overhead in this dataset; read from the data, not from device docs)
  forearm_turning    SD (degrees) of the wrist's orientation around its average during the set
  rep_speed_hz       dominant frequency (0.2-3 Hz) of the movement signal |a - gravity|
  movement_size_g    RMS of the movement signal (g)

Profiles = per-exercise medians over the recorded athletes' training sets; distances are scaled by the pooled
within-exercise SD of each measure (also from training sets only).
"""
import numpy as np
import pandas as pd

from . import config as C
from .variants import gravity_component

MEASURES = ["forearm_direction", "forearm_turning", "rep_speed_hz", "movement_size_g"]


def set_measures(seg, signals):
    rows = []
    for r, s in zip(seg.itertuples(), signals):
        g = gravity_component(s)
        u = g / np.linalg.norm(g, axis=1, keepdims=True)
        um = u.mean(0) / np.linalg.norm(u.mean(0))
        mot = np.linalg.norm(s - g, axis=1) / 9.81
        f = np.fft.rfftfreq(len(mot), 1 / C.FS)
        P = np.abs(np.fft.rfft(mot - mot.mean())) ** 2
        band = (f >= 0.2) & (f <= 3)
        rows.append(dict(segment_id=r.segment_id, athlete=r.athlete, cls=r.cls,
                         forearm_direction=u[:, 0].mean(),
                         forearm_turning=np.degrees(np.arccos(np.clip(u @ um, -1, 1))).std(),
                         rep_speed_hz=f[band][P[band].argmax()] if band.any() else np.nan,
                         movement_size_g=np.sqrt((mot ** 2).mean())))
    return pd.DataFrame(rows)


class CoachRules:
    def fit(self, M):
        self.templates = M.groupby("cls")[MEASURES].median()
        self.scale = np.sqrt(M.groupby("cls")[MEASURES].var().mean()) + 1e-9
        return self

    def predict(self, M):
        d = (((M[MEASURES].to_numpy()[:, None, :] - self.templates.to_numpy()[None]) / self.scale.to_numpy()) ** 2).sum(-1)
        return self.templates.index.to_numpy()[np.nanargmin(d, axis=1)]
