"""Data audit for uLift (plan section 1). Writes results/audit_*.csv and prints a summary.

Run from repo root:  python scripts/audit.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/uLift-dataset/uLift_sensor_dataset")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

SEG_RE = re.compile(r"^(?P<nick>[^_]+)_(?P<year>\d{4})_(?P<date>\d{4})_(?P<time>\d{6})_segment(?P<num>\d+)_(?P<type>\d{2}|rest)$")
WHOLE_RE = re.compile(r"^(?P<nick>[^_]+)_(?P<year>\d{4})_(?P<date>\d{4})_(?P<time>\d{6})_whole$")

rows, unmatched = [], []
for csv in sorted(ROOT.rglob("*.csv")):
    stem = csv.stem
    m = SEG_RE.match(stem)
    if not m:
        if not WHOLE_RE.match(stem):
            unmatched.append(str(csv))
        continue
    if m["type"] == "rest":
        continue
    df = pd.read_csv(csv, header=None, names=["row", "t", "x", "y", "z"])
    info = (csv.with_suffix(".info")).read_text().strip().splitlines()
    basic = info[0].split(",")
    wk = info[1].split(",") if len(info) > 1 else [None, None, None]
    t = df["t"].to_numpy(dtype=np.int64)
    dt = np.diff(t)
    dur = (t[-1] - t[0]) / 1000.0
    mag = np.sqrt(df.x**2 + df.y**2 + df.z**2)
    rows.append(dict(
        file=stem, folder=csv.parent.name, nick=basic[0], session=f"{m['nick']}_{m['year']}_{m['date']}_{m['time']}",
        seg_num=int(m["num"]), cls_file=int(m["type"]), cls_info=int(wk[0]) if wk[0] else -1, cls_name=wk[1],
        reps=int(wk[2]) if wk[2] else -1, n=len(df), n_unique_t=len(np.unique(t)), dur_s=dur,
        rate_hz=(len(df) - 1) / dur if dur > 0 else np.nan,
        frac_dt0=float(np.mean(dt == 0)) if len(dt) else np.nan,
        dt_non0_median=float(np.median(dt[dt > 0])) if np.any(dt > 0) else np.nan,
        max_gap_ms=int(dt.max()) if len(dt) else 0, n_neg_dt=int(np.sum(dt < 0)),
        row_types=",".join(map(str, sorted(df.row.unique()))),
        mag_median=float(np.median(mag)), absmax=float(np.abs(df[["x", "y", "z"]]).to_numpy().max()),
        exp_months=basic[1], gender=basic[2],
    ))

seg = pd.DataFrame(rows)
seg.to_csv(OUT / "audit_segments.csv", index=False)

print("unmatched filenames:", len(unmatched), unmatched[:5])
print("workout segments:", len(seg), " folders:", seg.folder.nunique(), " nicks(info):", seg.nick.str.lower().nunique())
print("sessions:", seg.session.nunique(), " total reps:", seg.reps.sum())
print("class mismatch file vs info:", int((seg.cls_file != seg.cls_info).sum()))
print("folder != info nick:", int((seg.folder.str.lower() != seg.nick.str.lower()).sum()))
print("row types:", seg.row_types.unique())
print("segments with reps<=1:", int((seg.reps <= 1).sum()))
print("\nSampling rate (samples-1)/duration, Hz:\n", seg.rate_hz.describe(percentiles=[.01, .05, .5, .95, .99]).round(2))
print("fraction of zero dt (burst duplicates):", seg.frac_dt0.describe().round(3).to_dict())
print("n unique t / n:", (seg.n_unique_t / seg.n).describe().round(3).to_dict())
print("max gap ms:", seg.max_gap_ms.describe(percentiles=[.5, .95, .99]).round(0).to_dict())
print("negative dt segments:", int((seg.n_neg_dt > 0).sum()))
print("duration s:", seg.dur_s.describe(percentiles=[.05, .5, .95]).round(1).to_dict())
print("median |acc| (units):", seg.mag_median.describe().round(2).to_dict(), " global abs max:", seg.absmax.max())

per_class = seg.groupby(["cls_file", "cls_name"]).agg(
    segments=("file", "size"), reps=("reps", "sum"), athletes=("folder", "nunique"),
    sessions=("session", "nunique"), total_min=("dur_s", lambda s: s.sum() / 60),
    median_seg_s=("dur_s", "median"), rep_rate_hz=("reps", lambda r: np.nan),
).reset_index()
per_class["rep_rate_hz"] = seg.groupby("cls_file").apply(lambda g: (g.reps / g.dur_s).median()).values
per_ath = seg.groupby(["cls_file", "folder"]).size().unstack(fill_value=0)
per_class["segs_per_athlete_median"] = per_ath.replace(0, np.nan).median(axis=1).values
per_class["segs_per_athlete_min"] = per_ath.replace(0, np.nan).min(axis=1).values
per_class["segs_per_athlete_max"] = per_ath.max(axis=1).values
per_class.to_csv(OUT / "audit_per_class.csv", index=False)
per_ath.to_csv(OUT / "audit_class_by_athlete.csv")
print("\n", per_class.round(2).to_string(index=False))
