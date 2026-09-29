"""Step 01: build a manifest with one row per segmented workout bout.

Reads every workout-segment CSV and its .info file (rest and whole-session files are skipped) and
writes outputs/tables/manifest.csv. No filtering here: bouts with <=1 rep are kept and flagged, so
the audit (step 02) can report them.

    python steps/01_manifest.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

OUT = ROOT / "outputs" / "tables"
SEG_RE = re.compile(r"^(?P<nick>[^_]+)_(?P<year>\d{4})_(?P<mmdd>\d{4})_(?P<hms>\d{6})_segment(?P<num>\d+)_(?P<type>\d{2})$")


def read_bout(csv):
    """Return per-file stats, or an 'error' string if the file cannot be parsed."""
    try:
        df = pd.read_csv(csv, header=None, names=["row", "t", "x", "y", "z"])
    except Exception as e:  # corrupt / unreadable
        return dict(error=f"read: {e}")
    problems = []
    if df[["t", "x", "y", "z"]].isna().any().any():
        problems.append("missing values")
    if (df.row != 1).any():
        problems.append("non-sensor rows")
    t = df.t.to_numpy(np.int64)
    dt = np.diff(t)
    if (dt < 0).any():
        problems.append("timestamps go backwards")
    dur = (t[-1] - t[0]) / 1000.0 if len(t) > 1 else 0.0
    if dur <= 0:
        problems.append("zero duration")
    return dict(
        n_samples=len(df), duration_seconds=dur,
        estimated_sampling_rate=(len(df) - 1) / dur if dur > 0 else np.nan,
        frac_shared_timestamp=float((dt == 0).mean()) if len(dt) else np.nan,
        max_gap_ms=int(dt.max()) if len(dt) else 0,
        error="; ".join(problems),
    )


def main():
    rows = []
    for csv in sorted((ROOT / C.DATA_ROOT).rglob("*.csv")):
        m = SEG_RE.match(csv.stem)
        if not m:
            continue  # *_rest.csv and *_whole.csv
        info_lines = csv.with_suffix(".info").read_text().strip().splitlines()
        nick, exp, gender, birth, weight, height = info_lines[0].split(",")[:6]
        cls, name, reps = info_lines[1].split(",")[:3]
        rows.append(dict(
            participant_id=csv.parent.name,
            session_id=f"{m['nick']}_{m['year']}_{m['mmdd']}_{m['hms']}",
            session_date=pd.Timestamp(f"{m['year']}-{m['mmdd'][:2]}-{m['mmdd'][2:]}"),
            session_time=m["hms"],
            set_order_in_session=int(m["num"]),
            exercise_class=int(cls), exercise_name=name, exercise_class_from_filename=int(m["type"]),
            rep_count=int(reps),
            info_nickname=nick, experience_months=pd.to_numeric(exp, errors="coerce"), gender=gender,
            birth_year=pd.to_numeric(birth, errors="coerce"), weight_kg=pd.to_numeric(weight, errors="coerce"),
            height_cm=pd.to_numeric(height, errors="coerce"),
            file_path=csv.relative_to(ROOT).as_posix(),
            **read_bout(csv),
        ))
    man = pd.DataFrame(rows)

    # Session and day order within each participant (chronological)
    man = man.sort_values(["participant_id", "session_date", "session_time", "set_order_in_session"])
    sess = man.drop_duplicates("session_id")[["participant_id", "session_id", "session_date", "session_time"]]
    sess["session_order"] = sess.groupby("participant_id").cumcount() + 1
    sess["day_order"] = sess.groupby("participant_id").session_date.rank(method="dense").astype(int)
    man = man.merge(sess[["session_id", "session_order", "day_order"]], on="session_id")

    man["exclude_reason"] = ""
    man.loc[man.rep_count <= 1, "exclude_reason"] = "rep_count<=1 (annotation issue per authors' loader)"
    man.loc[man.error != "", "exclude_reason"] = man.error
    man["include"] = man.exclude_reason == ""

    OUT.mkdir(parents=True, exist_ok=True)
    man.to_csv(OUT / "manifest.csv", index=False)
    print(f"{len(man)} bouts, {man.include.sum()} included -> {OUT / 'manifest.csv'}")


if __name__ == "__main__":
    main()
