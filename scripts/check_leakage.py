"""Leakage checks beyond the per-step asserts.

1. New athletes never appear in any repeat's team order (30 repeats).
2. No raw workout CSV is duplicated across participants (byte-identical files).
3. No resampled window is duplicated across participants (identical signal).
4. No two participant IDs share identical demographics (possible same person under two IDs).

    python scripts/check_leakage.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data import get_windows  # noqa: E402
from src.experiment import repeat_plan  # noqa: E402
from src.team import load_manifest  # noqa: E402

man = load_manifest()

# 1
bad = [r for r in range(30) if set(repeat_plan(r, man)["new"]) & set(repeat_plan(r, man)["order"])]
print("1. repeats where a new athlete is in the team order:", bad)

# 2
h = man.assign(md5=[hashlib.md5(Path(p).read_bytes()).hexdigest() for p in man.file_path])
dup = h.groupby("md5").participant_id.nunique()
print("2. raw set files shared by >1 participant:", int((dup > 1).sum()), "| duplicate files overall:", int(h.md5.duplicated().sum()))

# 3
X, F, meta = get_windows()
wh = pd.Series([hashlib.md5(np.round(x, 4).tobytes()).hexdigest() for x in X])
d = pd.DataFrame(dict(h=wh, athlete=meta.athlete.to_numpy())).groupby("h").athlete.nunique()
print("3. windows identical across >1 participant:", int((d > 1).sum()), "of", len(d))

# 4
demo = man.drop_duplicates("participant_id")[["participant_id", "gender", "birth_year", "weight_kg", "height_cm", "experience_months"]]
g = demo.groupby(["gender", "birth_year", "weight_kg", "height_cm"]).participant_id.apply(list)
print("4. participant IDs sharing gender+birth year+weight+height:", [v for v in g if len(v) > 1])
