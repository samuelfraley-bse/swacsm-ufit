"""Cross-check the authors' dataset_loader against scripts/audit.py output (plan section 1, item 5)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "data/uLift-dataset/dataset_loader")
from dataset_loader import DataSetLoader  # noqa: E402

if __name__ == "__main__":
    ld = DataSetLoader("data/uLift-dataset/uLift_sensor_dataset", multiprocess=False)
    rows = []
    for s in ld.datasets:
        for w in s.workout_segments:
            rows.append(dict(file=w.file_name, cls=w.workout_class_number, reps=w.repetition_number,
                             n_loader=len(w.raw_data), rate_loader=w.sampling_rate))
    L = pd.DataFrame(rows)
    A = pd.read_csv("results/audit_segments.csv")
    A = A[A.reps > 1]
    print("loader: subjects", len(ld.unique_names), "sessions", len(ld.datasets), "workout segs", len(L), "reps", L.reps.sum())
    print("mine (reps>1): segs", len(A), "reps", A.reps.sum())
    m = A.merge(L, on="file", how="outer", indicator=True)
    print(m["_merge"].value_counts().to_dict())
    both = m[m._merge == "both"]
    print("class agree:", (both.cls == both.cls_file).all(), " reps agree:", (both.reps_x == both.reps_y).all())
    print("loader drops duplicate timestamps -> kept fraction of samples:",
          (both.n_loader / both.n).describe().round(3).to_dict())
    print("loader 'sampling_rate' (after dedup), Hz:", both.rate_loader.describe().round(1).to_dict())
