"""Step 03: one random-forest run of the new-team design (seed 0), before building the repeat loop.

Holds out 6 "new" athletes, then trains on day-1 sets from a team of N recorded athletes and tests on
later days of the recorded team and of the new athletes. Runs N in {2, max} x k in {1, 2} to see both
ends of the curve. Bout label = majority vote of its 2 s windows.

    python steps/03_rf_single_run.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os  # noqa: E402

os.chdir(ROOT)
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from src import config as C  # noqa: E402
from src.data import get_windows  # noqa: E402
from src.metrics import bootstrap_ci, per_athlete, vote_bouts  # noqa: E402
from src.team import assert_no_leak, load_manifest, load_pool, split_new, test_bouts, train_bouts  # noqa: E402
from src.utils import git_hash  # noqa: E402

SEED = 0
N_NEW = 6
OUT = ROOT / "outputs" / "tables"


def main():
    X, F, meta = get_windows()
    man = load_manifest()
    pool = load_pool()
    rng = np.random.default_rng(SEED)
    candidates, new = split_new(pool, N_NEW, rng)
    print(f"pool {len(pool)}, team candidates {len(candidates)}, new athletes {new}")

    all_bouts, summary = [], []
    for n_team in (2, len(candidates)):
        team = sorted(np.random.default_rng([SEED, n_team]).choice(candidates, n_team, replace=False))
        te_rec, te_new = test_bouts(man, team), test_bouts(man, new)
        for k in (1, 2):
            tr = train_bouts(man, team, k, np.random.default_rng([SEED, n_team, k]))
            assert_no_leak(man, tr, te_rec + te_new, new)
            trm = meta.segment_id.isin(tr).to_numpy()
            rf = RandomForestClassifier(C.RF_TREES, class_weight="balanced", n_jobs=-1, random_state=SEED)
            rf.fit(F[trm], meta.cls.to_numpy()[trm])
            assert list(rf.classes_) == list(range(15))
            for group, ids in (("recorded", te_rec), ("new", te_new)):
                tm = meta.segment_id.isin(ids).to_numpy()
                bouts = vote_bouts(rf.predict_proba(F[tm]), meta[tm])
                bouts = bouts.assign(model="rf", n_team=n_team, k=k, group=group, seed=SEED)
                all_bouts.append(bouts)
                pa = per_athlete(bouts)
                m, lo, hi = bootstrap_ci(pa.macro_f1)
                win_acc = (rf.predict(F[tm]) == meta.cls.to_numpy()[tm]).mean()
                summary.append(dict(n_team=n_team, k=k, group=group, n_train_bouts=len(tr), n_train_windows=int(trm.sum()),
                                    n_test_athletes=len(pa), n_test_bouts=len(bouts), macro_f1=m, ci_lo=lo, ci_hi=hi,
                                    min_athlete_f1=pa.macro_f1.min(), window_acc=win_acc))
    s = pd.DataFrame(summary).assign(seed=SEED, git=git_hash())
    b = pd.concat(all_bouts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "03_rf_single_run_summary.csv", index=False)
    b.to_csv(OUT / "03_rf_single_run_bouts.csv", index=False)
    print(s.drop(columns=["seed", "git"]).round(3).to_string(index=False))

    big = b[(b.n_team == len(candidates)) & (b.k == 2)]
    wrong = big[big.true != big.pred]
    print(f"\nN={len(candidates)}, k=2: {len(wrong)} of {len(big)} test bouts wrong. Most common errors (true -> pred):")
    print(wrong.groupby(["true", "pred"]).size().sort_values(ascending=False).head(8)
          .rename(lambda i: i).to_frame("n").assign(
              pair=lambda d: [f"{C.CLASS_NAMES[t]} -> {C.CLASS_NAMES[p]}" for t, p in d.index]).to_string())


if __name__ == "__main__":
    main()
