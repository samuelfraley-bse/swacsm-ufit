"""Repeat loop for the new-team design, shared by all models.

Within a repeat (seed r):
- 6 pool athletes are held out as "new" (never recorded). Test set: their later-day bouts, fixed for all N and k.
- The remaining candidates are shuffled once; the team of size N is the first N (nested teams).
- For each athlete x exercise, day-1 sets are shuffled once; k sets = the first k (nested k).
- Every model sees exactly the same training bouts for a given (repeat, N, k).
"""
import numpy as np
import pandas as pd

from .metrics import per_athlete, vote_bouts
from .team import assert_no_leak, load_manifest, load_pool, test_bouts

N_NEW = 6
TEAM_SIZES = [2, 4, 6, 8, 12, 16, 17]
KS = [1, 2]


def repeat_plan(r, man=None, pool=None):
    """Return dict with new athletes, ordered team candidates, and nested day-1 set order per athlete x class."""
    man = load_manifest() if man is None else man
    pool = load_pool() if pool is None else pool
    rng = np.random.default_rng([r, 20261023])
    a = np.array(pool)
    rng.shuffle(a)
    new, order = sorted(a[:N_NEW]), list(a[N_NEW:])
    d1 = man[(man.day_order == 1) & man.participant_id.isin(order)]
    sets = {}
    for (pid, c), g in d1.groupby(["participant_id", "exercise_class"], sort=True):
        ids = g.segment_id.to_numpy().copy()
        rng.shuffle(ids)
        sets[(pid, c)] = list(ids)
    return dict(new=new, order=order, sets=sets)


def train_ids(plan, n_team, k, classes=range(15)):
    team = plan["order"][:n_team]
    ids = []
    for pid in team:
        for c in classes:
            s = plan["sets"][(pid, c)]
            if len(s) < k:
                raise ValueError(f"{pid} class {c}: {len(s)} day-1 sets < k={k}")
            ids += s[:k]
    return team, ids


def train_ids_extra(plan, n_team, extra_classes, classes=range(15)):
    """First n_team athletes: 1 set of every class, plus a 2nd set of `extra_classes` (nested with k=1/k=2)."""
    team = plan["order"][:n_team]
    ids = []
    for pid in team:
        for c in classes:
            s = plan["sets"][(pid, c)]
            ids += s[:2] if c in extra_classes else s[:1]
    return team, ids


def run_repeat(r, fit_predict, X, F, meta, model_name, team_sizes=TEAM_SIZES, ks=KS):
    """fit_predict(train_mask, test_mask, seed) -> (n_test_windows, 15) probabilities.

    Returns (per-athlete metrics DataFrame, per-bout predictions DataFrame).
    """
    man = load_manifest()
    plan = repeat_plan(r, man)
    seg = meta.segment_id.to_numpy()
    te_new = test_bouts(man, plan["new"])
    rows, bouts_all = [], []
    for n_team in team_sizes:
        if n_team > len(plan["order"]):
            print(f"skip N={n_team}: only {len(plan['order'])} candidates")
            continue
        for k in ks:
            team, tr = train_ids(plan, n_team, k)
            te_rec = test_bouts(man, team)
            assert_no_leak(man, tr, te_rec + te_new, plan["new"])
            trm = np.isin(seg, tr)
            tem = np.isin(seg, te_rec + te_new)
            proba = fit_predict(trm, tem, seed=r)
            bouts = vote_bouts(proba, meta[tem])
            bouts["group"] = np.where(bouts.athlete.isin(plan["new"]), "new", "recorded")
            info = dict(model=model_name, repeat=r, n_team=n_team, k=k,
                        n_train_bouts=len(tr), n_train_windows=int(trm.sum()))
            for g, bg in bouts.groupby("group"):
                rows.append(per_athlete(bg).assign(group=g, **info))
            bouts_all.append(bouts.assign(**info))
    return pd.concat(rows, ignore_index=True), pd.concat(bouts_all, ignore_index=True)
