"""Battery study: does recording a new athlete on a few movements help recognize their OTHER exercises?

For each repeat (same new athletes and team order as steps 04-05) and team size N (team uses k=2):
- none:    model trained on the team only (shared by all new athletes in the repeat)
- battery: team + this new athlete's day-1 sets of the battery movements only
- all15:   team + this new athlete's day-1 sets of all 15 exercises (= "recorded" reference)
Scoring uses only the athlete's later-day bouts of exercises NOT in the battery, so every condition
is compared on the same bouts. Predictions into battery classes count as errors.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .experiment import repeat_plan, train_ids
from .metrics import vote_bouts
from .team import load_manifest, test_bouts

K_TEAM = 2
K_BATTERY = 2  # sets per battery movement (day 1 holds two circuits)
NAMED = {"squat_deadlift": (0, 6), "pushup_pushpress": (1, 7)}
N_RANDOM = 2   # random 2-movement batteries per new athlete per repeat


def batteries_for(athlete_idx, r):
    rng = np.random.default_rng([r, 777, athlete_idx])
    pairs = [p for p in itertools.combinations(range(15), 2) if p not in NAMED.values()]
    rand = [pairs[i] for i in rng.choice(len(pairs), N_RANDOM, replace=False)]
    return dict(NAMED, **{f"random{j + 1}": p for j, p in enumerate(rand)})


def new_athlete_sets(man, athlete, r):
    """Day-1 set ids per class for a new athlete, shuffled with a fixed seed (first K are used)."""
    rng = np.random.default_rng([r, 555, int(athlete.removeprefix("Subject"))])
    d1 = man[(man.day_order == 1) & (man.participant_id == athlete)]
    out = {}
    for c, g in d1.groupby("exercise_class"):
        ids = g.segment_id.to_numpy().copy()
        rng.shuffle(ids)
        out[c] = list(ids[:K_BATTERY])
    return out


class RetrainLearner:
    """Random forest: 'none' = team-only model; other conditions retrain from scratch on team + extra."""

    def __init__(self, F, y, trees):
        self.F, self.y, self.trees = F, y, trees

    def _fit(self, mask, seed):
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(self.trees, class_weight="balanced", n_jobs=1, random_state=seed)
        return rf.fit(self.F[mask], self.y[mask])

    def base(self, team_mask, seed):
        self.team_mask = team_mask
        return self._fit(team_mask, seed)

    def cond(self, state, extra_mask, test_mask, seed):
        m = state if extra_mask is None else self._fit(self.team_mask | extra_mask, seed)
        return m.predict_proba(self.F[test_mask])


class FinetuneLearner:
    """CNN/LSTM: team model trained once; every condition (incl. 'none') gets the same class-balanced
    fine-tuning budget. 'none' fine-tunes on replayed team windows only, so gains come from the
    athlete's data, not from extra training or a shifted class prior."""

    def __init__(self, arch, X, y):
        self.arch, self.X, self.y = arch, X, y

    def base(self, team_mask, seed):
        from .models import train_model
        self.team_mask = team_mask
        return train_model(self.arch, self.X[team_mask], self.y[team_mask], seed)

    def cond(self, state, extra_mask, test_mask, seed):
        from .models import finetune, predict
        model, mu, sd = state
        empty = self.X[:0]
        Xa, ya = (empty, self.y[:0]) if extra_mask is None else (self.X[extra_mask], self.y[extra_mask])
        m = finetune(model, mu, sd, Xa, ya, self.X[self.team_mask], self.y[self.team_mask], seed)
        return predict(m, mu, sd, self.X[test_mask])


def run_battery_repeat(r, learner, meta, model_name, team_sizes):
    man = load_manifest()
    plan = repeat_plan(r, man)
    seg = meta.segment_id.to_numpy()
    preds = []
    for n_team in team_sizes:
        team, tr_team = train_ids(plan, n_team, K_TEAM)
        state = learner.base(np.isin(seg, tr_team), seed=r)
        for ai, a in enumerate(plan["new"]):
            te_ids = test_bouts(man, [a])
            tem = np.isin(seg, te_ids)
            ft_seed = r * 1000 + ai  # same seed for every condition of this athlete
            b = vote_bouts(learner.cond(state, None, tem, ft_seed), meta[tem])
            preds.append(b.assign(condition="none", battery="", n_team=n_team))
            sets = new_athlete_sets(man, a, r)
            conds = {"all15": tuple(range(15)), **batteries_for(ai, r)}
            for name, classes in conds.items():
                extra = [s for c in classes for s in sets[c]]
                assert not set(extra) & set(te_ids), "battery bout in test set"
                assert not set(extra) & set(tr_team), "battery bout already in team data"
                b = vote_bouts(learner.cond(state, np.isin(seg, extra), tem, ft_seed), meta[tem])
                preds.append(b.assign(condition="all15" if name == "all15" else "battery", battery=name,
                                      battery_classes=",".join(map(str, classes)), n_team=n_team))
    out = pd.concat(preds, ignore_index=True)
    return out.assign(model=model_name, repeat=r)


def score(preds):
    """One row per (model, repeat, n_team, athlete, battery): none vs battery vs all15 on non-battery bouts."""
    rows = []
    key = ["model", "repeat", "n_team", "athlete"]
    for k, g in preds.groupby(key):
        none = g[g.condition == "none"].set_index("segment_id")
        full = g[g.condition == "all15"].set_index("segment_id")
        for bname, gb in g[g.condition == "battery"].groupby("battery"):
            classes = [int(c) for c in gb.battery_classes.iloc[0].split(",")]
            ev = gb[~gb.true.isin(classes)].set_index("segment_id")
            labels = sorted(set(range(15)) - set(classes))
            row = dict(zip(key, k), battery=bname, battery_classes=gb.battery_classes.iloc[0], n_eval_bouts=len(ev))
            for cname, df in (("none", none.loc[ev.index]), ("battery", ev), ("all15", full.loc[ev.index])):
                row[f"f1_{cname}"] = f1_score(df.true, df.pred, labels=labels, average="macro", zero_division=0)
                row[f"acc_{cname}"] = accuracy_score(df.true, df.pred)
                row[f"pull_{cname}"] = float(df.pred.isin(classes).mean())  # non-battery bouts called a battery move
            # performance on the battery movements themselves (bouts of those classes)
            on = gb[gb.true.isin(classes)].set_index("segment_id")
            row["acc_on_battery_none"] = accuracy_score(on.true, none.loc[on.index].pred) if len(on) else np.nan
            row["acc_on_battery_battery"] = accuracy_score(on.true, on.pred) if len(on) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
