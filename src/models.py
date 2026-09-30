"""Small deep models with one fixed training recipe (no tuning on test athletes, no early stopping).

Input: windows (n, 4, T) = x, y, z, |a| at 30 Hz, standardized per channel with training-window stats.
Recipe: Adam lr 1e-3, weight decay 1e-4, 30 epochs, batch 64, cross-entropy.
"""
import numpy as np
import torch
from torch import nn

N_CLASSES = 15
EPOCHS = 30
BATCH = 64
LR = 1e-3
WD = 1e-4


class CNN(nn.Module):
    """3 conv blocks (32-64-64, kernel 5, BatchNorm, ReLU, max-pool) -> global average pool -> dropout -> linear."""

    def __init__(self, in_ch=4, n_classes=N_CLASSES):
        super().__init__()
        layers, c = [], in_ch
        for out in (32, 64, 64):
            layers += [nn.Conv1d(c, out, 5, padding=2), nn.BatchNorm1d(out), nn.ReLU(), nn.MaxPool1d(2)]
            c = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(c, n_classes))

    def forward(self, x):
        return self.head(self.features(x).mean(-1))


class LSTM(nn.Module):
    """2-layer LSTM (hidden 64) over time -> mean over time steps -> dropout -> linear."""

    def __init__(self, in_ch=4, hidden=64, n_classes=N_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, n_classes))

    def forward(self, x):
        out, _ = self.lstm(x.transpose(1, 2))  # (n, T, C)
        return self.head(out.mean(1))


ARCHS = {"cnn": CNN, "lstm": LSTM}


def fit_predict(arch, Xtr, ytr, Xte, seed, epochs=EPOCHS):
    """Train a fresh model on (Xtr, ytr) and return softmax probabilities for Xte."""
    return predict(*train_model(arch, Xtr, ytr, seed, epochs), Xte)


def train_model(arch, Xtr, ytr, seed, epochs=EPOCHS):
    """Train a fresh model; returns (model, mu, sd) where mu/sd are the training normalization stats."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    xtr = torch.from_numpy(((Xtr - mu) / sd).astype(np.float32))
    ytr_t = torch.from_numpy(ytr.astype(np.int64))

    model = ARCHS[arch]()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    n = len(xtr)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss_fn(model(xtr[idx]), ytr_t[idx]).backward()
            opt.step()
    return model, mu, sd


def predict(model, mu, sd, X):
    model.eval()
    x = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    with torch.no_grad():
        return torch.cat([torch.softmax(model(x[i:i + 2048]), 1) for i in range(0, len(x), 2048)]).numpy()


# Fine-tuning recipe for the battery study (fixed in advance, same for every condition)
FT_STEPS = 150
FT_LR = 1e-4


P_ATHLETE = 0.5  # within a class the athlete recorded, share of examples drawn from the athlete


def finetune(model, mu, sd, Xa, ya, Xrep, yrep, seed, steps=FT_STEPS, lr=FT_LR):
    """Copy `model` and continue training for a fixed number of steps with class-balanced batches.

    Each batch draws its labels uniformly over the 15 classes, so the class prior never shifts toward
    the athlete's recorded movements. For a class the athlete recorded, each example comes from the
    athlete's windows (Xa) with probability P_ATHLETE, else from replayed team windows of that class.
    With Xa empty (no-battery control), every example is replay; the training budget is identical.
    Normalization stays fixed at the team's mu/sd.
    """
    import copy

    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    norm = lambda X: torch.from_numpy(((X - mu) / sd).astype(np.float32))  # noqa: E731
    xr = norm(Xrep)
    xa = norm(Xa) if len(Xa) else None
    rep_idx = [torch.from_numpy(np.flatnonzero(yrep == c)) for c in range(N_CLASSES)]
    ath_idx = [torch.from_numpy(np.flatnonzero(ya == c)) for c in range(N_CLASSES)]
    m.train()
    for _ in range(steps):
        labels = torch.randint(N_CLASSES, (BATCH,), generator=g)
        use_a = torch.rand(BATCH, generator=g) < P_ATHLETE
        xs = []
        for i in range(BATCH):
            c = int(labels[i])
            if xa is not None and len(ath_idx[c]) and use_a[i]:
                pool, src = ath_idx[c], xa
            else:
                pool, src = rep_idx[c], xr
            xs.append(src[pool[torch.randint(len(pool), (1,), generator=g)]])
        opt.zero_grad()
        loss_fn(m(torch.cat(xs)), labels).backward()
        opt.step()
    return m
