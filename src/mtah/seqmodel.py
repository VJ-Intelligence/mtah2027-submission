"""Sequence model: dilated TCN over identity-free per-frame features -> P(D,H,A) per frame.

Trained on degraded external matches only. Team-swap augmentation, boundary-weighted loss
(frames near a ground-truth transition count more, since short spells decide spellF1).
"""
import os
import numpy as np
import torch
from torch import nn

from . import features

DEVICE = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")


class Block(nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.c2 = nn.Conv1d(ch, ch, 1)
        self.norm = nn.GroupNorm(8, ch)

    def forward(self, x):
        return x + self.c2(torch.nn.functional.gelu(self.norm(self.c1(x))))


class Net(nn.Module):
    """Non-causal dilated TCN (Borghesi et al. 2023 style). Receptive field ~ +-512 frames."""

    def __init__(self, n_feat: int, hidden: int = 96, dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 1, 4, 16)):
        super().__init__()
        self.inp = nn.Conv1d(n_feat, hidden, 1)
        self.blocks = nn.Sequential(*[Block(hidden, d) for d in dilations])
        self.head = nn.Conv1d(hidden, 4, 1)  # 3 state logits + 1 change logit
        fdrop = float(os.environ.get("FDROP", "0"))  # input channel dropout (experiment, off by default)
        self.drop = nn.Dropout1d(fdrop) if fdrop > 0 else None

    def forward(self, x):  # x (B, T, F) -> (B, 4, T) channel-first (MPS-safe for the losses)
        x = x.transpose(1, 2).contiguous()
        if self.drop is not None:
            x = self.drop(x)
        h = self.blocks(self.inp(x))
        return self.head(h)


class Normaliser:
    def __init__(self, X):
        Xn = np.where(np.isfinite(X), X, np.nan)
        self.mu = np.nan_to_num(np.nanmean(Xn, 0))
        self.sd = np.nan_to_num(np.nanstd(Xn, 0)) + 1e-3

    def __call__(self, X):
        Z = (np.where(np.isfinite(X), X, np.nan) - self.mu) / self.sd
        return np.clip(np.nan_to_num(Z, nan=0.0), -6, 6).astype(np.float32)


def _swap_y(y):
    s = y.copy()
    s[y == 1], s[y == 2] = 2, 1
    return s


import os

def _weights(y, radius=int(os.environ.get("BW_RADIUS", "12")), boost=float(os.environ.get("BW_BOOST", "3.0"))):
    w = np.ones(len(y), np.float32)
    t = np.flatnonzero(y[1:] != y[:-1]) + 1
    for k in t:
        w[max(0, k - radius):k + radius] = boost
    return w


SPELL_REF = int(os.environ.get("SPELL_REF", "0"))  # frames; 0 = off (experiment)


def _spell_weights(y):
    """Boundary weights plus extra weight on short possession spells (spellF1 counts every spell
    equally, per-frame CE weights a spell by its length). Computed on a whole half."""
    w = _weights(y)
    t = np.flatnonzero(y[1:] != y[:-1]) + 1
    edges = np.r_[0, t, len(y)]
    for a, b in zip(edges[:-1], edges[1:]):
        if y[a] > 0 and b - a < SPELL_REF:
            w[a:b] *= min(4.0, (SPELL_REF / (b - a)) ** 0.5)
    return w


def _change(y, radius=3):
    """1 near a team-to-team possession change (H<->A), else 0."""
    c = np.zeros(len(y), np.float32)
    t = np.flatnonzero((y[1:] != y[:-1]) & (y[1:] > 0) & (y[:-1] > 0)) + 1
    for k in t:
        c[max(0, k - radius):k + radius + 1] = 1
    return c


def _chunks(n, length, stride):
    out = [(s, min(n, s + length)) for s in range(0, max(1, n - length // 2), stride)]
    if n >= length and not any(b == n and b - a == length for a, b in out):
        out.append((n - length, n))  # the end of every half (end-of-half stoppages) is trained too
    return out


def train(parts, epochs=8, length=1500, stride=500, batch=16, lr=2e-3, seed=7, log=print, hidden=96):
    """parts: list of (X, y, names, period)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    names = parts[0][2]
    norm = Normaliser(np.concatenate([p[0][::10] for p in parts]))
    seqs = []
    for X, y, nm, per in parts:
        for xx, yy in ((X, y), (features.swap_teams(X, nm), _swap_y(y))):
            Z = norm(xx)
            for p in np.unique(per):
                idx = np.flatnonzero(per == p)
                wh = _spell_weights(yy[idx]) if SPELL_REF else None
                for a, b in _chunks(len(idx), length, stride):
                    seg = idx[a:b]
                    if len(seg) == length:
                        ww = wh[a:b] if SPELL_REF else _weights(yy[seg])
                        seqs.append((Z[seg], yy[seg], ww, _change(yy[seg])))
    net = Net(len(names), hidden=hidden).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=float(os.environ.get("WD", "1e-4")))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(seqs) // batch + 1))
    ls = float(os.environ.get("LS", "0"))
    lossf = nn.CrossEntropyLoss(reduction="none", label_smoothing=ls) if ls else nn.CrossEntropyLoss(reduction="none")
    swa = int(os.environ.get("SWA", "0"))  # average the weights of the last `swa` epochs (experiment)
    avg = None
    for ep in range(epochs):
        order = rng.permutation(len(seqs))
        tot = 0.0
        net.train()
        for i in range(0, len(order), batch):
            bi = order[i:i + batch]
            x = torch.from_numpy(np.stack([seqs[j][0] for j in bi])).to(DEVICE)
            y = torch.from_numpy(np.stack([seqs[j][1] for j in bi]).astype(np.int64)).to(DEVICE)
            w = torch.from_numpy(np.stack([seqs[j][2] for j in bi])).to(DEVICE)
            c = torch.from_numpy(np.stack([seqs[j][3] for j in bi])).to(DEVICE)
            out = net(x)
            loss = (lossf(out[:, :3], y) * w).sum() / w.sum()  # (B,3,T): reshape breaks MPS backward
            loss = loss + 0.5 * nn.functional.binary_cross_entropy_with_logits(
                out[:, 3], c, pos_weight=torch.tensor(8.0, device=DEVICE))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(bi)
        log(f"  epoch {ep + 1}/{epochs} loss {tot / len(seqs):.4f}  ({len(seqs)} seqs)")
        if swa and ep >= epochs - swa:
            sd = {k: v.detach().float().cpu().clone() / swa for k, v in net.state_dict().items()}
            avg = sd if avg is None else {k: avg[k] + sd[k] for k in avg}
    out = {"net": net.eval(), "norm": norm, "names": names}
    if avg is not None:
        net2 = Net(len(names), hidden=hidden).to(DEVICE)
        net2.load_state_dict(avg)
        out["swa"] = {"net": net2.eval(), "norm": norm, "names": names}
    return out


@torch.no_grad()
def predict_proba(m, X, period):
    """Symmetric inference over whole halves (model + team-swapped mirror)."""
    net, norm, names = m["net"], m["norm"], m["names"]
    out = np.zeros((len(X), 4), np.float32)
    for xx, flip in ((X, False), (features.swap_teams(X, names), True)):
        Z = norm(xx)
        for p in np.unique(period):
            idx = np.flatnonzero(period == p)
            pr = np.zeros((len(idx), 4), np.float32)
            win, pad = 20000, 1024
            for s0 in range(0, len(idx), win):
                a0, b0 = max(0, s0 - pad), min(len(idx), s0 + win + pad)
                x = torch.from_numpy(Z[idx[a0:b0]][None]).to(DEVICE)
                o = net(x)[0].cpu()  # (4, T)
                q = torch.cat([torch.softmax(o[:3], 0), torch.sigmoid(o[3:])], 0).T.numpy()
                pr[s0:min(len(idx), s0 + win)] = q[s0 - a0:s0 - a0 + min(win, len(idx) - s0)]
            out[idx] += pr[:, [0, 2, 1, 3]] if flip else pr
    return out / 2
