"""Post-decoding experiments on saved LOMO probabilities (external DFL matches only).

    python scripts/refine_lab.py runs/mix.npz

1. Short-spell insertion: inside a decoded H/A spell, relabel a run where the other team is the
   argmax, of length >= L frames, with mean P(other) >= q, if the host spell is >= 4x the run.
2. Shift probe: labels rolled by k frames per half (a systematic lag would show up here).
Reports mean spellF1 and folds won vs the plain Viterbi 8/24 decode.
"""
import sys
from multiprocessing import Pool

import numpy as np

from mtah import model
from mtah.bench import period_runs
from mtah.decode import decode
from mtah.external import idsse
from mtah.scoring import score

MIDS = idsse.MATCH_IDS


def refine(labels, P, q, L):
    out = labels.copy()
    other = {"H": 2, "A": 1}
    arg = P.argmax(1)
    ch = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    edges = np.r_[0, ch, len(labels)]
    for a, b in zip(edges[:-1], edges[1:]):
        lab = labels[a]
        if lab not in other:
            continue
        o = other[lab]
        hit = np.r_[False, arg[a:b] == o, False]
        d = np.flatnonzero(hit[1:] != hit[:-1])
        for s, e in zip(d[::2], d[1::2]):
            n = e - s
            if n >= L and (b - a) >= 4 * n and P[a + s:a + e, o].mean() >= q:
                out[a + s:a + e] = "A" if lab == "H" else "H"
    return out


def shift(labels, per, k):
    out = labels.copy()
    for p in np.unique(per):
        if p == 0:
            continue
        idx = np.flatnonzero(per == p)
        out[idx] = np.roll(labels[idx], k)
        if k > 0:
            out[idx[:k]] = labels[idx[0]]
        elif k < 0:
            out[idx[k:]] = labels[idx[-1]]
    return out


def _eval(args):
    mid, P, gt, per, cfg = args
    lab = decode(P, per, 8, 24)
    kind, a, b = cfg
    if kind == "refine":
        lab = refine(lab, P, a, b)
    elif kind == "shift":
        lab = shift(lab, per, a)
    return score(gt, lab, period_runs(per))["spell_f1"]


def _gt(mid):
    m = idsse.load(mid)
    return model.gt_labels(mid, m), m["period"]


if __name__ == "__main__":
    Z = np.load(sys.argv[1])
    data = {mid: _gt(mid) for mid in MIDS}
    cfgs = [("ref", 0, 0)] + [("refine", q, L) for q in (0.45, 0.55, 0.65) for L in (8, 12, 20)] + \
           [("shift", k, 0) for k in (-8, -4, -2, 2, 4, 8)]
    jobs = [(mid, Z[mid][:, :3], *data[mid], c) for c in cfgs for mid in MIDS]
    with Pool(8) as pool:
        res = pool.map(_eval, jobs, chunksize=2)
    table = {}
    for (mid, *_, c), f in zip(jobs, res):
        table.setdefault(c, {})[mid] = f
    ref = table[("ref", 0, 0)]
    for c, d in table.items():
        print(c, f"spellF1 {np.mean(list(d.values())):.4f}  folds>ref {sum(d[m] > ref[m] for m in MIDS)}/7")
