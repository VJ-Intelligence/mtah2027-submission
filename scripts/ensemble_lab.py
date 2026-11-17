"""Ensemble weights on saved LOMO probabilities (external DFL matches only), decoder fixed 8/24.

    python scripts/ensemble_lab.py runs/a.npz runs/b.npz [runs/c.npz]

Tries every weight vector on a 0.25 grid (sum 1) and reports mean spellF1 and folds won vs the
equal-weight average.
"""
import itertools
import sys
from multiprocessing import Pool

import numpy as np

from mtah import model
from mtah.bench import period_runs
from mtah.decode import decode
from mtah.external import idsse
from mtah.scoring import score

FILES = sys.argv[1:]
MIDS = idsse.MATCH_IDS


def _gt(mid):
    m = idsse.load(mid)
    return model.gt_labels(mid, m), m["period"]


def _eval(args):
    w, mid, Ps, gt, per = args
    P = sum(wi * p for wi, p in zip(w, Ps))
    return score(gt, decode(P, per, 8, 24), period_runs(per))["spell_f1"]


if __name__ == "__main__":
    Z = [np.load(f) for f in FILES]
    data = {mid: _gt(mid) for mid in MIDS}
    k = len(FILES)
    steps = np.arange(0, 1.0001, 0.25)
    W = [w for w in itertools.product(steps, repeat=k) if abs(sum(w) - 1) < 1e-6]
    eq = tuple([1 / k] * k)
    W = [eq] + [w for w in W if w != eq]
    jobs = [(w, mid, [z[mid][:, :3] for z in Z], *data[mid]) for w in W for mid in MIDS]
    with Pool(8) as pool:
        res = pool.map(_eval, jobs, chunksize=2)
    table = {}
    for (w, mid, *_), f in zip(jobs, res):
        table.setdefault(w, {})[mid] = f
    ref = table[eq]
    print("files:", FILES)
    for w, d in sorted(table.items(), key=lambda kv: -np.mean(list(kv[1].values()))):
        f = np.mean(list(d.values()))
        won = sum(d[m] > ref[m] for m in MIDS)
        print(" ".join(f"{x:.2f}" for x in w), f"  spellF1 {f:.4f}  folds>equal {won}/7")
