"""Decoder search on saved LOMO probabilities (external DFL matches only).

    python scripts/decoder_lab.py runs/a.npz [runs/b.npz ...]   (several files are averaged)

Labels are rebuilt from source (no feature cache). Reports mean spellF1 over the 7 folds and
how many folds beat the reference decoder (Viterbi 8/24, no minimum lengths).
"""
import itertools
import sys
from multiprocessing import Pool

import numpy as np

from mtah import model
from mtah.bench import period_runs
from mtah.decode import decode_md
from mtah.external import idsse
from mtah.scoring import score

FILES = [a for a in sys.argv[1:] if a.endswith(".npz")]
MIDS = idsse.MATCH_IDS
REF = (8, 24, 24, 1, 1)
GRID = [REF] + [c for c in itertools.product((2, 4, 6, 8, 12), (16, 24, 32, 40, 56), (16, 24, 32, 40, 56), (1,), (1,))
                if c != REF]


def _gt(mid):
    m = idsse.load(mid)
    return model.gt_labels(mid, m), m["period"]


def _eval(args):
    mid, P, gt, per, cfg = args
    pt, pe, px, md, mt = cfg
    return score(gt, decode_md(P, per, pt, pe, px, md, mt), period_runs(per))["spell_f1"]


if __name__ == "__main__":
    Ps = [np.load(f) for f in FILES]
    data = {mid: _gt(mid) for mid in MIDS}
    probs = {mid: np.mean([P[mid][:, :3] for P in Ps], axis=0) for mid in MIDS}
    jobs = [(mid, probs[mid], data[mid][0], data[mid][1], cfg) for cfg in GRID for mid in MIDS]
    with Pool(8) as pool:
        res = pool.map(_eval, jobs, chunksize=4)
    table = {}
    for (mid, _, _, _, cfg), f in zip(jobs, res):
        table.setdefault(cfg, {})[mid] = f
    ref = table[REF]
    rows = []
    for cfg, d in table.items():
        f = np.array([d[m] for m in MIDS])
        better = int(sum(d[m] > ref[m] for m in MIDS))
        rows.append((f.mean(), cfg, better, f.std()))
    rows.sort(reverse=True)
    print(f"files: {FILES}")
    print(f"reference {REF}: {np.mean(list(ref.values())):.4f}")
    print("pen_team enter_d exit_d min_d min_t   spellF1   folds>ref")
    for mean, cfg, better, sd in rows[:12]:
        print(f"{cfg[0]:8d} {cfg[1]:7d} {cfg[2]:6d} {cfg[3]:5d} {cfg[4]:5d}   {mean:.4f}   {better}/7")
