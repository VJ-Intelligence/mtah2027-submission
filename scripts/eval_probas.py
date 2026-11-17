"""Evaluate saved LOMO probability files (and their average) over a decoder grid.
    python scripts/eval_probas.py runs/a.npz [runs/b.npz ...]"""
import itertools
import sys

import numpy as np

from mtah import model
from mtah.bench import period_runs
from mtah.decode import decode
from mtah.external import idsse
from mtah.scoring import score

files = sys.argv[1:]
Ps = [np.load(f) for f in files]
mids = idsse.MATCH_IDS
data = {mid: model.sample(mid, 100 * i + 99) for i, mid in enumerate(mids)}
variants = {f: [P] for f, P in zip(files, Ps)}
if len(Ps) > 1:
    variants["AVG"] = Ps
for name, group in variants.items():
    best = None
    for pt, pdd in itertools.product((1, 2, 4, 8), (16, 24, 40)):
        f = []
        for mid in mids:
            _, y, _, per = data[mid]
            P = np.mean([g[mid][:, :3] for g in group], axis=0)
            f.append(score(np.array(list("DHA"))[y], decode(P, per, pt, pdd), period_runs(per))["spell_f1"])
        if best is None or np.mean(f) > best[0]:
            best = (np.mean(f), np.std(f), pt, pdd)
    print(f"{name:32s} spellF1 {best[0]:.4f} (sd {best[1]:.4f}) pen_team {best[2]} pen_dead {best[3]}", flush=True)
