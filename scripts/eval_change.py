"""Sweep the change-evidence decoder on saved 4-column TCN probabilities (optionally averaged
with other 3-column files for the state part)."""
import itertools
import sys

import numpy as np

from mtah import model
from mtah.bench import period_runs
from mtah.decode import decode
from mtah.external import idsse
from mtah.scoring import score

main = np.load(sys.argv[1])
others = [np.load(f) for f in sys.argv[2:]]
mids = idsse.MATCH_IDS
data = {mid: model.sample(mid, 100 * i + 99) for i, mid in enumerate(mids)}
res = []
for pt, pdd, al in itertools.product((4, 8, 12, 16), (16, 24, 40), (0, 0.5, 1, 2, 3)):
    f = []
    for mid in mids:
        _, y, _, per = data[mid]
        P = main[mid]
        states = np.mean([P[:, :3]] + [o[mid][:, :3] for o in others], axis=0)
        lab = decode(states, per, pt, pdd, change=P[:, 3], alpha=al)
        f.append(score(np.array(list("DHA"))[y], lab, period_runs(per))["spell_f1"])
    res.append((np.mean(f), np.std(f), pt, pdd, al))
for r in sorted(res, reverse=True)[:6]:
    print("spellF1 %.4f (sd %.4f) pen_team %s pen_dead %s alpha %s" % r)
print("best alpha=0:", max(r for r in res if r[4] == 0))
