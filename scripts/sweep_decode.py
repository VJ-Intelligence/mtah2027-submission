"""Re-decode saved LOMO probabilities with other penalties (fast, no retraining)."""
import itertools
import sys

import numpy as np

from mtah import model
from mtah.bench import period_runs
from mtah.decode import decode
from mtah.external import idsse
from mtah.scoring import score

P = np.load(sys.argv[1] if len(sys.argv) > 1 else "runs/lomo_probas.npz")
mids = idsse.MATCH_IDS
data = {}
for i, mid in enumerate(mids):
    _, y, _, per = model.sample(mid, 100 * i + 99)
    data[mid] = (np.array(["D", "H", "A"])[y], per)
grid = list(itertools.product((0.5, 1, 2, 3, 4, 6), (12, 24, 40, 60)))
for pt, pdd in grid:
    f, r = [], []
    for mid in mids:
        gt, per = data[mid]
        s = score(gt, decode(P[mid], per, pt, pdd), period_runs(per))
        f.append(s["spell_f1"]); r.append(s["n_pred_spells"] / s["n_gt_spells"])
    print(f"{pt:5.1f} {pdd:5.0f}  {np.mean(f):.4f} ({np.std(f):.4f})  ratio {np.mean(r):.2f}", flush=True)
