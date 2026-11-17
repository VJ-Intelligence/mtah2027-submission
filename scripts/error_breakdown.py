"""Where do we lose spellF1? Confusion + spell recall by length, on saved LOMO probabilities."""
import sys

import numpy as np

from mtah import model
from mtah.bench import period_runs
from mtah.decode import decode
from mtah.external import idsse
from mtah.scoring import pm, codes

P = np.load(sys.argv[1] if len(sys.argv) > 1 else "runs/lomo_probas.npz")
pt, pdd = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (4, 40)
cm = np.zeros((3, 3), int)
buckets = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 20), (20, 999)]
hit = {b: [0, 0] for b in buckets}
for i, mid in enumerate(idsse.MATCH_IDS):
    _, y, _, per = model.sample(mid, 100 * i + 99)
    gt = np.array(["D", "H", "A"])[y]
    pr = decode(P[mid], per, pt, pdd)
    for a, la in enumerate("DHA"):
        for b, lb in enumerate("DHA"):
            cm[a, b] += ((gt == la) & (pr == lb)).sum()
    runs = period_runs(per)
    g, p = codes(gt), codes(pr)
    gs, ps = pm.possession_spells(g, runs), pm.possession_spells(p, runs)
    ov = pm.spell_overlaps(gs, ps)
    best = {}
    for iou, gi, pi in ov:
        best[gi] = max(best.get(gi, 0), iou)
    for gi, sp in enumerate(gs):
        L = (sp[1] - sp[0]) / 25 if len(sp) >= 2 else 0
        for b in buckets:
            if b[0] <= L < b[1]:
                hit[b][0] += 1
                hit[b][1] += best.get(gi, 0) >= 0.5
print("confusion rows=GT cols=pred (D,H,A), row-normalised")
print(np.round(cm / cm.sum(1, keepdims=True), 3))
print("GT spells by length: count, share matched at IoU>=0.5")
for b in buckets:
    n, h = hit[b]
    print(f"{b[0]:>3}-{b[1]:<4}s  n={n:5d}  matched {h / max(n, 1):.2f}")
