"""Compare the reference baseline on degraded external data with the PUBLISHED full-match
baseline scores of the official repo (README/EVALUATION: raw 0.0486, smooth25 0.1996,
smooth50 0.2433, smooth125 0.2242). Published numbers only, no match data involved."""
import numpy as np

from mtah import baselines
from mtah.bench import period_runs
from mtah.degrade import degrade, describe
from mtah.external import idsse
from mtah.labels import controlled
from mtah.scoring import score

PUBLISHED = {0: 0.0486, 25: 0.1996, 50: 0.2433, 125: 0.2242}

res = {w: [] for w in PUBLISHED}
stats = []
for i, mid in enumerate(idsse.MATCH_IDS):
    m = idsse.load(mid)
    gt = controlled(m["label"], m["period"], idsse.action_frames(mid, m["period"]), 1.0)
    for seed in range(2):
        d = degrade(m, seed=1000 * i + seed)
        stats.append(describe(d))
        for w in PUBLISHED:
            res[w].append(score(gt, baselines.predict(d, w), period_runs(m["period"]))["spell_f1"])
print("window  published  ours(mean, sd)")
for w, v in PUBLISHED.items():
    print(f"{w:6d}  {v:.4f}     {np.mean(res[w]):.4f} {np.std(res[w]):.4f}")
for k in stats[0]:
    print(f"{k:16s} {np.mean([s[k] for s in stats]):.3f}")
