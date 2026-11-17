"""Leave-one-match-out on external matches: train LightGBM on the others, decode, official score.

    PYTHONPATH=src .venv/bin/python scripts/lomo.py
"""
import os
import itertools
import sys

import numpy as np

from mtah import model
from mtah.bench import period_runs
from mtah.decode import decode
from mtah.external import idsse
from mtah.scoring import score

TRAIN_SEEDS = (0, 1, 2, 3)  # same as submit.py
TEST_SEED = 99
PENS = [(t, d) for t, d in itertools.product((4, 8, 12, 16, 24), (8, 16, 24))]

USE_SC = "--sc" in sys.argv
OUT = os.environ.get("OUT") or ("runs/lomo_probas_sc.npz" if USE_SC else "runs/lomo_probas.npz")
mids = idsse.MATCH_IDS
probas, gts, periods = {}, {}, {}
for i, test in enumerate(mids):
    parts = []
    for mid in mids:
        if mid == test:
            continue
        for s in TRAIN_SEEDS:
            X, y, names, _ = model.sample(mid, 100 * mids.index(mid) + s)
            parts.append((X, y, names))
    if USE_SC:
        from mtah.external import skillcorner
        for j, sid in enumerate(skillcorner.MATCH_IDS):
            X, y, names, _ = model.sample_sc(sid, 5000 + j)
            parts.append((X, y, names))
    booster = model.train(parts)
    X, y, names, per = model.sample(test, 100 * i + TEST_SEED)
    probas[test] = model.predict_proba(booster, X, names)
    gts[test] = np.array(["D", "H", "A"])[y]
    periods[test] = per
    acc = (probas[test].argmax(1) == y).mean()
    print(f"{test}: frame acc {acc:.3f}", flush=True)

np.savez_compressed(OUT, **{k: v for k, v in probas.items()})
print("\npen_team pen_dead  spellF1 mean (sd)   pred/gt spells")
best = None
for pt, pd_ in PENS:
    f, ratio = [], []
    for mid in mids:
        lab = decode(probas[mid], periods[mid], pt, pd_)
        s = score(gts[mid], lab, period_runs(periods[mid]))
        f.append(s["spell_f1"])
        ratio.append(s["n_pred_spells"] / s["n_gt_spells"])
    m = np.mean(f)
    print(f"{pt:8.0f} {pd_:8.0f}  {m:.4f} ({np.std(f):.4f})   {np.mean(ratio):.2f}")
    if best is None or m > best[0]:
        best = (m, pt, pd_)
print("BEST", best)
