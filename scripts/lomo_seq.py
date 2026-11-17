"""LOMO for the sequence model. Saves probabilities to runs/lomo_seq_probas.npz."""
import os
import sys
import time

import numpy as np

from mtah import model, seqmodel
from mtah.external import idsse

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
FOLDS = sys.argv[2].split(",") if len(sys.argv) > 2 else idsse.MATCH_IDS
SEEDS = tuple(range(int(os.environ.get("SEEDS", "2"))))
OUT = os.environ.get("OUT", "runs/lomo_seq_probas.npz")
HIDDEN = int(os.environ.get("HIDDEN", "96"))
mids = idsse.MATCH_IDS
probas = {}
probas_swa = {}
for test in FOLDS:
    i = mids.index(test)
    t = time.time()
    parts = [model.sample(mid, 100 * mids.index(mid) + s) for mid in mids if mid != test for s in SEEDS]
    if os.environ.get("MIX_SC"):
        from mtah.external import skillcorner
        parts += [model.sample_sc(sid, 5000 + j) for j, sid in enumerate(skillcorner.MATCH_IDS)]
    if os.environ.get("MIX_ST"):
        from mtah.external import soccertrack
        parts += [model.sample_st(sid, 7000 + 100 * c + j) for c in range(int(os.environ.get("ST_COPIES", "1")))
                  for j, sid in enumerate(soccertrack.TRAIN_IDS)]
    if int(os.environ.get("MIX_PFF", "0")):
        from mtah.external import pff
        pids = pff.match_ids()[:int(os.environ["MIX_PFF"])]
        parts += [model.sample_pff(pid, 9000 + j) for j, pid in enumerate(pids)]
    m = seqmodel.train(parts, epochs=EPOCHS, hidden=HIDDEN, seed=int(os.environ.get("TSEED", "7")))
    X, y, names, per = model.sample(test, 100 * i + 99)
    probas[test] = seqmodel.predict_proba(m, X, per)
    if "swa" in m:
        probas_swa[test] = seqmodel.predict_proba(m["swa"], X, per)
    print(f"{test}: frame acc {(probas[test][:, :3].argmax(1) == y).mean():.3f}  ({time.time() - t:.0f}s)", flush=True)
np.savez_compressed(OUT, **probas)
if probas_swa:
    np.savez_compressed(OUT.replace(".npz", "_swa.npz"), **probas_swa)
