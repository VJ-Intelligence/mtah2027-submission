"""Smoke test: PFF WC2022 matches through the real training input path (degrader + features).

Mirrors model.sample_sc (estimated objects -> NaN, ball gaps topped up by the degrader) and
model.sample_st (controlled-possession labels from act_H / act_A), without touching model.py.
Prints aggregate numbers only.  Usage: PYTHONPATH=src python3 scripts/smoke_pff.py [id ...]
"""
import sys

import numpy as np

from mtah import features
from mtah.degrade import DegradeParams, degrade
from mtah.external import pff
from mtah.labels import controlled
from mtah.model import CLS

for mid in sys.argv[1:] or pff.match_ids():
    m = dict(pff.load(mid))
    m["home"] = np.where(m["home_detected"][..., None], m["home"], np.nan)
    m["away"] = np.where(m["away_detected"][..., None], m["away"], np.nan)
    m["ball"] = np.where(m["ball_detected"][:, None], m["ball"], np.nan)
    lab = controlled(m["label"], m["period"], pff.action_frames(mid), 1.0)
    live = lab != "D"
    base = DegradeParams()
    base.live_missing = max(0.0, base.live_missing - float(np.mean(np.isnan(m["ball"][live, 0]))))
    d = degrade(m, seed=9000, base=base)
    X, names = features.build(d)
    y = np.array([CLS[c] for c in lab], np.int8)
    assert X.shape[0] == len(y) == len(m["period"]), "row count mismatch"
    print(f"{mid}: X {X.shape}, features {len(names)}, NaN share {np.isnan(X).mean():.3f}, "
          f"all-NaN columns {int(np.isnan(X).all(0).sum())}, labels D/H/A {np.bincount(y, minlength=3).tolist()}",
          flush=True)
