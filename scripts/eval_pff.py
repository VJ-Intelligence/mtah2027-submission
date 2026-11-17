"""Score the committed submission models (models/tcn_*.pt) on the PFF WC2022 matches.

The models never saw these matches, so this is an out-of-domain test, scored with the official
metric. Same recipe as the external bench: the match is made challenge-like (estimated objects
blanked, then one degraded copy with a fixed seed), the TCN ensemble predicts P(D,H,A), Viterbi
decodes with the submission penalties. Ground truth = PFF labels after labels.controlled().

Per match, three numbers (spellF1 / macroF1):
  model        vs the labels as stored
  model+lag    vs the labels shifted by the measured event lag (event_lag_frames)
  baseline     official baseline approach (nearest player, window 125) vs the stored labels
Prints aggregate numbers only.

Usage: PYTHONPATH=src python3 scripts/eval_pff.py [--clean] [id ...]
  --clean  skip the degrader (estimated objects still blanked)
"""
import argparse
import pickle
import time

import numpy as np
import torch

from mtah import baselines, features, seqmodel
from mtah.bench import period_runs
from mtah.decode import decode
from mtah.degrade import DegradeParams, degrade
from mtah.external import pff
from mtah.labels import controlled
from mtah.scoring import score
from mtah.submit import MODELS, PEN_DEAD, PEN_TEAM, TCNS


def load_tcns():
    out = []
    for name, *_, w in TCNS:
        with open(MODELS / f"tcn_{name}_meta.pkl", "rb") as fh:
            meta = pickle.load(fh)
        norm = seqmodel.Normaliser.__new__(seqmodel.Normaliser)
        norm.mu, norm.sd = meta["mu"], meta["sd"]
        net = seqmodel.Net(len(meta["names"]))
        net.load_state_dict(torch.load(MODELS / f"tcn_{name}.pt", map_location="cpu", weights_only=True))
        out.append(({"net": net.eval(), "norm": norm, "names": meta["names"]}, w))
    return out


def shift(label, period, lag):
    """Shift labels by `lag` frames inside each period (np.roll semantics, edges repeated)."""
    out = label.copy()
    for a, b in period_runs(period):
        seg = label[a:b]
        out[a:b] = np.roll(seg, lag)
        if lag < 0:
            out[b + lag:b] = seg[-1]
        elif lag > 0:
            out[a:a + lag] = seg[0]
    return out


def challenge_like(m: dict, clean: bool) -> dict:
    m = dict(m)
    m["home"] = np.where(m["home_detected"][..., None], m["home"], np.nan)
    m["away"] = np.where(m["away_detected"][..., None], m["away"], np.nan)
    m["ball"] = np.where(m["ball_detected"][:, None], m["ball"], np.nan)
    if clean:
        return m
    live = m["label"] != "D"
    base = DegradeParams()
    base.live_missing = max(0.0, base.live_missing - float(np.mean(np.isnan(m["ball"][live, 0]))))
    return degrade(m, seed=9000, base=base)


def fmt(s):
    return f"{s['spell_f1']:.4f} / {s['macro_f1']:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    seqmodel.DEVICE = "cpu"
    tcns = load_tcns()
    res = {"model": [], "model+lag": [], "baseline": []}
    print(f"{'match':6s} {'model':>17s} {'model+lag':>17s} {'baseline':>17s}  spells gt/pred  lag")
    for mid in a.ids or pff.match_ids():
        t0 = time.time()
        m = pff.load(mid)
        gt = controlled(m["label"], m["period"], pff.action_frames(mid), 1.0)
        lag = int(m["event_lag_frames"])
        d = challenge_like(m, a.clean)
        X, names = features.build(d)
        assert names == tcns[0][0]["names"], "feature mismatch"
        with torch.no_grad():
            p = sum(w * seqmodel.predict_proba(t, X, m["period"])[:, :3] for t, w in tcns)
        p = p / sum(w for _, w in tcns)
        pred = decode(p, m["period"], PEN_TEAM, PEN_DEAD)
        runs = period_runs(m["period"])
        s = score(gt, pred, runs)
        sl = score(shift(gt, m["period"], lag), pred, runs)
        sb = score(gt, baselines.predict(d, window=125), runs)
        for k, v in (("model", s), ("model+lag", sl), ("baseline", sb)):
            res[k].append(v["spell_f1"])
        print(f"{mid:6s} {fmt(s):>17s} {fmt(sl):>17s} {fmt(sb):>17s}  "
              f"{s['n_gt_spells']:4d}/{s['n_pred_spells']:4d}  {lag:4d}  ({time.time() - t0:.0f}s)", flush=True)
    print("MEAN spellF1  " + "  ".join(f"{k} {np.mean(v):.4f}" for k, v in res.items()))


if __name__ == "__main__":
    main()
