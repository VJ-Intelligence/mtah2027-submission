"""External bench: score a predictor on every external match with the official metric.

    python -m mtah.bench --pred baseline --window 125
"""
import argparse

import numpy as np

from . import baselines
from .external import idsse, skillcorner
from .scoring import score


def period_runs(period: np.ndarray):
    edges = np.flatnonzero(np.diff(period)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(period)]))
    return list(zip(starts.tolist(), stops.tolist()))


# SoccerTrack v2 is not ingested yet: its Hugging Face dataset is gated (login + terms).
DATASETS = {"idsse": idsse, "skillcorner": skillcorner}


def matches(datasets=None):
    for ds, mod in DATASETS.items():
        if datasets and ds not in datasets:
            continue
        for mid in mod.MATCH_IDS:
            if mod.path(mid).exists():
                yield f"{ds}/{mid}", mod.load(mid)


def run(predict, datasets=None, **kw):
    rows = []
    for name, m in matches(datasets):
        pred = predict(m, **kw)
        s = score(m["label"], pred, period_runs(m["period"]))
        rows.append((name, s))
        print(f"{name:20s} spellF1 {s['spell_f1']:.4f}  macroF1 {s['macro_f1']:.4f}  "
              f"spells gt {s['n_gt_spells']:4d} pred {s['n_pred_spells']:4d}")
    groups = {}
    for name, s in rows:
        groups.setdefault(name.split("/")[0], []).append(s["spell_f1"])
    if len(groups) > 1:
        groups["ALL"] = [s["spell_f1"] for _, s in rows]
    for g, v in groups.items():
        f = np.array(v)
        print(f"{'MEAN ' + g:20s} spellF1 {f.mean():.4f}  (n={len(f)}, sd {f.std():.4f})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="baseline", choices=["baseline", "oracle"])
    ap.add_argument("--window", type=int, default=125)
    ap.add_argument("--datasets", nargs="*", choices=list(DATASETS), help="default: all")
    a = ap.parse_args()
    if a.pred == "oracle":
        run(lambda m: m["label"], a.datasets)
    else:
        run(baselines.predict, a.datasets, window=a.window)


if __name__ == "__main__":
    main()
