"""Challenge tracking -> possession CSV. Prints aggregate statistics ONLY (no rows, no frames).

    make train             train the models on degraded EXTERNAL matches -> models/
    make submission        models/ + challenge ATD -> submissions/<team>.csv

Weighted ensemble of dilated TCNs on identity-free features, symmetric (team-swap)
inference, Viterbi decoding. Nothing is fitted on the challenge match. Inference runs on CPU from
the committed model files, so the CSV is bit-for-bit reproducible.
"""
import argparse
import os
import pickle

import numpy as np

from . import features, model
from .decode import decode
from .external import atd, idsse
from .paths import REPO, official_on_path

MODELS = REPO / "models"
TRAIN_SEEDS = (0, 1, 2, 3)
EPOCHS = 6
PEN_TEAM, PEN_DEAD = 8.0, 24.0
# TCN ensemble: (name, torch seed, + SkillCorner, SoccerTrack copies, + PFF WC2022, weight).
# Composition chosen on the external leave-one-match-out bench (scripts/ensemble_lab.py).
# PFF (64 matches) added 14.11.: single TCN 0.661 -> 0.691 on the bench, 6/7 folds better.
TCNS = (
    ("sc_st_pff", 7, True, 1, True, 1.0),
    ("sc_st_pff_s8", 8, True, 1, True, 1.0),
    ("sc_st3_pff", 7, True, 3, True, 1.0),
    ("dfl_st_pff", 7, False, 1, True, 1.0),
)
GBDT_WEIGHT = 0.0  # LightGBM adds nothing on the bench once the TCNs are in; not trained at 0


def challenge_match() -> dict:
    official_on_path()
    import games  # official repo
    paths = games.asset_paths(data_dir=os.environ.get("MTAH_DATA_DIR"))
    n = games.source_frames()
    period = np.zeros(n, np.int8)
    for half, a, b in games.half_video_blocks(games.GAME):
        period[a:b + 1] = half
    return atd.convert(paths["atd_meta"], paths["atd_raw"], n, games.GAMES[games.GAME]["pitch"], period)


def train_all():
    import torch
    from . import seqmodel
    parts = [model.sample(mid, 100 * i + s) for i, mid in enumerate(idsse.MATCH_IDS) for s in TRAIN_SEEDS]
    MODELS.mkdir(exist_ok=True)
    if GBDT_WEIGHT > 0:
        booster = model.train([(X, y, n) for X, y, n, _ in parts])
        booster.save_model(str(MODELS / "gbdt.txt"))
    from .external import pff, skillcorner, soccertrack
    sc_parts = [model.sample_sc(sid, 5000 + j) for j, sid in enumerate(skillcorner.MATCH_IDS)]
    pff_parts = [model.sample_pff(pid, 9000 + j) for j, pid in enumerate(pff.match_ids())]
    for name, seed, with_sc, st_copies, with_pff, _ in TCNS:
        st_parts = [model.sample_st(sid, 7000 + 100 * c + j) for c in range(st_copies)
                    for j, sid in enumerate(soccertrack.TRAIN_IDS)]
        tcn = seqmodel.train(parts + (sc_parts if with_sc else []) + st_parts + (pff_parts if with_pff else []),
                             epochs=EPOCHS, seed=seed)
        torch.save(tcn["net"].to("cpu").state_dict(), MODELS / f"tcn_{name}.pt")
        with open(MODELS / f"tcn_{name}_meta.pkl", "wb") as fh:
            pickle.dump({"mu": tcn["norm"].mu, "sd": tcn["norm"].sd, "names": tcn["names"]}, fh)
    print("models written to", MODELS)


def load_models():
    import lightgbm as lgb
    import torch
    from . import seqmodel
    booster = lgb.Booster(model_file=str(MODELS / "gbdt.txt")) if GBDT_WEIGHT > 0 else None
    tcns = []
    for name, *_ in TCNS:
        with open(MODELS / f"tcn_{name}_meta.pkl", "rb") as fh:
            meta = pickle.load(fh)
        norm = seqmodel.Normaliser.__new__(seqmodel.Normaliser)
        norm.mu, norm.sd = meta["mu"], meta["sd"]
        net = seqmodel.Net(len(meta["names"]))
        net.load_state_dict(torch.load(MODELS / f"tcn_{name}.pt", map_location="cpu", weights_only=True))
        tcns.append({"net": net.eval(), "norm": norm, "names": meta["names"]})
    return booster, tcns


def predict(m: dict) -> np.ndarray:
    import torch
    from . import seqmodel
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    seqmodel.DEVICE = "cpu"
    booster, tcns = load_models()
    X, names = features.build(m)
    assert names == tcns[0]["names"], "feature mismatch"
    # TCN column 3 is the auxiliary change probability, not used
    p = sum(w * seqmodel.predict_proba(t, X, m["period"])[:, :3] for t, (*_, w) in zip(tcns, TCNS))
    if booster is not None:
        p = p + GBDT_WEIGHT * model.predict_proba(booster, X, names)
    p = p / (sum(w for *_, w in TCNS) + GBDT_WEIGHT)
    labels = decode(p, m["period"], PEN_TEAM, PEN_DEAD)
    labels = pre_kickoff_dead(labels, m["period"], kickoff_frames())
    labels[m["period"] == 0] = "X"
    return labels


def kickoff_frames() -> dict:
    """{half: first frame of its 'Kick Off (Start)' Smart Tagging window}. Generic rule input:
    before a half's kickoff the ball is dead (FAQ: the 2nd half is graded ~2,900 frames before
    its kickoff, all D). The tag window contains the real kickoff, so its start is safe."""
    official_on_path()
    import load_smart_tagging as st
    out = {}
    for e in st.load(data_dir=os.environ.get("MTAH_DATA_DIR")):
        if (e["code"] or "").strip().lower() == "set pieces" and \
                (e["tags"].get("Type") or "").strip().lower() == "kick off (start)":
            half = 1 if (e["tags"].get("Half") or "").startswith("1") else 2
            out[half] = min(out.get(half, e["start_frame"]), e["start_frame"])
    return out


def pre_kickoff_dead(labels, period, kick):
    out = labels.copy()
    n_forced = n_changed = 0
    for half, f in kick.items():
        idx = np.flatnonzero(period == half)
        if not len(idx) or f <= idx[0]:
            continue
        sel = idx[idx < f]
        n_changed += int((out[sel] != "D").sum())
        n_forced += len(sel)
        out[sel] = "D"
    print(f"pre-kickoff rule: {n_forced} frames set to D ({n_changed} changed)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    if a.train:
        train_all()
        return
    labels = predict(challenge_match())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{l}\n" for i, l in enumerate(labels))
    graded = labels != "X"
    print(f"wrote {len(labels)} rows; graded share H {np.mean(labels[graded] == 'H'):.3f} "
          f"A {np.mean(labels[graded] == 'A'):.3f} D {np.mean(labels[graded] == 'D'):.3f}; "
          f"label changes {np.flatnonzero(labels[1:] != labels[:-1]).size}")


if __name__ == "__main__":
    main()
