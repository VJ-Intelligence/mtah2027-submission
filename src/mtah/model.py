"""Frame classifier (LightGBM) on identity-free features, trained on degraded external matches."""
import numpy as np

from . import features
from .degrade import degrade
from .external import idsse
from .labels import controlled
from .paths import ext_dir

CLS = {"D": 0, "H": 1, "A": 2}


def _code_version() -> str:
    """Hash of the code behind cached samples: a code change can never reuse stale features."""
    import hashlib
    from pathlib import Path
    here = Path(__file__).parent
    h = hashlib.sha1()
    for f in ("features.py", "degrade.py", "labels.py", "model.py", "external/idsse.py",
              "external/skillcorner.py", "external/soccertrack.py", "external/pff.py"):
        h.update((here / f).read_bytes())
    return h.hexdigest()[:8]


VER = _code_version()
PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, seed=7, deterministic=True, num_threads=8)


def gt_labels(mid: str, m: dict) -> np.ndarray:
    return controlled(m["label"], m["period"], idsse.action_frames(mid, m["period"]), 1.0)


def sample(mid: str, seed: int):
    """(X, y, names, period) for one degraded copy of an external match, cached on disk."""
    cache = ext_dir() / "cache" / f"idsse_{mid}_s{seed}_{VER}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["X"], z["y"], list(z["names"]), z["period"]
    m = idsse.load(mid)
    d = degrade(m, seed=seed)
    X, names = features.build(d)
    y = np.array([CLS[c] for c in gt_labels(mid, m)], np.int8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, names=np.array(names), period=m["period"])
    return X, y, names, m["period"]


def train(parts, step=4, rounds=400):
    import lightgbm as lgb
    Xs, ys = [], []
    for X, y, names in parts:
        Xs += [X[::step], features.swap_teams(X, names)[::step]]
        sw = y.copy()
        sw[y == 1], sw[y == 2] = 2, 1
        ys += [y[::step], sw[::step]]
    ds = lgb.Dataset(np.concatenate(Xs), np.concatenate(ys), free_raw_data=True)
    return lgb.train(PARAMS, ds, num_boost_round=rounds)


def predict_proba(booster, X, names):
    """Symmetric inference: average the model and its team-swapped mirror."""
    p = booster.predict(X)
    q = booster.predict(features.swap_teams(X, names))[:, [0, 2, 1]]
    return (p + q) / 2


def sc_labels(label: np.ndarray, period: np.ndarray, max_gap_s: float = 2.0) -> np.ndarray:
    """SkillCorner: short 'D' gaps inside play between same-team spells = loose ball -> keep team."""
    from .labels import runs_of
    out = label.copy()
    bounds = np.flatnonzero(np.diff(period)) + 1
    for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(label)]):
        r = runs_of(out, lo, hi)
        for i in range(1, len(r) - 1):
            a, b, v = r[i]
            if v == "D" and (b - a) < max_gap_s * 25 and r[i - 1][2] == r[i + 1][2] != "D":
                out[a:b] = r[i - 1][2]
    return out


def sample_sc(mid: str, seed: int):
    """SkillCorner match -> degraded features. Undetected objects become NaN (like the ATD)."""
    from .external import skillcorner
    cache = ext_dir() / "cache" / f"sc_{mid}_s{seed}_{VER}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["X"], z["y"], list(z["names"]), z["period"]
    m = skillcorner.load(mid)
    m = dict(m)
    m["home"] = np.where(m["home_detected"][..., None], m["home"], np.nan)
    m["away"] = np.where(m["away_detected"][..., None], m["away"], np.nan)
    m["ball"] = np.where(m["ball_detected"][:, None], m["ball"], np.nan)
    m["label"] = sc_labels(m["label"], m["period"])
    from .degrade import DegradeParams
    live = m["label"] != "D"
    already = float(np.mean(np.isnan(m["ball"][live, 0]))) if live.any() else 0.0
    base = DegradeParams()
    base.live_missing = max(0.0, base.live_missing - already)
    d = degrade(m, seed=seed, base=base)
    X, names = features.build(d)
    y = np.array([CLS[c] for c in m["label"]], np.int8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, names=np.array(names), period=m["period"])
    return X, y, names, m["period"]


def sample_st(mid: str, seed: int):
    """SoccerTrack v2 match -> degraded features, controlled-possession labels."""
    from .external import soccertrack
    cache = ext_dir() / "cache" / f"st_{mid}_s{seed}_{VER}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["X"], z["y"], list(z["names"]), z["period"]
    m = soccertrack.load(mid)
    lab = controlled(m["label"], m["period"], soccertrack.action_frames(mid), 1.0)
    d = degrade(m, seed=seed)
    X, names = features.build(d)
    y = np.array([CLS[c] for c in lab], np.int8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, names=np.array(names), period=m["period"])
    return X, y, names, m["period"]


def sample_pff(mid: str, seed: int):
    """PFF FC World Cup 2022 match -> degraded features, controlled-possession labels.
    Broadcast tracking sees only ~6-11 players per frame; the provider-estimated players are KEPT
    (the challenge ATD tracks ~20.5 of 22), only the estimated ball is blanked. Then degraded."""
    from .external import pff
    cache = ext_dir() / "cache" / f"pff_{mid}_s{seed}_{VER}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["X"], z["y"], list(z["names"]), z["period"]
    m = dict(pff.load(mid))
    m["ball"] = np.where(m["ball_detected"][:, None], m["ball"], np.nan)
    lab = controlled(m["label"], m["period"], {"H": m["act_H"], "A": m["act_A"]}, 1.0)
    from .degrade import DegradeParams
    live = lab != "D"
    already = float(np.mean(np.isnan(m["ball"][live, 0]))) if live.any() else 0.0
    base = DegradeParams()
    base.live_missing = max(0.0, base.live_missing - already)
    d = degrade(m, seed=seed, base=base)
    X, names = features.build(d)
    y = np.array([CLS[c] for c in lab], np.int8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, names=np.array(names), period=m["period"])
    return X, y, names, m["period"]
