"""Regression tests on synthetic data (no match data)."""
import numpy as np

from mtah import features


def _synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    home = np.stack([30 + 10 * np.sin(t / 200)[:, None] + rng.normal(0, 3, (n, 11)),
                     34 + rng.normal(0, 10, (n, 11))], -1)
    away = np.stack([70 + 10 * np.sin(t / 200)[:, None] + rng.normal(0, 3, (n, 11)),
                     34 + rng.normal(0, 10, (n, 11))], -1)
    ball = np.c_[50 + 20 * np.sin(t / 50), 34 + 10 * np.cos(t / 70), np.full(n, np.nan)]
    return {"ball": ball.astype(np.float32), "home": home.astype(np.float32),
            "away": away.astype(np.float32), "pitch": np.array([105.0, 68.0], np.float32)}


def test_rows_stay_on_their_frames_with_scattered_periods():
    """Challenge layout: period 0 before, between and after the halves. Every row must describe
    its own frame, i.e. equal the features of that period block computed on its own."""
    m = _synthetic()
    per = np.zeros(3000, np.int8)
    per[300:1400] = 1
    per[1700:2800] = 2
    m["period"] = per
    X, names = features.build(m)
    for p in (1, 2):
        idx = np.flatnonzero(per == p)
        sub = {k: (v[idx] if k in ("ball", "home", "away") else v) for k, v in m.items()}
        sub["period"] = np.full(len(idx), p, np.int8)
        Xp, _ = features.build(sub)
        assert np.allclose(np.nan_to_num(X[idx]), np.nan_to_num(Xp), atol=1e-4)


def test_team_speed_ignores_dropouts():
    m = _synthetic()
    m["period"] = np.ones(3000, np.int8)
    X, names = features.build(m)
    j = names.index("tspeed")
    m2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
    m2["home"][::7, :3] = np.nan          # players dropping in and out
    X2, _ = features.build(m2)
    assert abs(np.nanmean(X2[:, j]) - np.nanmean(X[:, j])) < 0.3
