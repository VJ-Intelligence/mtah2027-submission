"""Make clean external tracking look like the challenge's automatic tracking (ATD).

Calibration targets are DESCRIPTIVE statistics published in the official DATA.md / CHALLENGE.md
(not fitted on the match): ball present on ~59% of graded frames, missing frames concentrated in
stoppages (~70%) and in long gaps, ball position smoothed (~17-frame window), ~20.5 of 22 players
tracked on average with occasional duplicates, no identities. Every call is seeded -> reproducible.

Components (each randomised per match, domain-randomisation style):
  1. aerial projection: ball with height z is projected onto the grass along the camera ray
  2. ball noise + smoothing (centred or causal window)
  3. ball gaps: most stoppage time, plus in-play gaps triggered by aerial / crowded situations
  4. player jitter, dropouts, duplicates, slot shuffling
"""
from dataclasses import dataclass

import numpy as np

FPS = 25


@dataclass
class DegradeParams:
    cam_height: float = 25.0          # m
    cam_offset: float = 35.0          # m behind the near touchline
    ball_sigma: float = 0.6           # m, white noise before smoothing
    smooth_window: int = 17           # frames
    smooth_causal_p: float = 0.3      # probability the kernel is causal (trailing)
    dead_missing: float = 0.64        # share of dead-ball frames without a ball
    live_missing: float = 0.22        # share of live frames without a ball
    live_gap_median_s: float = 3.0
    player_sigma: float = 0.25        # m, position error of a tracked player
    player_noise_window: int = 50     # frames; tracking error drifts slowly, it does not jump
    aerial_strength: float = 1.0      # 0 = tracker corrects for height, 1 = full ground projection
    out_missing: float = 0.8          # chance a ball stretch outside the pitch is not tracked
    jump_rate: float = 0.002          # per frame: start of a short false ball detection
    players_present: float = 20.5 / 22
    duplicate_rate: float = 0.002     # per player-frame start of a duplicate track
    randomise: bool = True


def _sample_params(rng: np.random.Generator, base: DegradeParams) -> DegradeParams:
    if not base.randomise:
        return base
    p = DegradeParams(**base.__dict__)
    p.cam_height = rng.uniform(15, 35)
    p.cam_offset = rng.uniform(20, 50)
    p.ball_sigma = rng.uniform(0.1, 0.8)
    p.smooth_window = int(rng.integers(13, 26))  # DATA.md: ~17-frame smoothing
    p.dead_missing = float(np.clip(rng.normal(base.dead_missing, 0.08), 0.3, 0.9))
    p.live_missing = float(np.clip(rng.normal(base.live_missing, 0.06), 0.05, 0.45))
    p.live_gap_median_s = rng.uniform(3.5, 9.0)
    p.player_sigma = rng.uniform(0.1, 0.5)
    p.player_noise_window = int(rng.integers(25, 100))
    p.aerial_strength = rng.uniform(0.0, 1.0)
    p.out_missing = rng.uniform(0.7, 1.0)
    p.jump_rate = rng.uniform(0.0, 0.001)
    p.players_present = float(np.clip(rng.normal(base.players_present, 0.03), 0.8, 1.0))
    return p


def project_aerial(ball: np.ndarray, pitch: np.ndarray, p: DegradeParams) -> np.ndarray:
    """Ground-plane point where the camera ray through the 3D ball hits z=0."""
    out = ball[:, :2].copy()
    z = np.nan_to_num(ball[:, 2], nan=0.0)
    cam = np.array([pitch[0] / 2, -p.cam_offset])
    h = p.cam_height
    z = np.clip(z, 0, h * 0.5)
    scale = 1 + p.aerial_strength * (h / (h - z) - 1)
    out = cam + (out - cam) * scale[:, None]
    return out


def _smooth(x: np.ndarray, w: int, causal: bool) -> np.ndarray:
    """Moving mean over valid samples (NaN-aware), per column."""
    if w <= 1:
        return x
    valid = ~np.isnan(x)
    xv = np.where(valid, x, 0.0)
    cs = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(xv, axis=0)])
    cn = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(valid, axis=0)])
    n = len(x)
    i = np.arange(n)
    if causal:
        lo, hi = np.maximum(0, i - w + 1), i + 1
    else:
        lo, hi = np.maximum(0, i - w // 2), np.minimum(n, i + w // 2 + 1)
    s, c = cs[hi] - cs[lo], cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[~valid] = np.nan
    return out


def _crowded(ball_xy, home, away, radius=3.0, k=3):
    pl = np.concatenate([home, away], axis=1)
    d = np.linalg.norm(pl - ball_xy[:, None, :], axis=2)
    return (np.nan_to_num(d, nan=1e9) < radius).sum(axis=1) >= k


def ball_gaps(label, ball_xyz, home, away, period, p: DegradeParams, rng) -> np.ndarray:
    """Boolean mask of frames where the ball is missing."""
    n = len(label)
    miss = np.zeros(n, bool)
    dead = label == "D"
    # stoppages: drop a long contiguous chunk of each dead run with prob ~dead_missing coverage
    cut = np.flatnonzero((dead[1:] != dead[:-1]) | (period[1:] != period[:-1])) + 1
    starts, stops = np.r_[0, cut], np.r_[cut, n]
    for a, b in zip(starts, stops):
        if not dead[a]:
            continue
        L = b - a
        if rng.random() < min(1.0, p.dead_missing * 1.15):
            cover = rng.uniform(0.6, 1.0)
            m = int(L * cover)
            s = a + int(rng.integers(0, max(1, L - m + 1)))
            miss[s:s + m] = True
    # in play: gaps triggered by aerial or crowded situations until live_missing reached
    live = ~dead
    z = np.nan_to_num(ball_xyz[:, 2], nan=0.0)
    trig = live & ((z > 1.8) | _crowded(ball_xyz[:, :2], home, away))
    weights = np.where(trig, 5.0, 1.0) * live
    target = p.live_missing * live.sum()
    mu = np.log(p.live_gap_median_s * FPS)
    pw = weights / weights.sum()
    for _ in range(50):
        if (miss & live).sum() >= target:
            break
        starts = rng.choice(n, size=200, p=pw)
        lens = np.clip(rng.lognormal(mu, 0.9, size=200), 1 * FPS, 30 * FPS).astype(int)
        for s, L in zip(starts, lens):
            if (miss & live).sum() >= target:
                break
            e = min(n, s + L)
            same = int((period[s:e] == period[s]).sum())
            miss[s:s + same] = True
    return miss


def degrade(match: dict, seed: int, base: DegradeParams = DegradeParams()) -> dict:
    rng = np.random.default_rng(seed)
    p = _sample_params(rng, base)
    label, period, pitch = match["label"], match["period"], match["pitch"]
    ball = match["ball"].astype(np.float64)
    home = match["home"].astype(np.float64).copy()
    away = match["away"].astype(np.float64).copy()

    # 1-2. ball: aerial projection, noise, smoothing
    xy = project_aerial(ball, pitch, p)
    out_miss = _out_of_pitch_missing(xy, pitch, p, rng)   # decided before noise: no flicker
    xy = xy + rng.normal(0, p.ball_sigma, xy.shape)
    miss = ball_gaps(label, ball, home, away, period, p, rng)
    miss |= out_miss
    xy = _ball_jumps(xy, home, away, p, rng)
    xy[miss] = np.nan
    causal = rng.random() < p.smooth_causal_p
    xy = _smooth(xy, p.smooth_window, causal)

    # 4. players: jitter, dropouts (runs), duplicates, shuffle slots
    for team in (home, away):
        team += _smooth_noise(rng, team.shape, p.player_sigma, p.player_noise_window)
        drop_p = 1 - p.players_present
        n, P = team.shape[:2]
        # dropouts as runs: Markov chain per slot
        # mean run length 35 frames; only players that are actually tracked can drop out
        start = (rng.random((n, P)) < drop_p / 35) & ~np.isnan(team[..., 0])
        run = np.zeros((n, P), bool)
        lengths = rng.integers(10, 60, size=(n, P))
        for j in range(P):
            for s in np.flatnonzero(start[:, j]):
                run[s:s + lengths[s, j], j] = True
        team[run] = np.nan
    dup_home = _duplicates(home, p, rng)
    dup_away = _duplicates(away, p, rng)
    home = _shuffle_slots(dup_home, rng)
    away = _shuffle_slots(dup_away, rng)

    return {
        "ball": np.c_[xy, np.full(len(xy), np.nan)].astype(np.float32),
        "home": home.astype(np.float32), "away": away.astype(np.float32),
        "label": label, "period": period, "pitch": pitch,
        "params": np.array(str({**p.__dict__, "causal": causal})),
    }


def _out_of_pitch_missing(xy, pitch, p, rng, margin=0.0):
    """Ball stretches outside the pitch rectangle are mostly not tracked (camera framing,
    occlusion by boards and staff): each stretch is dropped with probability out_missing."""
    L, W = float(pitch[0]), float(pitch[1])
    out = (xy[:, 0] < -margin) | (xy[:, 0] > L + margin) | (xy[:, 1] < -margin) | (xy[:, 1] > W + margin)
    out &= ~np.isnan(xy[:, 0])
    miss = np.zeros(len(xy), bool)
    if not out.any():
        return miss
    d = np.diff(np.r_[0, out.astype(int), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a >= 10 and rng.random() < p.out_missing:
            miss[a:b] = True
    return miss


def _ball_jumps(xy, home, away, p, rng):
    """Short false detections: for 2-10 frames the ball sits on a random player's position."""
    n = len(xy)
    starts = np.flatnonzero(rng.random(n) < p.jump_rate)
    if not len(starts):
        return xy
    xy = xy.copy()
    allp = np.concatenate([home, away], axis=1)
    for s in starts:
        L = int(rng.integers(2, 11))
        cand = allp[s]
        cand = cand[~np.isnan(cand[:, 0])]
        if not len(cand):
            continue
        xy[s:s + L] = cand[rng.integers(len(cand))] + rng.normal(0, 0.5, 2)
    return xy


def _smooth_noise(rng, shape, sigma, window):
    """Temporally correlated position error: white noise averaged over `window` frames, rescaled
    to standard deviation `sigma`. Per-frame white noise would add ~sigma*25*sqrt(2) m/s of fake
    speed (a 0.25 m jitter reads as ~9 m/s), which no real tracking system shows."""
    from scipy.ndimage import uniform_filter1d
    w = rng.normal(0, 1, shape)
    w = uniform_filter1d(w, size=max(1, window), axis=0, mode="nearest")
    return w * (sigma / (w.std() + 1e-9))


def _duplicates(team, p, rng):
    n, P = team.shape[:2]
    extra = np.full((n, 2, 2), np.nan)
    for k in range(2):
        starts = np.flatnonzero(rng.random(n) < p.duplicate_rate)
        for s in starts:
            j = int(rng.integers(0, P))
            L = int(rng.integers(5, 50))
            seg = team[s:s + L, j] + rng.normal(0, 0.8, (min(L, n - s), 2))
            extra[s:s + L, k] = seg
    return np.concatenate([team, extra], axis=1)


def _shuffle_slots(team, rng):
    """Identity-free: permute slots in blocks so slot index carries no identity."""
    n, P = team.shape[:2]
    out = team.copy()
    block = 250
    for s in range(0, n, block):
        perm = rng.permutation(P)
        out[s:s + block] = team[s:s + block][:, perm]
    return out


def describe(m: dict) -> dict:
    """Descriptive stats to compare with the published DATA.md numbers."""
    live = m["label"] != "D"
    miss = np.isnan(m["ball"][:, 0])
    present = (~np.isnan(m["home"][..., 0])).sum(1) + (~np.isnan(m["away"][..., 0])).sum(1)
    cut = np.flatnonzero(np.diff(miss.astype(int)) != 0) + 1
    runs = np.split(miss, cut)
    gaps = np.array([len(r) / FPS for r in runs if r[0]])
    return {"ball_missing": miss.mean(), "missing_in_dead": (miss & ~live).sum() / max(1, miss.sum()),
            "gap_median_s": float(np.median(gaps)) if len(gaps) else 0, "n_gaps": len(gaps),
            "gaps_1_3s": float(((gaps >= 1) & (gaps <= 3)).mean()) if len(gaps) else 0,
            "objects_mean": present.mean()}
