"""Identity-free per-frame features from (degraded or challenge) tracking in the common format.

Everything is computed per half, team-symmetric (H block / A block), so the model can be trained
with team-swap augmentation. No player identities are used.
"""
import numpy as np

FPS = 25


def _roll(x: np.ndarray, w: int, mode: str = "c") -> np.ndarray:
    """NaN-aware rolling mean. mode c=centred, p=past only, f=future only."""
    valid = ~np.isnan(x)
    xv = np.where(valid, x, 0.0)
    cs = np.r_[0.0, np.cumsum(xv)]
    cn = np.r_[0, np.cumsum(valid)]
    n = len(x)
    i = np.arange(n)
    if mode == "c":
        lo, hi = np.maximum(0, i - w), np.minimum(n, i + w + 1)
    elif mode == "p":
        lo, hi = np.maximum(0, i - w), i + 1
    else:
        lo, hi = i, np.minimum(n, i + w + 1)
    c = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, (cs[hi] - cs[lo]) / np.maximum(c, 1), np.nan)


SINCE_CAP = 250  # frames


def _team_vx(T: np.ndarray, k: int = 5, max_jump: float = 3.0) -> np.ndarray:
    """Median x-velocity (m/s) of a team's players. Each player at frame t is matched to the
    nearest player of the same team at t-k; matches further than max_jump m are discarded
    (identity switch or a player appearing). Slot order is irrelevant (identity-free)."""
    n = len(T)
    v = np.full(n, np.nan, np.float32)
    if n <= k:
        return v
    a, b = T[k:], T[:-k]                                    # (n-k, P, 2)
    d = np.linalg.norm(a[:, :, None, :] - b[:, None, :, :], axis=-1)   # (n-k, P, P)
    d = np.where(np.isnan(d), np.inf, d)
    j = d.argmin(2)
    dmin = np.take_along_axis(d, j[..., None], 2)[..., 0]
    dx = a[..., 0] - np.take_along_axis(b[..., 0], j, 1)
    dx = np.where(dmin <= max_jump, dx, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(dx, axis=1)
    v[k:] = med * FPS / k
    return v


def _since_seen(present: np.ndarray, reverse=False) -> np.ndarray:
    """Frames since (or until, if reverse) the ball was last present."""
    p = present[::-1] if reverse else present
    idx = np.where(p, np.arange(len(p)), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last >= 0, np.arange(len(p)) - last, SINCE_CAP).astype(np.float32)
    out = np.minimum(out, SINCE_CAP)  # beyond ~10 s the exact gap length carries no information
    return out[::-1] if reverse else out


def _ffill(x: np.ndarray) -> np.ndarray:
    idx = np.where(~np.isnan(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    return out


def _nms(a, w):
    """Keep frames that are the max of +-w (non-maximum suppression)."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(a, w, constant_values=-np.inf)
    mx = sliding_window_view(pad, 2 * w + 1).max(axis=1)
    return a >= mx


def _touch_features(bxy, present, hd, ad, thr_acc=30.0, near=3.5):
    """Candidate touches = local maxima of ball acceleration (direction or speed change) with a
    player close to the ball. Side +1 = home nearer, -1 = away nearer."""
    n = len(bxy)
    sx = np.stack([_roll(bxy[:, 0], 3), _roll(bxy[:, 1], 3)], axis=1)
    vel = np.gradient(sx, axis=0) * FPS
    acc = np.linalg.norm(np.gradient(vel, axis=0), axis=1) * FPS
    acc = np.where(present, acc, np.nan)
    a = np.nan_to_num(acc, nan=0.0)
    peak = (a > thr_acc) & _nms(a, 12)
    dmin = np.fmin(hd, ad)
    peak &= np.nan_to_num(dmin, nan=99) < near
    side = np.where(np.nan_to_num(hd, nan=99) < np.nan_to_num(ad, nan=99), 1.0, -1.0)
    t_idx = np.flatnonzero(peak)
    t_side = side[t_idx]
    out = {}
    # last / next touch side and distance in time
    pos = np.searchsorted(t_idx, np.arange(n), side="right") - 1
    has = pos >= 0
    out["t_last_side"] = np.where(has, t_side[np.clip(pos, 0, None)], 0.0) if len(t_idx) else np.zeros(n)
    out["t_last_age"] = np.minimum(np.where(has, np.arange(n) - t_idx[np.clip(pos, 0, None)], SINCE_CAP) if len(t_idx) else np.full(n, SINCE_CAP), SINCE_CAP).astype(float)
    nxt = pos + 1
    hasn = nxt < len(t_idx)
    out["t_next_side"] = np.where(hasn, t_side[np.clip(nxt, 0, len(t_idx) - 1)], 0.0) if len(t_idx) else np.zeros(n)
    out["t_next_in"] = np.minimum(np.where(hasn, t_idx[np.clip(nxt, 0, len(t_idx) - 1)] - np.arange(n), SINCE_CAP) if len(t_idx) else np.full(n, SINCE_CAP), SINCE_CAP).astype(float)
    # previous touch side (chain): same side twice in a row = likely controlled
    prev = pos - 1
    out["t_prev_side"] = np.where(prev >= 0, t_side[np.clip(prev, 0, None)], 0.0) if len(t_idx) else np.zeros(n)
    tb = np.zeros(n)
    tb[t_idx] = t_side
    out["tbal"] = tb
    out["t_here"] = peak.astype(float)
    return out


def _half(ball, home, away, pitch):
    L, W = float(pitch[0]), float(pitch[1])
    n = len(ball)
    bxy = ball[:, :2].astype(np.float64)
    present = ~np.isnan(bxy[:, 0])
    # attacking direction: home defends the side where its centroid sits on average
    hcx = np.nanmean(home[..., 0], axis=1)
    acx = np.nanmean(away[..., 0], axis=1)
    home_left = np.nanmean(hcx) < np.nanmean(acx)
    sgn = 1.0 if home_left else -1.0  # +x = towards away goal (home attacking)

    def team(T, other):
        d = np.linalg.norm(T - bxy[:, None, :], axis=2)
        dmin = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
        dmin[~np.isfinite(dmin)] = np.nan
        n3 = np.nansum(d < 3, axis=1).astype(float)
        n6 = np.nansum(d < 6, axis=1).astype(float)
        cx = np.nanmean(T[..., 0], axis=1)
        cy = np.nanmean(T[..., 1], axis=1)
        spread = np.nanstd(T[..., 0], axis=1) + np.nanstd(T[..., 1], axis=1)
        cnt = (~np.isnan(T[..., 0])).sum(1).astype(float)
        n3[~present] = np.nan
        n6[~present] = np.nan
        return dmin, n3, n6, cx, cy, spread, cnt

    with np.errstate(all="ignore"):
        hd, hn3, hn6, hcx, hcy, hsp, hcnt = team(home, away)
        ad, an3, an6, acx, acy, asp, acnt = team(away, home)
        # forward velocities of team centroids (m/s), positive = towards opponent goal
        # median player displacement (nearest-neighbour match k frames back), robust to players
        # dropping in and out of the tracking, which would make the centroid jump
        hv = _roll(_team_vx(home), 5) * sgn
        av = -_roll(_team_vx(away), 5) * sgn
        bspeed = np.linalg.norm(np.gradient(bxy, axis=0), axis=1) * FPS
        bspeed[~present] = np.nan
        bacc = np.abs(np.gradient(_roll(bspeed, 2))) * FPS
        # ball position in home-attacking frame, 0 = home goal line, 1 = away goal line
        bx_rel = (bxy[:, 0] / L) if home_left else (1 - bxy[:, 0] / L)
        by = np.abs(bxy[:, 1] / W - 0.5)
        out_of_pitch = ((bxy[:, 0] < 0) | (bxy[:, 0] > L) | (bxy[:, 1] < 0) | (bxy[:, 1] > W)).astype(float)
        out_of_pitch[~present] = np.nan
        dd = hd - ad
        dd_last = _ffill(dd)
        dn3 = hn3 - an3
        dv = hv - av
        # centroid gap: compactness and who is higher up the pitch
        hcx_rel = (hcx / L) if home_left else (1 - hcx / L)
        acx_rel = (acx / L) if home_left else (1 - acx / L)
        allsp = np.nanmean(np.stack([hv, -av]), axis=0)

    tf = _touch_features(bxy, present, hd, ad)

    f = {
        **tf,
        "bp": present.astype(float), "bx": bx_rel, "by": by, "bspeed": bspeed, "bacc": bacc,
        "oop": out_of_pitch, "hd": hd, "ad": ad, "dd": dd, "dd_last": dd_last,
        "hn3": hn3, "an3": an3, "dn3": dn3, "hn6": hn6, "an6": an6,
        "hcx": hcx_rel, "acx": acx_rel, "hsp": hsp, "asp": asp, "hv": hv, "av": av, "dv": dv,
        "hcnt": hcnt, "acnt": acnt, "since": _since_seen(present), "until": _since_seen(present, True),
        "tspeed": np.abs(hv) + np.abs(av), "allsp": allsp,
    }
    for k in ("bp", "dd", "dn3", "dv", "bspeed", "tspeed", "oop", "hn6", "an6"):
        for w in (12, 50, 125):
            f[f"{k}_c{w}"] = _roll(f[k], w, "c")
    for k in ("dd", "dn3", "dv"):
        f[f"{k}_c250"] = _roll(f[k], 250, "c")
    for k in ("dd", "dn3", "bp", "tbal"):
        for w in (25, 75):
            f[f"{k}_p{w}"] = _roll(f[k], w, "p")
            f[f"{k}_f{w}"] = _roll(f[k], w, "f")
    return f


SWAP = {"hd": "ad", "hn3": "an3", "hn6": "an6", "hcx": "acx", "hsp": "asp", "hv": "av", "hcnt": "acnt"}


def build(match: dict) -> tuple[np.ndarray, list]:
    """Feature matrix (N, F) for a match in the common format, plus feature names."""
    period = match["period"]
    X = None
    names = None
    for p in np.unique(period):
        # rows go back to their own frames: periods need not be contiguous or ordered
        # (the challenge match has period 0 before kick-off, at half time and after the end)
        idx = np.flatnonzero(period == p)
        f = _half(match["ball"][idx], match["home"][idx], match["away"][idx], match["pitch"])
        if names is None:
            names = list(f)
            X = np.full((len(period), len(names)), np.nan, np.float32)
        X[idx] = np.stack([f[k] for k in names], axis=1)
    return X, names


def swap_teams(X: np.ndarray, names: list) -> np.ndarray:
    """Mirror H<->A in the feature matrix (for augmentation / symmetric inference)."""
    Y = X.copy()
    ix = {n: i for i, n in enumerate(names)}
    for a, b in SWAP.items():
        Y[:, ix[a]], Y[:, ix[b]] = X[:, ix[b]], X[:, ix[a]]
    for n in names:
        if n == "bx":
            Y[:, ix[n]] = 1 - X[:, ix[n]]
    Y[:, ix["hcx"]], Y[:, ix["acx"]] = 1 - X[:, ix["acx"]], 1 - X[:, ix["hcx"]]
    for n in names:
        if n.startswith(("dd", "dn3", "dv")) or n in ("allsp",):
            Y[:, ix[n]] = -X[:, ix[n]]
    return Y
