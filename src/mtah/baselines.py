"""Reference baseline on the common format: nearest player to the ball + majority smoothing.

Same idea as the official starter/baseline.py, re-implemented on arrays so it runs on any match
in the common format. Used to sanity-check the bench, not as a submission.
"""
import numpy as np

HOME, AWAY, NOBODY = 0, 1, 2


def nearest_team(ball: np.ndarray, home: np.ndarray, away: np.ndarray) -> np.ndarray:
    b = ball[:, None, :2]
    dh = np.nanmin(np.linalg.norm(home - b, axis=2), axis=1, initial=np.inf, where=~np.isnan(home[..., 0]))
    da = np.nanmin(np.linalg.norm(away - b, axis=2), axis=1, initial=np.inf, where=~np.isnan(away[..., 0]))
    out = np.where(dh <= da, HOME, AWAY).astype(np.int8)
    out[np.isnan(ball[:, 0]) | (np.isinf(dh) & np.isinf(da))] = NOBODY
    return out


def smooth(codes: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    hc = np.concatenate(([0], np.cumsum(codes == HOME)))
    ac = np.concatenate(([0], np.cumsum(codes == AWAY)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    out = np.full(len(codes), "D", dtype="<U1")
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], "H", "A")
    return out


def predict(match: dict, window: int = 125) -> np.ndarray:
    return smooth(nearest_team(match["ball"], match["home"], match["away"]), window)
