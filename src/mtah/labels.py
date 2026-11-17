"""Turn source possession labels into Metrica-like "controlled possession" labels.

The challenge GT flips when the other team gains CONTROLLED possession (CHALLENGE.md): touches,
deflections, duels without control keep the previous team. Source datasets (e.g. DFL BallPossession)
flip on touches, giving ~1.5-2x too many spells. Rule (calibrated on external data only):
a spell of team T sandwiched between spells of the other team, inside one live stretch, is
merged back into the other team unless T shows control: a deliberate T action (pass/shot event
by a T player inside the spell) or the spell lasting at least `min_control_s`.
"""
import numpy as np

FPS = 25


def runs_of(label: np.ndarray, lo: int, hi: int):
    """[(start, stop, value)] maximal runs inside [lo, hi)."""
    seg = label[lo:hi]
    if len(seg) == 0:
        return []
    cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    starts = np.concatenate(([0], cut))
    stops = np.concatenate((cut, [len(seg)]))
    return [(lo + a, lo + b, seg[a]) for a, b in zip(starts, stops)]


def controlled(label: np.ndarray, period: np.ndarray, action_frames: dict,
               min_control_s: float = 1.0, max_iter: int = 5) -> np.ndarray:
    """action_frames: {'H': sorted frame idx of deliberate H actions, 'A': ...}."""
    out = label.copy()
    bounds = np.flatnonzero(np.diff(period)) + 1
    halves = list(zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [len(label)]))))
    for _ in range(max_iter):
        changed = False
        for lo, hi in halves:
            r = runs_of(out, lo, hi)
            for i in range(1, len(r) - 1):
                a, b, t = r[i]
                if t == "D":
                    continue
                prev_t, next_t = r[i - 1][2], r[i + 1][2]
                if prev_t == "D" or next_t == "D" or prev_t != next_t or prev_t == t:
                    continue
                if (b - a) >= min_control_s * FPS:
                    continue
                af = action_frames.get(t)
                if af is not None and len(af):
                    k = np.searchsorted(af, a)
                    if k < len(af) and af[k] < b:
                        continue
                out[a:b] = prev_t
                changed = True
        if not changed:
            break
    return out


def spell_stats(label: np.ndarray, period: np.ndarray) -> dict:
    bounds = np.flatnonzero(np.diff(period)) + 1
    halves = list(zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [len(label)]))))
    durs = [(b - a) / FPS for lo, hi in halves for a, b, v in runs_of(label, lo, hi) if v != "D"]
    live_min = (label != "D").sum() / FPS / 60
    d = np.array(durs)
    return {"n": len(d), "per_live_min": len(d) / live_min, "mean_s": d.mean(),
            "median_s": float(np.median(d)), "lt2": (d < 2).mean(), "lt4": (d < 4).mean(),
            "dead_share": (label == "D").mean()}
