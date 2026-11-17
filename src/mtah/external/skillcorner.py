"""SkillCorner open data: 20 A-League 2024/25 matches, 10 fps broadcast tracking, resampled to 25 fps.

Source: https://github.com/SkillCorner/opendata, pinned at commit `SHA` below. Licence: the repo
is MIT ("Copyright (c) 2020 SkillCorner"), and the README adds no separate data licence; it only
asks users to credit SkillCorner. So: MIT, credit SkillCorner.

Files used per match: `{id}_tracking_extrapolated.jsonl` (Git LFS, ~90 MB), `{id}_match.json`
(teams, pitch size), `{id}_phases_of_play.csv` (ball-in-play windows). `{id}_dynamic_events.csv`
is downloaded too for later label refinement but not used yet.

Tracking: broadcast computer vision. Every frame lists all 22 players, but players outside the
camera view are *extrapolated* by SkillCorner (`is_detected` false); positions are kept as
given and the masks `home_detected` / `away_detected` / `ball_detected` say which were seen.
About a fifth of frames (replays, close-ups, cutaways) have no data at all: everything is NaN
there. Coordinates are centre-origin metres; shifted by (L/2, W/2) to the corner origin.

Resampling 10 -> 25 fps, per period, on the source clock t = frame / 10: positions are linearly
interpolated between the two neighbouring source frames (falling back to the nearest source
frame when one neighbour is NaN); labels and detection masks take the nearest source frame.
`fps` (25) and `source_fps` (10) are stored in the npz.

Labels (label_source "skillcorner_phase_alive_possession_group"):
  - alive  = frame lies inside a phases_of_play row, [frame_start, frame_end) (frame_end is
    exclusive: consecutive phases share the boundary frame). SkillCorner defines phases only
    while the ball is in play, so everything outside a phase is dead -> 'D'.
  - alive and per-frame `possession.group` == "home team" / "away team" -> 'H' / 'A'.
  - alive and `possession.group` null -> 'D' (SkillCorner gives no team, e.g. loose/contested
    ball; about 4% of alive frames).
Differences from DFL (idsse): DFL's BallPossession is sticky through the whole alive time, so
DFL has almost no alive 'D'; here short alive 'D' gaps exist. SkillCorner's possession group
is itself a computer-vision-derived team possession, not hand-annotated, and the ball-in-play
boundaries come from the phase framework, not from a ball-status feed.
"""
import csv
import json
import urllib.request

import numpy as np

from ..paths import ext_dir

SHA = "4340d274572876239c154c90bc507a9b3250a656"
MATCH_IDS = ["1874553", "1886347", "1899585", "1925299", "1927964", "1953632", "1959846",
             "1986691", "1996435", "1996436", "2006229", "2006363", "2007448", "2007721",
             "2010085", "2011166", "2013725", "2015213", "2016236", "2017461"]
MAX_PLAYERS = 20
SRC_FPS, FPS = 10, 25
_LFS = f"https://media.githubusercontent.com/media/SkillCorner/opendata/{SHA}/data/matches"
_RAW = f"https://raw.githubusercontent.com/SkillCorner/opendata/{SHA}/data/matches"
_FILES = {"tracking_extrapolated.jsonl": _LFS, "match.json": _RAW,
          "phases_of_play.csv": _RAW, "dynamic_events.csv": _RAW}


def raw_dir(match_id: str):
    return ext_dir() / "skillcorner" / "raw" / str(match_id)


def download(match_id: str) -> None:
    d = raw_dir(match_id)
    d.mkdir(parents=True, exist_ok=True)
    for suffix, base in _FILES.items():
        f = d / f"{match_id}_{suffix}"
        if not f.exists() or f.stat().st_size == 0:
            urllib.request.urlretrieve(f"{base}/{match_id}/{match_id}_{suffix}", f)


def _interp(src: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Linear interpolation along axis 0 at fractional source indices s (nearest if a side is NaN)."""
    i0 = np.floor(s).astype(int)
    i1 = np.minimum(i0 + 1, len(src) - 1)
    w = (s - i0).reshape((-1,) + (1,) * (src.ndim - 1))
    lin = src[i0] * (1 - w) + src[i1] * w
    near = src[np.rint(s).astype(int)]
    return np.where(np.isnan(lin), near, lin).astype(np.float32)


def _compact(xy: np.ndarray, det: np.ndarray):
    """Identity-free: move present players into the first slots, frame by frame."""
    present = ~np.isnan(xy[..., 0])
    order = np.argsort(~present, axis=1, kind="stable")[:, :MAX_PLAYERS]
    rows = np.arange(len(xy))[:, None]
    k = order.shape[1]
    out = np.full((len(xy), MAX_PLAYERS, 2), np.nan, np.float32)
    dout = np.zeros((len(xy), MAX_PLAYERS), bool)
    out[:, :k] = xy[rows, order]
    dout[:, :k] = det[rows, order]
    return out, dout


def convert(match_id: str) -> dict:
    d = raw_dir(match_id)
    meta = json.loads((d / f"{match_id}_match.json").read_text())
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    team_of = {p["id"]: p["team_id"] for p in meta["players"]}
    cols = {home_id: {}, away_id: {}}
    for p in meta["players"]:
        if p["team_id"] in cols:
            cols[p["team_id"]][p["id"]] = len(cols[p["team_id"]])

    frames = [json.loads(line) for line in open(d / f"{match_id}_tracking_extrapolated.jsonl")]
    nsrc = max(f["frame"] for f in frames) + 1
    period = np.zeros(nsrc, np.int8)
    ball = np.full((nsrc, 3), np.nan, np.float32)
    ball_det = np.zeros(nsrc, bool)
    group = np.full(nsrc, "D", "<U1")
    xy = {t: np.full((nsrc, len(c), 2), np.nan, np.float32) for t, c in cols.items()}
    det = {t: np.zeros((nsrc, len(c)), bool) for t, c in cols.items()}
    for f in frames:
        i = f["frame"]
        period[i] = f["period"] or 0
        b = f["ball_data"]
        if b["x"] is not None:
            ball[i] = (b["x"], b["y"], np.nan if b["z"] is None else b["z"])
            ball_det[i] = bool(b["is_detected"])
        g = f["possession"]["group"]
        group[i] = "H" if g == "home team" else "A" if g == "away team" else "D"
        for p in f["player_data"]:
            t = team_of.get(p["player_id"])
            if t in cols and p["x"] is not None:
                j = cols[t][p["player_id"]]
                xy[t][i, j] = (p["x"], p["y"])
                det[t][i, j] = bool(p["is_detected"])
    ball[:, 0] += length / 2
    ball[:, 1] += width / 2
    for t in xy:
        xy[t][..., 0] += length / 2
        xy[t][..., 1] += width / 2

    alive = np.zeros(nsrc, bool)
    with open(d / f"{match_id}_phases_of_play.csv") as fh:
        for row in csv.DictReader(fh):
            alive[int(row["frame_start"]):int(row["frame_end"])] = True
    label = np.where(alive, group, "D")

    parts = {k: [] for k in ("ball", "ball_det", "home", "home_det", "away", "away_det",
                             "label", "period")}
    for per in sorted(set(period.tolist()) - {0}):
        idx = np.flatnonzero(period == per)
        f0, f1 = idx[0], idx[-1]
        assert f1 - f0 + 1 == len(idx), "period frames not contiguous"
        t = np.arange(int(np.ceil(f0 * FPS / SRC_FPS)), int(np.floor(f1 * FPS / SRC_FPS)) + 1)
        s = t * SRC_FPS / FPS  # absolute fractional source frame
        near = np.rint(s).astype(int)
        parts["ball"].append(_interp(ball, s))
        parts["ball_det"].append(ball_det[near])
        for side, tid in (("home", home_id), ("away", away_id)):
            pos, dm = _compact(_interp(xy[tid], s), det[tid][near])
            parts[side].append(pos)
            parts[side + "_det"].append(dm)
        parts["label"].append(label[near])
        parts["period"].append(np.full(len(t), per, np.int8))
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    return {
        "ball": cat["ball"], "home": cat["home"], "away": cat["away"],
        "label": cat["label"].astype("<U1"), "period": cat["period"],
        "pitch": np.array([length, width], np.float32),
        "label_source": np.array("skillcorner_phase_alive_possession_group"),
        "ball_detected": cat["ball_det"], "home_detected": cat["home_det"],
        "away_detected": cat["away_det"],
        "fps": np.array(FPS), "source_fps": np.array(SRC_FPS),
    }


def path(match_id: str):
    return ext_dir() / "skillcorner" / f"{match_id}.npz"


def fetch_all(ids=MATCH_IDS) -> None:
    for mid in ids:
        p = path(mid)
        if p.exists():
            print(f"{mid}: cached")
            continue
        download(mid)
        d = convert(mid)
        np.savez_compressed(p, **d)
        lab, n = d["label"], len(d["label"])
        print(f"{mid}: {n} frames, H {np.mean(lab=='H'):.2f} A {np.mean(lab=='A'):.2f} "
              f"D {np.mean(lab=='D'):.2f}, ball missing {np.isnan(d['ball'][:,0]).mean():.3f} "
              f"(undetected incl. extrapolated {np.mean(~d['ball_detected']):.3f})")


def load(match_id: str) -> dict:
    with np.load(path(match_id), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    fetch_all()
