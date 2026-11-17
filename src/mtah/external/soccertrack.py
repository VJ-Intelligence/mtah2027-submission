"""SoccerTrack v2: 10 university matches, fixed panoramic camera, 25 fps (CC BY 4.0).

Source: Hugging Face `atomscott/soccertrack-v2`, release v1.2. Please cite the SoccerTrack v2
paper. Raw game-state JSON (~2.7 GB per half) is converted once by
`scripts/convert_soccertrack.py` into `soccertrack-v2/compact/<match>_<half>.npz`; this module
reads those plus `ball/` and `bas/`.

Tracking: players and goalkeepers in pitch metres (centre origin, 105 x 68), rounded by the
provider to a 1.05 m x 0.68 m grid. Shifted by (52.5, 34) to the corner origin.
`team` in the source is the pitch SIDE (left/right) and swaps at half time; teams are made
consistent across halves via persistent player ids (home = the side that is left in half 1).

Ball: NOT measured per frame. The provider anchors it at annotated ball events and moves it in
straight constant-speed lines between anchors, so the raw track carries exact event timing.
Use only through the degrader (gaps + smoothing) and judge it on the bench.

Labels (label_source "soccertrack_events_in_live_status"): possession = team of the latest
possession event (passes, drives, crosses, shots, restarts, successful tackles, headers; not
OUT/GOAL/blocks), only where the ball is alive per the provider feed. The feed alone is kept as
`label_feed`; it is much coarser (about 4 spells per live minute, 7% under 2 s, against 7.4 and
26% for DFL), so it is used for alive/dead only. The ball file's undocumented `status` field
reads as a possession feed (checked against the events: every THROW IN and FREE KICK has 0,
most OUT events have 0, blocks mostly 3):
  0 / -1 -> 'D' (dead / no record); 1 / 2 -> the team whose events dominate that code in the
  half; 3 (loose / contested) -> previous team inside the live stretch (next team if the
  stretch starts loose) [this gives label_feed]. The controlled-possession rule (labels.controlled) is applied on top
  by the caller, with deliberate actions from `action_frames`.
"""
import json
from collections import Counter

import numpy as np

from ..paths import ext_dir

MATCH_IDS = ["117092", "117093", "118575", "118576", "118577",
             "118578", "128057", "128058", "132831", "132877"]
# 132831 (M9) and 132877 (M10) carry no possession feed (status 4 on every frame) and a
# pitch-clamped ball: not used for training.
TRAIN_IDS = MATCH_IDS[:8]
HALVES = ("1st", "2nd")
LENGTH, WIDTH = 105.0, 68.0
MAX_PLAYERS = 14
DELIBERATE = {"PASS", "HIGH PASS", "CROSS", "SHOT", "DRIVE", "THROW IN", "FREE KICK"}
POSSESSION_EVENTS = DELIBERATE | {"PLAYER SUCCESSFUL TACKLE", "HEADER"}


def root():
    return ext_dir() / "soccertrack-v2"


def _events(mid):
    return json.loads((root() / "bas" / mid / f"{mid}_12_class_events.json").read_text())["actions"]


def _pack(frame, xy, n):
    """Per-frame rows -> (n, MAX_PLAYERS, 2), NaN padded, slot order arbitrary (identity-free)."""
    out = np.full((n, MAX_PLAYERS, 2), np.nan, np.float32)
    order = np.argsort(frame, kind="stable")
    f = frame[order]
    starts = np.r_[0, np.flatnonzero(np.diff(f)) + 1]
    slot = np.arange(len(f)) - np.repeat(starts, np.diff(np.r_[starts, len(f)]))
    keep = slot < MAX_PLAYERS
    out[f[keep] - 1, slot[keep]] = xy[order][keep]
    return out, int((~keep).sum())


def _fill_loose(lab: np.ndarray, loose: np.ndarray) -> np.ndarray:
    """Loose frames take the previous team (H/A) of their stretch; if the stretch is preceded
    by a dead ball (or nothing), they take the next team; if neither side is a team -> 'D'."""
    out = lab.copy()
    known = np.flatnonzero(~loose)
    if len(known) == 0:
        return out
    i = np.flatnonzero(loose)
    k = np.searchsorted(known, i)                    # first known index > i
    prev = np.where(k > 0, lab[known[np.maximum(k - 1, 0)]], "D")
    nxt = np.where(k < len(known), lab[known[np.minimum(k, len(known) - 1)]], "D")
    out[i] = np.where(prev != "D", prev, nxt)
    return out


def convert(mid: str) -> dict:
    halves = []
    home_ids = None
    ev = _events(mid)
    for h_i, h in enumerate(HALVES, start=1):
        z = np.load(root() / "compact" / f"{mid}_{h}.npz")
        b = np.load(root() / "ball" / f"{mid}_{h}_ball.npz")
        n = len(b["frame"])
        assert (b["frame"] == np.arange(1, n + 1)).all()
        side, pid = z["team"], z["player_id"]
        if home_ids is None:
            home_ids = set(pid[side == 0].tolist())
        # which side is home in this half (by persistent player ids)
        frac_left_home = np.mean([p in home_ids for p in np.unique(pid[side == 0])])
        home_side = 0 if frac_left_home > 0.5 else 1
        xy = z["pitch_xy"] + np.array([LENGTH / 2, WIDTH / 2], np.float32)
        home, dh = _pack(z["frame"][side == home_side], xy[side == home_side], n)
        away, da = _pack(z["frame"][side == 1 - home_side], xy[side == 1 - home_side], n)

        ball = np.full((n, 3), np.nan, np.float32)
        ball[:, 0] = b["x"] + LENGTH / 2
        ball[:, 1] = b["y"] + WIDTH / 2
        st = b["status"].astype(int)
        ball[st == -1] = np.nan

        # map status codes 1/2 to sides via the events of this half
        he = [e for e in ev if e["gameTime"].startswith(f"{h_i} ")]
        votes = {1: Counter(), 2: Counter()}
        for e in he:
            s = int(st[e["frame"] - 1])
            if s in votes:
                votes[s][e["team"]] += 1
        side_name = {0: "left", 1: "right"}
        home_name = side_name[home_side]
        code_home = 1 if votes[1][home_name] - votes[1].get("left" if home_name == "right" else "right", 0) \
            >= votes[2][home_name] - votes[2].get("left" if home_name == "right" else "right", 0) else 2
        agree = (votes[code_home][home_name] + votes[3 - code_home][side_name[1 - home_side]]) / \
            max(1, sum(votes[1].values()) + sum(votes[2].values()))
        lab = np.full(n, "D", "<U1")
        lab[st == code_home] = "H"
        lab[st == 3 - code_home] = "A"
        lab_feed = _fill_loose(lab, st == 3)
        # event-based possession (denser than the feed): team of the latest possession event,
        # only while the feed says the ball is alive
        pe = sorted((e["frame"] - 1, "H" if e["team"] == home_name else "A")
                    for e in he if e["label"] in POSSESSION_EVENTS)
        fr = np.array([q[0] for q in pe], np.int64)
        tm = np.array([q[1] for q in pe])
        k = np.searchsorted(fr, np.arange(n), side="right") - 1
        evlab = np.where(k >= 0, tm[np.maximum(k, 0)], "D")
        lab = np.where(lab_feed != "D", evlab, "D").astype("<U1")
        acts = {"H": [], "A": []}
        for e in he:
            if e["label"] in DELIBERATE:
                t = "H" if e["team"] == home_name else "A"
                acts[t].append(e["frame"] - 1)
        halves.append(dict(home=home, away=away, ball=ball, label=lab, label_feed=lab_feed, n=n,
                           acts=acts, agree=agree, dropped=dh + da))
    off = np.cumsum([0] + [hh["n"] for hh in halves])
    act = {t: np.sort(np.concatenate([np.array(hh["acts"][t], np.int64) + off[k]
                                      for k, hh in enumerate(halves)])) for t in ("H", "A")}
    return {
        "ball": np.concatenate([hh["ball"] for hh in halves]),
        "home": np.concatenate([hh["home"] for hh in halves]),
        "away": np.concatenate([hh["away"] for hh in halves]),
        "label": np.concatenate([hh["label"] for hh in halves]),
        "label_feed": np.concatenate([hh["label_feed"] for hh in halves]),
        "period": np.concatenate([np.full(hh["n"], k + 1, np.int8) for k, hh in enumerate(halves)]),
        "pitch": np.array([LENGTH, WIDTH], np.float32),
        "label_source": np.array("soccertrack_events_in_live_status"),
        "act_H": act["H"], "act_A": act["A"],
        "status_team_agreement": np.array([hh["agree"] for hh in halves], np.float32),
        "players_dropped": np.array(sum(hh["dropped"] for hh in halves)),
    }


def path(mid: str):
    return root() / f"{mid}.npz"


def fetch_all(ids=MATCH_IDS) -> None:
    for mid in ids:
        d = convert(mid)
        np.savez_compressed(path(mid), **d)
        lab = d["label"]
        print(f"{mid}: {len(lab)} frames, H {np.mean(lab == 'H'):.2f} A {np.mean(lab == 'A'):.2f} "
              f"D {np.mean(lab == 'D'):.2f}, ball missing {np.isnan(d['ball'][:, 0]).mean():.3f}, "
              f"status/event team agreement {d['status_team_agreement'].round(3).tolist()}, "
              f"acts H {len(d['act_H'])} A {len(d['act_A'])}, players dropped {int(d['players_dropped'])}")


def load(mid: str) -> dict:
    with np.load(path(mid), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def action_frames(mid: str, period=None) -> dict:
    m = load(mid)
    return {"H": m["act_H"], "A": m["act_A"]}


if __name__ == "__main__":
    fetch_all()
