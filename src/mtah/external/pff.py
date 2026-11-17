"""PFF FC FIFA Men's World Cup 2022: broadcast tracking + event data, 29.97 fps, resampled to 25 fps.

Source: PFF FC open World Cup 2022 release (tracking spec v2.2, change log of 3 Jun 2025), laid
out under `root()` (default `<external dir>/wc2022`, outside the repo; override with MTAH_PFF_DIR):
  metadata/<id>.json      teams, fps, pitch size, period start/end in video seconds
  rosters/<id>.json       players per team (not needed for conversion)
  event/<id>.json         one row per possession event; game event fields repeated per row
  tracking/<id>.jsonl.bz2 one JSON object per video frame
Converted matches are written to `<root>/converted/<id>.npz` in the common format of
`mtah.external` (see its docstring), plus `act_H` / `act_A` like SoccerTrack.

Tracking: Sportlogiq broadcast tracking as delivered by PFF. The RAW positions are used
(`homePlayers`, `awayPlayers`, `balls`), not the `*Smoothed` ones: PFF builds the smoothed ball
by placing it on the event ball carrier and interpolating between events, which would leak
event timing into the input. Off-screen players and parts of the ball path are estimated by the
provider (`visibility` "ESTIMATED"); positions are kept as given and the masks `home_detected` /
`away_detected` / `ball_detected` say which were seen (same convention as SkillCorner, so the
caller can blank the estimated ones). Frames without a ball entry have a NaN ball. Frames
duplicated by PFF (one per extra possession event) are dropped by frameNum. Coordinates are
centre-origin metres (x along the touchlines); shifted by (L/2, W/2) to the corner origin.
Slots are filled per (team, jersey number) for interpolation, then compacted (identity-free).

Resampling 29.97 -> 25 fps, per period, on the video clock (videoTimeMs): the 25 fps grid starts
at the first tracking frame of the period (the kick-off) and ends at its last frame (the end
whistle). Positions are linearly interpolated between the two neighbouring source frames
(nearest when one side is NaN, via skillcorner._interp); detection masks take the nearest
source frame. Labels are computed directly on the 25 fps clock from event times.

Labels (label_source "pff_otb_team_in_live_status"), from the event file, per period:
  - dead   = [OUT event, start of the next team OTB event). PFF emits OUT both for the ball
    leaving the pitch and for the whistle (fouls, offside); every set-piece OTB (throw-in, free
    kick, goal kick, corner, kick-off, dropped ball) follows an OUT and starts at the moment the
    restart is taken (duration 0), so this is "dead until the restart is taken".
  - alive  = team (homeTeam) of the latest OTB or kick-off game event at or before the frame.
    OTB = "on the ball": a player has control. Between two OTBs (ball travelling) the previous
    team keeps the label, so a change happens when the other team gains control. Alive frames
    before the first team event of a stretch take the next team event's team.
  - OTB events without a team are ignored.
  - `act_H` / `act_A`: frames of deliberate actions = possession events PA (pass), CR (cross),
    SH (shot), BC (ball carry), excluding rows PFF flags as nonEvent. Use with
    labels.controlled() exactly like SoccerTrack (`action_frames`).
  - Timing: PFF event times trail the tracking by about 0.3 to 0.55 s around possession changes
    (`event_lag(m)`, stored as `event_lag_frames`, -7 to -14 on the five matches). `label` and
    `act_*` are NOT shifted; to align, use np.roll(label, lag) per match and bench it.
    Evidence against shifting (scripts/eval_pff.py): the committed DFL-trained ensemble agrees
    better with the unshifted labels (mean spellF1 0.475 vs 0.466 shifted), so the lag is
    more likely an artefact of the nearest-player proxy than a real clock offset.

Other stored keys: `video_time` (N,) source clock in seconds, `restart_after_out` (JSON: set
piece type of the restart following each OUT, a consistency check of the dead-ball rule).

Usage: PYTHONPATH=src python3 -m mtah.external.pff [id ...]   (prints aggregate stats only)
"""
import bz2
import json
import os
from array import array
from collections import Counter
from pathlib import Path

import numpy as np

from ..paths import ext_dir
from .skillcorner import _compact, _interp

FPS = 25
DELIBERATE = {"PA", "CR", "SH", "BC"}
TEAM_EVENTS = {"OTB", "FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF"}


def root():
    return Path(os.environ.get("MTAH_PFF_DIR", ext_dir() / "wc2022")).expanduser()


def match_ids() -> list:
    """Matches with metadata, events and tracking all present."""
    r = root()
    ids = [p.stem for p in (r / "metadata").glob("*.json")]
    return sorted(i for i in ids if (r / "event" / f"{i}.json").exists()
                  and (r / "tracking" / f"{i}.jsonl.bz2").exists())


def _metadata(mid: str) -> dict:
    m = json.loads((root() / "metadata" / f"{mid}.json").read_text())
    return m[0] if isinstance(m, list) else m


def _read_tracking(mid: str):
    """Stream the bz2 JSONL once -> per-frame arrays on the source clock (deduplicated).
    Players are placed in one column per (jersey, occurrence in frame) per team, so the same
    player keeps a column across frames for interpolation."""
    t, period, ball, ball_det = array("d"), array("b"), array("f"), array("b")
    pl = {s: {"cols": {}, "f": array("i"), "c": array("i"), "x": array("f"), "y": array("f"),
              "v": array("b")} for s in ("home", "away")}
    last, n = None, 0
    with bz2.open(root() / "tracking" / f"{mid}.jsonl.bz2", "rt") as fh:
        for line in fh:
            d = json.loads(line)
            fn = d["frameNum"]
            if fn == last or d.get("period") is None:
                continue
            last = fn
            t.append(d["videoTimeMs"] / 1000.0)
            period.append(int(d["period"]))
            b = d.get("balls") or []
            if b and b[0].get("x") is not None:
                z = b[0].get("z")
                ball.extend((b[0]["x"], b[0]["y"], np.nan if z is None else z))
                ball_det.append(b[0].get("visibility") == "VISIBLE")
            else:
                ball.extend((np.nan, np.nan, np.nan))
                ball_det.append(False)
            for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
                a, seen = pl[side], Counter()
                for p in d.get(key) or []:
                    if p.get("x") is None:
                        continue
                    j = str(p.get("jerseyNum"))
                    c = a["cols"].setdefault((j, seen[j]), len(a["cols"]))
                    seen[j] += 1
                    a["f"].append(n)
                    a["c"].append(c)
                    a["x"].append(p["x"])
                    a["y"].append(p["y"])
                    a["v"].append(p.get("visibility") == "VISIBLE")
            n += 1
    team = {}
    for side, a in pl.items():
        xy = np.full((n, max(1, len(a["cols"])), 2), np.nan, np.float32)
        det = np.zeros((n, max(1, len(a["cols"]))), bool)
        f, c = np.frombuffer(a["f"], np.int32), np.frombuffer(a["c"], np.int32)
        xy[f, c, 0] = np.frombuffer(a["x"], np.float32)
        xy[f, c, 1] = np.frombuffer(a["y"], np.float32)
        det[f, c] = np.frombuffer(a["v"], np.int8).astype(bool)
        team[side] = (xy, det)
    return (np.frombuffer(t, np.float64).copy(), np.frombuffer(period, np.int8).copy(),
            np.frombuffer(ball, np.float32).reshape(-1, 3).copy(),
            np.frombuffer(ball_det, np.int8).astype(bool), team)


def _events(mid: str):
    """Event rows -> (game events deduplicated by gameEventId and sorted, all rows)."""
    rows = json.loads((root() / "event" / f"{mid}.json").read_text())
    ge = {}
    for r in rows:
        ge.setdefault(r["gameEventId"], r)
    return sorted(ge.values(), key=lambda r: (r["startTime"], r["gameEventId"])), rows


def _labels(ge: list, per: int, t: np.ndarray) -> tuple:
    """H/A/D on the times t (seconds, video clock) of one period, plus event diagnostics."""
    g = [r for r in ge if r["gameEvents"]["period"] == per]
    typ = [r["gameEvents"]["gameEventType"] for r in g]
    anchors = [(r["startTime"], "H" if r["gameEvents"]["homeTeam"] else "A")
               for r, ty in zip(g, typ) if ty in TEAM_EVENTS and r["gameEvents"]["homeTeam"] is not None]
    dead = np.zeros(len(t), bool)
    after_out = Counter()
    for i, ty in enumerate(typ):
        if ty == "OUT":
            nxt = next((k for k in range(i + 1, len(g)) if typ[k] == "OTB"
                        and g[k]["gameEvents"]["homeTeam"] is not None), None)
            end = g[nxt]["startTime"] if nxt is not None else np.inf
            dead |= (t >= g[i]["startTime"]) & (t < end)
            after_out[g[nxt]["gameEvents"]["setpieceType"] if nxt is not None else "none"] += 1
        elif ty == "END":
            dead |= t >= g[i]["startTime"]
    lab = np.full(len(t), "D", "<U1")
    if anchors:
        at = np.array([a[0] for a in anchors])
        team = np.array([a[1] for a in anchors])
        k = np.searchsorted(at, t, side="right") - 1
        lab = np.where(k >= 0, team[np.maximum(k, 0)], team[0])  # before the first: next team
    lab = np.where(dead, "D", lab).astype("<U1")
    return lab, after_out


def event_lag(m: dict, lo: int = -25, hi: int = 5, near_m: float = 1.5) -> int:
    """Diagnostic: shift (frames) of the label that best matches the tracking around possession
    changes. Within 1 s of each H<->A change, on frames where the detected ball is within
    `near_m` of a player, count how often the labelled team is the nearer one, for each shift.
    Negative = event timestamps trail the tracking (np.roll(label, lag) aligns them).
    Measured on the five WC2022 matches: -7 to -14 frames. Not applied to `label`."""
    import warnings
    lab, b = m["label"], m["ball"][:, :2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hd = np.nanmin(np.linalg.norm(m["home"] - b[:, None], axis=2), axis=1)
        ad = np.nanmin(np.linalg.norm(m["away"] - b[:, None], axis=2), axis=1)
    near = np.where(hd < ad, "H", "A")
    ok = m["ball_detected"] & (np.fmin(hd, ad) < near_m)
    ch = np.flatnonzero((lab[1:] != lab[:-1]) & (lab[1:] != "D") & (lab[:-1] != "D")) + 1
    win = np.zeros(len(lab), bool)
    for c in ch:
        win[max(0, c - FPS):c + FPS] = True
    best, score = 0, -1.0
    for sh in range(lo, hi + 1):
        lb = np.roll(lab, sh)
        sel = ok & win & (lb != "D")
        a = float(np.mean(near[sel] == lb[sel])) if sel.any() else -1.0
        if a > score:
            best, score = sh, a
    return best


def convert(mid: str) -> dict:
    md = _metadata(mid)
    pitch = md["stadium"]["pitches"][0]
    length, width = float(pitch["length"]), float(pitch["width"])
    src_fps = float(md.get("fps") or 29.97)
    st, sper, sball, sball_det, team = _read_tracking(mid)
    ge, ev_rows = _events(mid)

    parts = {k: [] for k in ("ball", "ball_det", "home", "home_det", "away", "away_det",
                             "label", "period", "t")}
    after_out = Counter()
    for per in sorted(set(sper.tolist())):
        idx = np.flatnonzero(sper == per)
        assert idx[-1] - idx[0] + 1 == len(idx), "period frames not contiguous"
        ts = st[idx]
        assert np.all(np.diff(ts) > 0), "source clock not increasing"
        t = ts[0] + np.arange(int(np.floor((ts[-1] - ts[0]) * FPS)) + 1) / FPS
        s = idx[0] + np.interp(t, ts, np.arange(len(idx)))  # fractional source frame
        near = np.rint(s).astype(int)
        parts["ball"].append(_interp(sball, s))
        parts["ball_det"].append(sball_det[near])
        for side in ("home", "away"):
            xy, det = team[side]
            pos, dm = _compact(_interp(xy, s), det[near])
            parts[side].append(pos)
            parts[side + "_det"].append(dm)
        lab, ao = _labels(ge, per, t)
        after_out += ao
        parts["label"].append(lab)
        parts["period"].append(np.full(len(t), per, np.int8))
        parts["t"].append(t)
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    for k in ("ball", "home", "away"):
        cat[k][..., 0] += length / 2
        cat[k][..., 1] += width / 2

    # deliberate actions -> frame index on the 25 fps grid of their period
    off = np.r_[0, np.cumsum([np.sum(cat["period"] == p) for p in np.unique(cat["period"])])]
    t0 = {p: cat["t"][off[i]] for i, p in enumerate(np.unique(cat["period"]))}
    n_per = {p: off[i + 1] - off[i] for i, p in enumerate(np.unique(cat["period"]))}
    start = {p: off[i] for i, p in enumerate(np.unique(cat["period"]))}
    acts = {"H": [], "A": []}
    for r in ev_rows:
        pe, gev = r.get("possessionEvents") or {}, r["gameEvents"]
        if pe.get("possessionEventType") not in DELIBERATE or pe.get("nonEvent") or gev["homeTeam"] is None:
            continue
        p = gev["period"]
        if p not in t0:
            continue
        k = int(round((r["eventTime"] - t0[p]) * FPS))
        if 0 <= k < n_per[p]:
            acts["H" if gev["homeTeam"] else "A"].append(start[p] + k)
    acts = {k: np.unique(np.array(v, np.int64)) for k, v in acts.items()}

    out = {
        "ball": cat["ball"], "home": cat["home"], "away": cat["away"],
        "label": cat["label"], "period": cat["period"],
        "pitch": np.array([length, width], np.float32),
        "label_source": np.array("pff_otb_team_in_live_status"),
        "ball_detected": cat["ball_det"], "home_detected": cat["home_det"],
        "away_detected": cat["away_det"],
        "fps": np.array(FPS), "source_fps": np.array(src_fps),
        "video_time": cat["t"].astype(np.float64),
        "act_H": acts["H"], "act_A": acts["A"],
        "restart_after_out": np.array(json.dumps(dict(after_out))),
    }
    out["event_lag_frames"] = np.array(event_lag(out))
    return out


def path(mid: str):
    return root() / "converted" / f"{mid}.npz"


def fetch_all(ids=None, force: bool = False) -> None:
    from ..labels import controlled, spell_stats
    ids = ids or match_ids()
    for mid in ids:
        p = path(mid)
        if p.exists() and not force:
            print(f"{mid}: cached")
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        d = convert(mid)
        np.savez_compressed(p, **d)
        lab = d["label"]
        live = lab != "D"
        ctl = controlled(lab, d["period"], {"H": d["act_H"], "A": d["act_A"]}, 1.0)
        s = spell_stats(ctl, d["period"])
        print(f"{mid}: {len(lab)} frames, periods {np.unique(d['period']).tolist()}, "
              f"H {np.mean(lab == 'H'):.2f} A {np.mean(lab == 'A'):.2f} D {np.mean(lab == 'D'):.2f}, "
              f"ball missing {np.isnan(d['ball'][:, 0]).mean():.3f} (in play {np.isnan(d['ball'][live, 0]).mean():.3f}), "
              f"ball undetected incl. estimated {np.mean(~d['ball_detected']):.3f}, "
              f"players detected {d['home_detected'].sum(1).mean() + d['away_detected'].sum(1).mean():.1f}, "
              f"acts H {len(d['act_H'])} A {len(d['act_A'])}, restart after OUT {d['restart_after_out']}, "
              f"event lag {int(d['event_lag_frames'])} frames; "
              f"controlled: {s['per_live_min']:.1f} spells/live min, <2s {s['lt2']:.2f}, "
              f"median {s['median_s']:.1f}s", flush=True)


def load(mid: str) -> dict:
    with np.load(path(mid), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def action_frames(mid: str, period=None) -> dict:
    m = load(mid)
    return {"H": m["act_H"], "A": m["act_A"]}


if __name__ == "__main__":
    import sys
    fetch_all(sys.argv[1:] or None, force=bool(sys.argv[1:]))
