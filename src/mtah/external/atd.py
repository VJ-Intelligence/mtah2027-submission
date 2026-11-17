"""Metrica FIFA/EPTS automatic tracking (ATD) -> common format.

Used for the challenge tracking (via `games.asset_paths`) and, for format testing only, the
Metrica public sample game 3 (same EPTS format). Coordinates 0-1 -> metres with the given pitch.
Track slots are tracklets, not players; we only keep team membership (identity-free).
"""
import xml.etree.ElementTree as ET

import numpy as np


def _tag(e):
    return e.tag.split("}")[-1]


def specs(meta_path):
    """[(start_frame, end_frame, teams)] from the DataFormatSpecifications: for each spec the
    team (0 home, 1 away) of every player group, in DATA COLUMN order (not <Player> order,
    which can differ). Home = first <Team> (EPTS convention). Falls back to <Player> order."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter() if _tag(t) == "Team"]
    home = team_ids[0]
    team_of_player = {p.get("id"): (0 if p.get("teamId") == home else 1)
                      for p in root.iter() if _tag(p) == "Player"}
    chan_player = {c.get("id"): c.get("playerId") for c in root.iter() if _tag(c) == "PlayerChannel"}
    out = []
    for spec in (e for e in root.iter() if _tag(e) == "DataFormatSpecification"):
        order = []
        for r in spec.iter():
            if _tag(r) == "PlayerChannelRef":
                pid = chan_player.get(r.get("playerChannelId"))
                if pid and pid not in order:
                    order.append(pid)
        if order:
            out.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                        np.array([team_of_player[p] for p in order], np.int8)))
    if not out:
        teams = np.array([team_of_player[p.get("id")] for p in root.iter() if _tag(p) == "Player"], np.int8)
        out = [(0, 10 ** 9, teams)]
    return sorted(out, key=lambda x: x[0])


def slot_teams(meta_path) -> np.ndarray:
    """Team per column for the first spec (kept for compatibility / checks)."""
    return specs(meta_path)[0][2]


def convert(meta_path, raw_path, n_frames: int, pitch, period: np.ndarray, frame_base: int = 1,
            max_per_team: int = 30) -> dict:
    """Parse raw ATD, routing each player group to home/away by the spec valid for its frame."""
    sp = specs(meta_path)
    starts = np.array([s[0] for s in sp])
    L, W = float(pitch[0]), float(pitch[1])
    home = np.full((n_frames, max_per_team, 2), np.nan, np.float32)
    away = np.full((n_frames, max_per_team, 2), np.nan, np.float32)
    ball = np.full((n_frames, 3), np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            fid = int(parts[0])
            f = fid - frame_base
            if not 0 <= f < n_frames:
                continue
            k = max(0, int(np.searchsorted(starts, fid, side="right")) - 1)
            teams = sp[k][2]
            ih = ia = 0
            for j, g in enumerate(parts[1].split(";")):
                if j >= len(teams):
                    break
                v = g.split(",")
                if len(v) < 2 or not v[0] or v[0][:1] == "N":
                    if teams[j] == 0:
                        ih += 1
                    else:
                        ia += 1
                    continue
                x, y = float(v[0]) * L, float(v[1]) * W
                if teams[j] == 0 and ih < max_per_team:
                    home[f, ih] = x, y
                    ih += 1
                elif teams[j] == 1 and ia < max_per_team:
                    away[f, ia] = x, y
                    ia += 1
            v = parts[2].split(",")
            if len(v) >= 2 and v[0] and v[0][:1] != "N":
                ball[f, 0], ball[f, 1] = float(v[0]) * L, float(v[1]) * W
    return {"ball": ball, "home": home, "away": away, "period": period.astype(np.int8),
            "pitch": np.array([L, W], np.float32)}


def sample_game3() -> dict:
    """Metrica public sample game 3 in the common format (format test only, no labels)."""
    from ..paths import ext_dir
    d = ext_dir() / "metrica_sample"
    meta, raw = d / "Sample_Game_3_metadata.xml", d / "Sample_Game_3_tracking.txt"
    n = sum(1 for _ in open(raw))
    return convert(meta, raw, n + 1, (105.0, 68.0), np.ones(n + 1, np.int8))
