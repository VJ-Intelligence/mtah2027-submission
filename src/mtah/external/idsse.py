"""DFL / IDSSE open data (Bassek et al. 2025, CC BY 4.0): 7 matches, 25 fps tracking + events.

Labels: DFL's own per-frame BallStatus + BallPossession -> H/A/D. That is DFL semantics, not yet
Metrica's "controlled possession"; refined later from events.
"""
import numpy as np

from ..paths import ext_dir

MATCH_IDS = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
MAX_PLAYERS = 20


def convert(match_id: str) -> dict:
    from kloppy import sportec

    ds = sportec.load_open_tracking_data(match_id)
    md = ds.metadata
    length, width = md.pitch_dimensions.pitch_length, md.pitch_dimensions.pitch_width
    home_team, away_team = md.teams[0], md.teams[1]
    df = ds.to_df()

    def team_block(team):
        ids = [p.player_id for p in team.players if f"{p.player_id}_x" in df.columns]
        out = np.full((len(df), MAX_PLAYERS, 2), np.nan, dtype=np.float32)
        xs = df[[f"{i}_x" for i in ids]].to_numpy(np.float32) * length
        ys = df[[f"{i}_y" for i in ids]].to_numpy(np.float32) * width
        # compact present players into the first slots, frame by frame (identity-free)
        present = ~np.isnan(xs)
        order = np.argsort(~present, axis=1, kind="stable")[:, :MAX_PLAYERS]
        rows = np.arange(len(df))[:, None]
        k = order.shape[1]
        out[:, :k, 0] = xs[rows, order]
        out[:, :k, 1] = ys[rows, order]
        return out

    ball = np.stack([df.ball_x.to_numpy(float) * length,
                     df.ball_y.to_numpy(float) * width,
                     df.ball_z.to_numpy(float)], axis=1).astype(np.float32)
    state = df.ball_state.astype(str).str.lower().to_numpy()
    owner = df.ball_owning_team_id.astype(str).to_numpy()
    label = np.where(state != "alive", "D",
                     np.where(owner == str(home_team.team_id), "H",
                              np.where(owner == str(away_team.team_id), "A", "D")))
    return {
        "ball": ball, "home": team_block(home_team), "away": team_block(away_team),
        "label": label.astype("<U1"), "period": df.period_id.to_numpy(np.int8),
        "pitch": np.array([length, width], np.float32),
        "label_source": np.array("dfl_ballstatus_ballpossession"),
    }


def path(match_id: str):
    return ext_dir() / "idsse" / f"{match_id}.npz"


def fetch_all(ids=MATCH_IDS) -> None:
    for mid in ids:
        p = path(mid)
        if p.exists():
            print(f"{mid}: cached")
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        d = convert(mid)
        np.savez_compressed(p, **d)
        lab, n = d["label"], len(d["label"])
        print(f"{mid}: {n} frames, H {np.mean(lab=='H'):.2f} A {np.mean(lab=='A'):.2f} "
              f"D {np.mean(lab=='D'):.2f}, ball missing {np.isnan(d['ball'][:,0]).mean():.3f}")


def load(match_id: str) -> dict:
    with np.load(path(match_id), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    fetch_all()


DELIBERATE = {"PASS", "SHOT"}


def action_frames(match_id: str, period: np.ndarray) -> dict:
    """Frames of deliberate on-ball actions per team ('H'/'A'), from DFL events (cached)."""
    import pandas as pd
    cache = ext_dir() / "idsse" / f"{match_id}_actions.parquet"
    if not cache.exists():
        from kloppy import sportec
        ds = sportec.load_open_event_data(match_id)
        home_id = ds.metadata.teams[0].team_id
        ev = ds.to_df()
        ev = ev[ev.event_type.isin(DELIBERATE) & ev.team_id.notna()]
        t = ev.timestamp.dt.total_seconds()
        pd.DataFrame({"period": ev.period_id.astype(int), "t": t,
                      "team": np.where(ev.team_id == home_id, "H", "A"),
                      "type": ev.event_type}).to_parquet(cache)
    a = pd.read_parquet(cache)
    a = a[a.t >= 0]
    out = {"H": [], "A": []}
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        s, n = idx[0], len(idx)
        sub = a[a.period == p]
        f = (sub.t * 25).round().astype(int).to_numpy() + s
        ok = (f >= s) & (f < s + n)
        for team in ("H", "A"):
            out[team].extend(f[ok & (sub.team.to_numpy() == team)].tolist())
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in out.items()}
