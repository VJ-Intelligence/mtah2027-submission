"""Automatic checks on the challenge data. Prints AGGREGATE numbers only (no rows, no frames).

    python -m mtah.checks

1. Team mapping: our parser routes data columns to teams via the DataFormatSpecifications
   (column order per frame range). The official baseline uses <Player> order. Compare both:
   how many column slots get a different team, and whether each mapping gives two compact
   teams on opposite halves of the pitch that swap ends at half time.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

from .external import atd
from .paths import official_on_path


def _player_order_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter() if atd._tag(t) == "Team"]
    return np.array([0 if p.get("teamId") == team_ids[0] else 1
                     for p in root.iter() if atd._tag(p) == "Player"], np.int8)


def team_mapping():
    official_on_path()
    import games
    paths = games.asset_paths(data_dir=os.environ.get("MTAH_DATA_DIR"))
    sp = atd.specs(paths["atd_meta"])
    po = _player_order_teams(paths["atd_meta"])
    print(f"DataFormatSpecifications: {len(sp)}; <Player> entries: {len(po)}")
    for k, (a, b, t) in enumerate(sp):
        n = min(len(t), len(po))
        diff = int((t[:n] != po[:n]).sum())
        print(f"  spec {k}: {len(t)} columns (home {int((t == 0).sum())}, away {int((t == 1).sum())}); "
              f"slots with a different team than <Player> order: {diff}")

    n = games.source_frames()
    period = np.zeros(n, np.int8)
    for half, a, b in games.half_video_blocks(games.GAME):
        period[a:b + 1] = half
    m = atd.convert(paths["atd_meta"], paths["atd_raw"], n, games.GAMES[games.GAME]["pitch"], period)
    L = float(games.GAMES[games.GAME]["pitch"][0])
    print("ours (spec order): per half, mean x of each team / spread (std of x within team, m)")
    for h in (1, 2):
        idx = period == h
        hx = np.nanmean(m["home"][idx, :, 0], axis=1)
        ax = np.nanmean(m["away"][idx, :, 0], axis=1)
        hs = np.nanmean(np.nanstd(m["home"][idx, :, 0], axis=1))
        as_ = np.nanmean(np.nanstd(m["away"][idx, :, 0], axis=1))
        side = np.nanmean(hx < ax)
        print(f"  half {h}: home mean x {np.nanmean(hx):.1f}, away {np.nanmean(ax):.1f} (pitch {L:.0f}); "
              f"home left of away on {side:.2f} of frames; spread home {hs:.1f} away {as_:.1f}")




def public_confusion(sub_path=None):
    """Counts of (public GT label, our label) over the graded public frames. Aggregates only."""
    import pandas as pd
    from .paths import OFFICIAL, REPO
    gt = pd.read_csv(OFFICIAL / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv")
    sub = pd.read_csv(sub_path or REPO / "submissions" / "VJ Intelligence.csv")
    col = [c for c in gt.columns if c != gt.columns[0]][0]
    g, s = gt[col].to_numpy(), sub[[c for c in sub.columns if c != sub.columns[0]][0]].to_numpy()
    graded = g != "X"
    print(f"graded frames {graded.sum()}")
    print("rows = public GT, cols = ours:     H      A      D      X")
    for a in "HAD":
        print(f"  {a}  " + " ".join(f"{int(((g == a) & (s == b) & graded).sum()):6d}" for b in "HADX"))
    agree = (g == s)[graded].mean()
    swapped = np.where(s == "H", "A", np.where(s == "A", "H", s))
    print(f"frame agreement {agree:.3f}; with H/A swapped {(g == swapped)[graded].mean():.3f}")


def feature_shift():
    """Per-feature mean/NaN share on the challenge match vs a degraded DFL match. Aggregates only."""
    from . import features, model
    from .submit import challenge_match
    m = challenge_match()
    live = m["period"] > 0
    Xc, names = features.build(m)
    Xc = Xc[live]
    from .external import idsse
    copies = [model.sample(mid, 100 * i + 99)[0] for i, mid in enumerate(idsse.MATCH_IDS)]
    rows = []
    fin = lambda a: np.where(np.isfinite(a), a, np.nan)
    for j, nm in enumerate(names):
        c = Xc[:, j]
        cm = np.nanmean(fin(c))
        means = np.array([np.nanmean(fin(X[:, j])) for X in copies])
        dm, sd = means.mean(), means.std() + 1e-3 * (abs(means.mean()) + 1e-3)
        # z relative to the spread of per-match means across the 7 degraded DFL matches
        rows.append((abs(cm - dm) / sd, nm, cm, dm, np.mean(~np.isfinite(c)),
                     np.mean([np.mean(~np.isfinite(X[:, j])) for X in copies])))
    rows.sort(reverse=True)
    print("feature        z vs spread of 7 DFL match means   match mean   DFL mean   match NaN   DFL NaN")
    for z, nm, cm, dm, cn, dn in rows[:20]:
        print(f"{nm:30s} {z:8.2f} {cm:12.3f} {dm:10.3f} {cn:10.3f} {dn:9.3f}")
    print("ball:", f"present {np.mean(np.isfinite(m['ball'][live, 0])):.3f}",
          f"x range {np.nanmin(m['ball'][live, 0]):.1f}..{np.nanmax(m['ball'][live, 0]):.1f}",
          f"y range {np.nanmin(m['ball'][live, 1]):.1f}..{np.nanmax(m['ball'][live, 1]):.1f}")
    for side in ("home", "away"):
        n = np.sum(np.isfinite(m[side][live, :, 0]), axis=1)
        print(f"{side}: players per frame mean {n.mean():.1f}, frames with 0: {np.mean(n == 0):.3f}")


def baseline_parity():
    """Our parser + our baseline re-implementation vs the official baseline CSVs (no GT involved).
    Frame-level agreement per window; anything below ~0.99 means we read the tracking differently."""
    import pandas as pd
    from . import baselines
    from .paths import OFFICIAL
    from .submit import challenge_match
    m = challenge_match()
    for w, name in ((0, "raw"), (25, "smooth25"), (125, "smooth125")):
        ours = baselines.predict(m, w)
        off = pd.read_csv(OFFICIAL / "evaluation" / "results" / f"baseline_{name}.csv")["label"].to_numpy()
        inside = off != "X"
        print(f"window {w:3d}: agreement on non-X frames {np.mean(ours[inside] == off[inside]):.4f}; "
              f"official X frames {int((~inside).sum())}")
    for side in ("home", "away"):
        print(f"{side}: columns {m[side].shape[1]}")


def pred_stats(sub_path=None, probas="runs/v2_tcn_sc.npz"):
    """GT-free: spell statistics of our prediction on the match vs our LOMO predictions and the
    GT on external matches. Aggregates only."""
    import pandas as pd
    from .decode import decode
    from .external import idsse
    from .labels import spell_stats
    from .paths import REPO
    from . import model
    sub = pd.read_csv(sub_path or REPO / "submissions" / "VJ Intelligence.csv")["label"].to_numpy()
    graded = sub != "X"
    starts = graded & ~np.r_[False, graded[:-1]]          # each graded block = one half
    per = (np.cumsum(starts) * graded).astype(np.int8)
    lab = sub[graded]
    def show(name, st):
        print(f"{name:28s} spells/live-min {st['per_live_min']:.2f}  median {st['median_s']:.1f}s  "
              f"<2s {st['lt2']:.2f}  <4s {st['lt4']:.2f}  dead share {st['dead_share']:.2f}")
    show("MATCH prediction", spell_stats(lab, per[graded]))
    P = np.load(REPO / probas)
    for mid in idsse.MATCH_IDS[:4]:
        i = idsse.MATCH_IDS.index(mid)
        _, y, _, pr = model.sample(mid, 100 * i + 99)
        show(f"DFL {mid} GT", spell_stats(np.array(list("DHA"))[y], pr))
        show(f"DFL {mid} prediction", spell_stats(decode(P[mid][:, :3], pr, 8, 24), pr))


if __name__ == "__main__":
    import sys
    {"team": team_mapping, "public": public_confusion, "shift": feature_shift, "parity": baseline_parity, "pred": pred_stats}[sys.argv[1] if len(sys.argv) > 1 else "team"]()
