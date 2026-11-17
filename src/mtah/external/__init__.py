"""Openly licensed external matches, converted to one common format.

Common format, one .npz per match (metres, origin at a pitch corner, 25 fps):
  ball     (N, 3)      x, y, z (z NaN if unknown); NaN when missing
  home     (N, P, 2)   player positions, NaN padded (identity-free use only)
  away     (N, P, 2)
  label    (N,)        'H' / 'A' / 'D' per frame (source possession semantics, see `label_source`)
  period   (N,)        1, 2, ...
  pitch    (2,)        length, width in metres
  label_source  ()     string naming the label rule

Optional keys (dataset specific, ignore if absent):
  ball_detected (N,), home_detected / away_detected (N, P)   seen vs extrapolated (SkillCorner)
  fps, source_fps                                            when the source was resampled

Datasets: idsse (DFL, 7 matches, CC BY 4.0), skillcorner (20 matches, MIT),
pff (PFF FC World Cup 2022, 64 matches, public release; raw + converted files live outside the
repo under <external dir>/wc2022 and are never committed).
SoccerTrack v2 (CC BY 4.0, Hugging Face `atomscott/soccertrack-v2`) is not ingested: the
dataset is gated (Hugging Face login + accepting the terms) and the pitch-coordinate GSR files
are ~2.7 GB per half (~56 GB total).
"""
