# Plan (Phase 1, deadline Tue 17 Nov 2026 23:59 CET)

Final version 09.11., after reviews by Fable (x2), GPT-6 Astra, Gemini 3.8 and a local Qwen math pass.

## What decides the score
- spellF1 = mean over IoU {0.3..0.7} of 2TP / (N_gt + N_pred). A spell is a maximal H or A run.
- A correct extra spell gains roughly twice what a false one costs: insert a spell when its probability
  > ~F/2 (~0.3), unless it breaks a neighbouring spell. Decide per candidate (marginal utility).
- Allowed boundary shift for a spell of length L at IoU 0.7 is 0.18 x L (1 s spell: ~4 frames).
  Precision pays most for 0.5-4 s spells.
- Possession changes on CONTROLLED possession only. Touches, deflections, aerial duels without
  control keep the previous team. Loose ball = previous team.
- D only with real dead-ball evidence. Never D as an uncertainty hedge. X = "continue previous label".

## System
1. **Common data layer**: external matches (DFL/IDSSE 7 matches, SoccerTrack v2, SkillCorner open)
   converted to the challenge tracking format; per-frame H/A/D labels built with controlled-possession
   semantics from events + ball status.
2. **Bench**: leave-one-match-out on external matches with the official scorer, bootstrap intervals,
   results split by spell length.
3. **Degrader**: makes external tracking look like the challenge tracking (ball gaps correlated with
   aerial/crowded play and stoppages, smoothing variants centred/causal/interpolated, jitter, ball
   jumps, identity switches, duplicate/missing players).
4. **Features**: identity-free team aggregates + local interaction (nearest k players per team to
   the ball or candidate ball region: distance, relative velocity, approach angle, pressure).
5. **Dead-ball / restart engine**: onset at ball-out (trajectory + homography, robust to aerial
   balls), collective-motion change, restart execution detector near Smart Tagging set-piece anchors
   (offset measured), restarting team from geometry.
6. **Control-transfer detector**: contact candidates (ball velocity change + player-ball distance
   minimum, since player tracks are not smoothed), then "did the other team gain control?" using
   future evidence (next action by same team), flip backdated to the contact.
7. **Team-control model**: Conv1d + BiGRU (or GBDT) over features, as one specialist, latent loose
   states (H_loose / A_loose).
8. **Decoder**: Viterbi + snapping (safe) -> candidate-constrained semi-Markov with state-dependent
   duration priors -> local marginal-utility insert/delete (Dinkelbach for the ratio objective).
9. **Packaging**: bit-exact reproducible `make submission`, tech doc, repo public before upload.

All learnable parameters come from external data only. Public 5-min GT = smoke test only.

## Schedule
| Day | Work |
|---|---|
| Tue 10 | Step 1 (data layer, DFL first) + step 2 (bench) |
| Wed 11 | Step 3 (degrader) + step 4 (features) + simple baseline (GBDT/rules + Viterbi) on the bench |
| Thu 12 | Data arrives: reproduce official baselines exactly, first full-match file, validate |
| Fri 13 | Step 5 (dead-ball / restart engine) |
| **Sat 14** | **Safe submission uploaded** (best bench version so far) |
| Sat 14 - Sun 15 | Step 6 (control-transfer detector) + step 7 (sequence model) |
| Mon 16 | Step 8 decoder upgrades, pick final version on the bench, tech doc |
| Tue 17 | 10:00 freeze, clean-clone reproduction, make repo public, 14:00 upload |

Cut order if behind: 8 upgrades, then 7 (keep GBDT), never 5 or 6.

## Not doing in Phase 1
Video ball re-detection, V-JEPA 2 / DINOv3, JEPA/self-supervised pretraining, PFF data (licence),
broadcast-video fusion, anything trained, fitted or tuned on the provided match, checking output
against the video.

## Open organiser questions (sent 09.11.)
4 Nov clarification text, data link, Metrica sample + SkillCorner OK, tracking statistics for the
degrader, restart start (kick vs reception), AI coding assistant, technical parameters measured on the
provided data, D start definition, deflections, SoccerTrack v2.

## Progress (10 Nov, 04:15)
- Done: steps 1-4, 7, 9 and a basic 8 (Viterbi). Pipeline produces a valid, bit-identical CSV
  from the committed models (`make submission`), tested on a fake bundle in the same format.
- Final ensemble: TCN (DFL + SkillCorner) + TCN (DFL) + LightGBM (DFL), Viterbi 8/24.
  External LOMO (7 degraded DFL matches): **0.648 spellF1** (baseline approach 0.24).
- Out-of-domain check (SkillCorner, before mixing it into training): 0.35 vs baseline 0.20.
- Note: `make train` uses the Apple GPU and is not bit-exact; the committed models are the
  reference, inference on CPU is bit-exact. Decided 10 Nov: document it (README), no CPU retrain.
- Open: dead-ball / restart engine (step 5), decoder upgrade, Thursday data checks (team
  mapping via DataFormatSpecification vs <Player> order, official baselines reproduced exactly).
