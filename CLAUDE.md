# mtah2027: rules for AI assistants (HARD)

- **Never read, print, plot, screenshot or summarise match data.** The data is licensed for the
  competition only and must not reach any hosted AI service (incl. this session). It lives OUTSIDE
  the repo in `$MTAH_DATA_DIR` and is blocked by `.claude/hooks/data-airlock.sh`.
- Develop and debug on external open data (Metrica sample, IDSSE/DFL, SkillCorner open) which has
  the same formats.
- Pipeline commands (`make baselines|submission|validate|score`, `python -m mtah.*`) may touch the
  data but must print **aggregate numbers only** (scores, counts, durations). Never add code that
  prints rows, frames, coordinates or labels of the match.
- Rules (incl. 4 Nov clarification, unverified until organisers confirm):
  - ALL training, fitting, tuning, calibration (incl. unsupervised/self-supervised, EM, clustering,
    pseudo-label self-training) uses EXTERNAL data only. Never on the provided match, GT or video.
  - Models trained elsewhere may take the match video/tracking/tagging as INPUT at inference time.
  - Local models only (no data to hosted AI); output must be bit-exact reproducible.
  - No hand labels on the provided match. No checking our output against the video.
  - Public 5-min GT = pipeline smoke test only.
- Commits: Valerio is sole author, no AI co-author trailers.
- No em dashes in docs.

## Organiser rules (clarification of 4 Nov 2026, received 10 Nov)
Not allowed: checking output against the video; hand labels or any human-made data on the
provided game; uploading provided data to AI services; training, fitting or calibrating on the
ground-truth possession or on the provided video. Allowed: automatic checks using only the
provided data; inspecting data and video to understand what counts as possession (humans only,
never to set parameters); using video, tracking, tagging as input; intermediate files; any
dataset we have the rights to. Training and tuning on external data only. Anything not
explicitly allowed is not permitted.
