# VJ Intelligence: More than a Hack 2027, Phase 1

Frame-by-frame ball possession (H/A/D) for the More than a Hack 2027 challenge
(Metrica Sports, FC Barcelona Innovation Hub, Mobile World Capital Barcelona).

## Reproduce
```bash
git clone --recursive https://github.com/VJ-Intelligence/mtah2027-submission && cd mtah2027-submission
make setup                                   # Python 3.11 venv, pinned deps
export MTAH_DATA_DIR=/path/to/unpacked/bundle
make submission && make validate             # -> submissions/VJ Intelligence.csv
```

The match data is licensed for the competition only and is not in this repository.
The official challenge repository is included as a git submodule in `vendor/official` (MIT).

## Reproducibility
- **Inference is bit-exact.** `make submission` runs on CPU with deterministic algorithms and
  fixed thread counts, from the trained models committed in `models/`. Two runs on the same
  bundle produce byte-identical CSVs (checked by MD5).
- **The committed models are the reference.** `make train` rebuilds them from external data
  only (DFL/IDSSE, SkillCorner, SoccerTrack v2, PFF FC World Cup 2022). The four TCNs are trained
  on the Apple GPU (PyTorch MPS), which has no fully deterministic kernels, so a retrain gives
  models that differ in the last digits and can change a few frames of the output. Retraining
  on CPU (`DEVICE=cpu`) is deterministic but slower.
- No model was trained, fitted or tuned on the provided match.

Method, data and results: `docs/TECHNICAL.md`. Plan: `docs/PLAN.md`.
