# Data lives OUTSIDE the repo. Point MTAH_DATA_DIR at the unpacked bundle.
MTAH_DATA_DIR ?= $(HOME)/Documents/Business/VJ Intelligence/INTERNAL/mtah2027-data
TEAM := VJ Intelligence
PY := .venv/bin/python
export MTAH_DATA_DIR
export PYTHONPATH := src

.PHONY: setup baselines train submission validate score clean

setup:            ## create venv (Python 3.11) and install pinned deps
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -r requirements.txt -r vendor/official/requirements.txt

baselines:        ## rebuild the official baselines (sanity check: must match published scores)
	cd vendor/official && ../../$(PY) starter/baseline.py

train:            ## train models on degraded EXTERNAL data -> models/
	$(PY) -m mtah.submit --train

submission:       ## data -> submissions/<team>.csv (prints only aggregate stats)
	$(PY) -m mtah.submit --out "submissions/$(TEAM).csv"

validate:         ## official validator
	$(PY) vendor/official/evaluation/validate_submission.py "submissions/$(TEAM).csv"

score:            ## official scorer on the PUBLIC 5 min (smoke test only, never for selection)
	rm -rf runs/score && mkdir -p runs/score/out
	cp vendor/official/evaluation/results/*.csv "submissions/$(TEAM).csv" runs/score/
	cd vendor/official && ../../$(PY) evaluation/run_evaluation.py --results ../../runs/score --out ../../runs/score/out --no-detail

clean:
	rm -rf runs submissions/*.csv
