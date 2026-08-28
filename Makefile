VENV ?= .venv
PY   ?= $(VENV)/bin/python

.PHONY: help setup test eval collect merge preview inspect verify replay viewer lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## create the venv and install the package with dev + text extras
	uv venv --python 3.13
	uv pip install -e ".[dev,text]"

test:  ## run the unit and property test suite
	$(PY) -m pytest -q

eval:  ## 100-seed expert success evaluation on held-out seeds
	$(PY) -m so101_sim.cli.evaluate --num-seeds 100

collect:  ## generate a dataset (override N, SEED, RES, OUT)
	$(PY) -m so101_sim.cli.collect \
	  --num-episodes $(or $(N),100) \
	  --start-seed $(or $(SEED),0) \
	  --image-height $(or $(RES),128) --image-width $(or $(RES),128) \
	  --out $(or $(OUT),data/so101_lang_act.hdf5) \
	  --report $(or $(REPORT),data/collection_report.json)

merge:  ## merge dataset shards: make merge SHARDS="data/part_*.hdf5" OUT=data/train.hdf5
	$(PY) -m so101_sim.cli.merge $(SHARDS) --out $(or $(OUT),data/train.hdf5)

preview:  ## render one episode to mp4 (override SEED)
	$(PY) -m so101_sim.cli.visualize --seed $(or $(SEED),5)

inspect:  ## print the structure of a dataset (override OUT)
	$(PY) scripts/inspect_dataset.py $(or $(OUT),data/so101_lang_act.hdf5)

verify:  ## automated health checks on a dataset (reads disk, never re-simulates)
	$(PY) scripts/verify_dataset.py $(or $(OUT),data/so101_lang_act.hdf5)

replay:  ## render a stored episode to mp4 with overlays (override EP, OUT)
	$(PY) scripts/replay_dataset.py $(or $(OUT),data/so101_lang_act.hdf5) \
	  --episode $(or $(EP),0)

viewer:  ## interactive MuJoCo viewer for one seed (override SEED)
	$(PY) -m so101_sim.cli.viewer --seed $(or $(SEED),5)

lint:
	$(VENV)/bin/ruff check src tests scripts

clean:
	rm -rf data/*.hdf5 data/*.part*.hdf5 data/*.mp4 .pytest_cache .ruff_cache
