VENV ?= .venv
PY   ?= $(VENV)/bin/python

.PHONY: help setup test eval collect preview inspect lint clean

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

collect:  ## generate a dataset (override N, WORKERS, OUT)
	$(PY) -m so101_sim.cli.collect \
	  --num-episodes $(or $(N),100) \
	  --num-workers $(or $(WORKERS),4) \
	  --out $(or $(OUT),data/so101_lang_act.hdf5) \
	  --report $(or $(REPORT),data/collection_report.json)

preview:  ## render one episode to mp4 (override SEED)
	$(PY) -m so101_sim.cli.visualize --seed $(or $(SEED),5)

inspect:  ## print the structure of a dataset (override OUT)
	$(PY) scripts/inspect_dataset.py $(or $(OUT),data/so101_lang_act.hdf5)

lint:
	$(VENV)/bin/ruff check src tests scripts

clean:
	rm -rf data/*.hdf5 data/*.part*.hdf5 data/*.mp4 .pytest_cache .ruff_cache
