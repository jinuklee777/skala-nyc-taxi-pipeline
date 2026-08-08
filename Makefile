.PHONY: setup run test clean-generated

setup:
	UV_CACHE_DIR=.uv-cache uv venv --python 3.11 .venv
	UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -r requirements.txt

run:
	.venv/bin/python run_pipeline.py

test:
	.venv/bin/python -m pytest -q

clean-generated:
	rm -rf artifacts figures reports data/processed

