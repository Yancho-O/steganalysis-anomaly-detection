.PHONY: install dev test lint demo clean build

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

demo: dev
	python -m scripts.make_toy_dataset --out data/toy --n 800 --size 256 --stego_rate 0.5 --embed_rate 0.15
	stegano-anomaly extract data/toy --out artifacts/features.csv --label-from-parent
	stegano-anomaly train artifacts/features.csv --model iforest --out artifacts/model.joblib

clean:
	rm -rf artifacts data/toy dist build *.egg-info

build: dev
	bash scripts/build_binary.sh
