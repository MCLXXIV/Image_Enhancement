PYTHON ?= python3
ENHANCER := services/enhancer
VENV := $(ENHANCER)/.venv
PY := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy

WEIGHTS_DIR := data/weights

.DEFAULT_GOAL := help

.PHONY: help install test lint fmt full-check up up-infra up-gpu down down-gpu logs logs-gpu clean weights data find-bars demo enhance-curl load-test

GPU_COMPOSE := docker compose -f docker-compose.yml -f docker-compose.gpu.yml

help:
	@echo "Image Enhancement - Makefile"
	@echo ""
	@echo "Dev (внутри services/enhancer/.venv):"
	@echo "  install      создать venv и поставить зависимости (-e '.[dev]')"
	@echo "  test         pytest -q"
	@echo "  lint         ruff check + ruff format --check + mypy"
	@echo "  fmt          ruff check --fix + ruff format (исправляет на месте)"
	@echo "  full-check   lint + test (= то же, что CI)"
	@echo ""
	@echo "Веса и данные:"
	@echo "  weights      скачать веса 3 моделей (SAFMN++/Retinexformer/SCUNet) в $(WEIGHTS_DIR)"
	@echo "  data         скачать Kaggle-датасет недвиги в data/"
	@echo "  find-bars    эвристикой найти фото с чёрными полосами -> CSV + HTML для ручной разметки screenshot"
	@echo ""
	@echo "Стек (docker compose):"
	@echo "  up           docker compose up -d --build (весь стек, CPU torch)"
	@echo "  up-infra     то же, но без enhancer/demo (если разрабатываешь сервис локально)"
	@echo "  up-gpu       стек с CUDA torch + nvidia runtime (для хоста с RTX)"
	@echo "  down         docker compose down"
	@echo "  down-gpu     то же для GPU-стека"
	@echo "  logs         docker compose logs -f enhancer"
	@echo "  logs-gpu     то же для GPU-стека"
	@echo "  demo         открыть Streamlit: http://localhost:8501"
	@echo "  enhance-curl smoke-тест POST /enhance из shell"
	@echo "  load-test    нагрузочный тест (TOTAL=50 CONCURRENCY=4 IMAGE=test.jpeg)"
	@echo ""
	@echo "  clean        вычистить __pycache__, .venv, .egg-info"

install:
	@if [ ! -d "$(VENV)" ]; then \
		echo "creating venv at $(VENV) using $(PYTHON)"; \
		cd $(ENHANCER) && $(PYTHON) -m venv .venv; \
	fi
	$(VENV)/bin/pip install --upgrade pip --quiet
	$(VENV)/bin/pip install -e '$(ENHANCER)[dev]'

test:
	cd $(ENHANCER) && .venv/bin/pytest -q

lint:
	cd $(ENHANCER) && .venv/bin/ruff check src tests
	cd $(ENHANCER) && .venv/bin/ruff format --check src tests
	cd $(ENHANCER) && .venv/bin/mypy

fmt:
	cd $(ENHANCER) && .venv/bin/ruff check --fix src tests
	cd $(ENHANCER) && .venv/bin/ruff format src tests

full-check: lint test

weights:
	$(PY) scripts/download_weights.py --output-dir $(WEIGHTS_DIR)

data:
	$(PY) scripts/download_kaggle_dataset.py

find-bars:
	$(PY) scripts/find_black_bars.py $(if $(IMAGES_DIR),--images-dir $(IMAGES_DIR),) $(if $(LIMIT),--limit $(LIMIT),)

up:
	docker compose up -d --build

up-infra:
	docker compose up -d minio postgres mlflow prometheus grafana

up-gpu:
	$(GPU_COMPOSE) up -d --build

down:
	docker compose down

down-gpu:
	$(GPU_COMPOSE) down

logs:
	docker compose logs -f enhancer

logs-gpu:
	$(GPU_COMPOSE) logs -f enhancer

demo:
	@echo "Streamlit: http://localhost:8501"
	@echo "Enhancer:  http://localhost:8000/healthz"
	@echo "Grafana:   http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "MLflow:    http://localhost:5000"

enhance-curl:
	@test -f sample.jpg || (echo "положи sample.jpg в корень репо" && exit 1)
	curl -s -o /tmp/enhanced.jpg -D /tmp/enhanced.headers \
		-F "image=@sample.jpg;type=image/jpeg" \
		http://localhost:8000/enhance
	@grep -i '^x-enhance' /tmp/enhanced.headers
	@echo "результат: /tmp/enhanced.jpg"

TOTAL ?= 50
CONCURRENCY ?= 4
IMAGE ?= test.jpeg
URL ?= http://localhost:8000/enhance

load-test:
	$(PYTHON) scripts/load_test.py \
		--url $(URL) --image $(IMAGE) \
		--total $(TOTAL) --concurrency $(CONCURRENCY) \
		$(if $(PARAMS),--params '$(PARAMS)',)

clean:
	find . -path ./.git -prune -o -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -path ./.git -prune -o -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(VENV)
