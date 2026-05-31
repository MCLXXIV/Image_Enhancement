PYTHON ?= python3
ENHANCER := services/enhancer
VENV := $(ENHANCER)/.venv
PY := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy

WEIGHTS_DIR := .volumes/weights
SAFMN_WEIGHTS := $(WEIGHTS_DIR)/Real_SAFMNpp_x4.pth
SAFMN_WEIGHTS_URL := https://huggingface.co/Meloo/SAFMN/resolve/main/Real_SAFMNpp_x4.pth

.DEFAULT_GOAL := help

.PHONY: help install test lint fmt full-check up up-infra down logs clean weights demo enhance-curl

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
	@echo "Веса SAFMN:"
	@echo "  weights      скачать Real_SAFMN++ x4 в $(SAFMN_WEIGHTS)"
	@echo ""
	@echo "Стек (docker compose):"
	@echo "  up           docker compose up -d --build (весь стек)"
	@echo "  up-infra     то же, но без enhancer/demo (если разрабатываешь сервис локально)"
	@echo "  down         docker compose down"
	@echo "  logs         docker compose logs -f enhancer"
	@echo "  demo         открыть Streamlit: http://localhost:8501"
	@echo "  enhance-curl smoke-тест POST /enhance из shell"
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
	@mkdir -p $(WEIGHTS_DIR)
	@if [ -f "$(SAFMN_WEIGHTS)" ]; then \
		echo "weights already at $(SAFMN_WEIGHTS), skipping"; \
	else \
		echo "downloading Real_SAFMN++ x4 from HuggingFace Meloo/SAFMN..."; \
		curl -L -o $(SAFMN_WEIGHTS) $(SAFMN_WEIGHTS_URL); \
	fi

up:
	docker compose up -d --build

up-infra:
	docker compose up -d minio postgres mlflow prometheus grafana

down:
	docker compose down

logs:
	docker compose logs -f enhancer

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

clean:
	find . -path ./.git -prune -o -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -path ./.git -prune -o -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(VENV)
