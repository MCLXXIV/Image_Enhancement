PYTHON ?= python3
ENHANCER := services/enhancer
VENV := $(ENHANCER)/.venv
PY := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy

.DEFAULT_GOAL := help

.PHONY: help install test lint fmt full-check up up-infra down logs clean

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
	@echo "Стек (docker compose):"
	@echo "  up           docker compose up -d --build (весь стек)"
	@echo "  up-infra     то же, но без enhancer/demo (если разрабатываешь сервис локально)"
	@echo "  down         docker compose down"
	@echo "  logs         docker compose logs -f enhancer (наш способ смотреть логи)"
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

up:
	docker compose up -d --build

up-infra:
	docker compose up -d minio postgres mlflow prometheus grafana

down:
	docker compose down

logs:
	docker compose logs -f enhancer

clean:
	find . -path ./.git -prune -o -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -path ./.git -prune -o -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(VENV)
