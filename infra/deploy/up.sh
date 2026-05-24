#!/usr/bin/env bash
# Поднять весь стек на хосте. Запускать из корня репо или указать REPO_DIR.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_DIR"

ENV_FILE="${ENV_FILE:-.env.prod}"
if [[ -f "$ENV_FILE" ]]; then
  echo "loading env from $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

docker compose pull --ignore-pull-failures || true
docker compose up -d --build --remove-orphans

docker compose ps
