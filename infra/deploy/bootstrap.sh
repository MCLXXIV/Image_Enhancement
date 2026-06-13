#!/usr/bin/env bash
# Подготовка свежего Ubuntu-хоста под image-enhancement stack.
# Устанавливает docker + compose plugin + nvidia-container-toolkit.

set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

echo "[1/4] apt update + базовые пакеты"
$SUDO apt-get update -y
$SUDO apt-get install -y ca-certificates curl gnupg lsb-release

echo "[2/4] Docker Engine + compose plugin"
if ! command -v docker >/dev/null 2>&1; then
  $SUDO install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    $SUDO gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  $SUDO chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu $(. /etc/os-release; echo "$VERSION_CODENAME") stable" | \
    $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
  $SUDO apt-get update -y
  $SUDO apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  $SUDO usermod -aG docker "$USER" || true
  echo "  перелогинься, чтобы группа docker применилась"
else
  echo "  docker уже установлен: $(docker --version)"
fi

echo "[3/4] NVIDIA Container Toolkit (для проброса GPU в контейнеры)"
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    $SUDO apt-get update -y
    $SUDO apt-get install -y nvidia-container-toolkit
    $SUDO nvidia-ctk runtime configure --runtime=docker
    $SUDO systemctl restart docker
  else
    echo "  nvidia-container-toolkit уже установлен"
  fi
else
  echo "  nvidia-smi не найден, пропускаем GPU toolkit (CPU-only режим)"
fi

echo "[4/4] Готово. Запуск стека:  infra/deploy/up.sh"
