#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
image="ai-control-lyznet-faithful"
out_dir="${1:-paper_runs/faithful_related_work/lyznet_pendulum_seed123}"
shift || true
mode="${LYZNET_MODE:---full}"

case "$mode" in
  --full|--smoke|--preflight) ;;
  *)
    echo "error: LYZNET_MODE must be --full, --smoke, or --preflight" >&2
    exit 2
    ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  echo "error: Docker is required for the real dReal binding" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "error: Docker daemon is unavailable; start Docker Desktop/daemon first" >&2
  exit 1
fi

docker build --platform linux/amd64 \
  -f "$repo_root/faithful_related_work/lyznet/Dockerfile" \
  -t "$image" "$repo_root"
docker run --rm \
  --platform linux/amd64 \
  -v "$repo_root:/workspace" \
  "$image" \
  "$mode" --out-dir "/workspace/$out_dir" --seed 123 "$@"
