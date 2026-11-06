#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CAD_BASELINE_IMAGE="${CAD_FULL_METHOD_IMAGE:-${CAD_BASELINE_IMAGE:-cadgenbench-pi-agent:round-comparison-20261106-v1}}"
bash "$script_dir/../baselines/setup.sh"
