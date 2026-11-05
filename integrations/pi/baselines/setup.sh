#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
for binary in node npm docker; do command -v "$binary" >/dev/null; done
node --input-type=module -e 'const [major,minor]=process.versions.node.split(".").map(Number); if(major<22||(major===22&&minor<19))throw Error("Node >=22.19 is required")'
npm --prefix "$repo_root/integrations/pi" ci
npm --prefix "$repo_root/integrations/pi" run build
docker build --network "${CADGENBENCH_BUILD_NETWORK:-default}" \
  --file "$repo_root/integrations/pi/docker/Dockerfile" \
  --tag "${CAD_BASELINE_IMAGE:-cadgenbench-pi-agent:baseline-protocols-20261105-v1}" "$repo_root"
docker build --network "${CADGENBENCH_BUILD_NETWORK:-default}" \
  --build-arg "CAD_BASELINE_IMAGE=${CAD_BASELINE_IMAGE:-cadgenbench-pi-agent:baseline-protocols-20261105-v1}" \
  --file "$repo_root/integrations/pi/baselines/Dockerfile.evaluator" \
  --tag "${CAD_EVAL_IMAGE:-cadgenbench-pi-agent:baseline-evaluator-20261105-v1}" "$repo_root"
