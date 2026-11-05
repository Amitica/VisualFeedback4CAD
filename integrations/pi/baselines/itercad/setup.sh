#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
upstream="$repo_root/external_baselines/itercad"
commit=55998c4a4537573f67ca241dedc52ee2840652ca
if [[ ! -d "$upstream" ]]; then
  mkdir -p "$repo_root/external_baselines"
  git clone --no-checkout https://github.com/KnowledgeXLab/IterCAD.git "$upstream"
  git -C "$upstream" checkout --detach "$commit"
fi
if [[ "$(git -C "$upstream" rev-parse HEAD)" != "$commit" ]]; then
  printf 'Existing IterCAD checkout differs from pinned commit; preserving it: %s\n' "$upstream" >&2
  exit 2
fi
"${PYTHON:-python3}" "$repo_root/integrations/pi/baselines/itercad/native.py" source-check --upstream-root "$upstream"
docker build --network "${CADGENBENCH_BUILD_NETWORK:-default}" \
  --file "$repo_root/integrations/pi/baselines/itercad/Dockerfile" \
  --tag "${ITERCAD_IMAGE:-cadgenbench-itercad-native:55998c4}" "$repo_root"
printf 'Built IterCAD native image. Run its doctor before inference.\n'
