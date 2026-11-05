#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
command -v node >/dev/null
command -v npm >/dev/null
command -v docker >/dev/null
node --input-type=module -e 'const [a,b]=process.versions.node.split(".").map(Number); if(a<22||(a===22&&b<19))throw Error("Node >=22.19 is required")'
docker version --format '{{.Server.Version}}'
npm --prefix "$repo_root/integrations/pi" ci
npm --prefix "$repo_root/integrations/pi" run build
image="${CAD_FULL_METHOD_IMAGE:-cadgenbench-pi-agent:full-method-gpt56-low-v1}"
docker build --file "$repo_root/integrations/pi/docker/Dockerfile" --tag "$image" "$repo_root"
printf '\nReady: %s\n' "$image"
printf 'Configure credentials, then run node integrations/pi/portable-full-method/run.mjs --doctor\n'
