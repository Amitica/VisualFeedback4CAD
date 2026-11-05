#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 || $# > 3 )); then
  printf 'Usage: bash %s <baseline-output-or-evaluation-dir> <GT-fixtures-dir> [workers=8]\n' "${BASH_SOURCE[0]}" >&2
  exit 64
fi
eval_root="$(realpath -- "$1")"
gt_root="$(realpath -- "$2")"
workers="${3:-8}"
[[ "$workers" =~ ^[1-9][0-9]*$ ]] && (( workers <= 64 )) || { echo 'workers must be in [1,64]' >&2; exit 64; }
if [[ -d "$eval_root/evaluation" ]]; then eval_root="$eval_root/evaluation";
elif [[ -d "$eval_root/cases" ]]; then eval_root="$eval_root/cases"; fi
[[ -d "$eval_root" && -d "$gt_root" ]] || { echo 'Evaluation and GT directories must exist' >&2; exit 64; }
[[ "$eval_root" != *,* && "$gt_root" != *,* ]] || { echo 'Docker mount paths cannot contain commas' >&2; exit 64; }
docker run --rm --read-only --network none --cap-drop ALL \
  --security-opt no-new-privileges --tmpfs /tmp:rw,nosuid,nodev,size=2g,mode=1777 \
  --user "$(id -u):$(id -g)" --cpus "$workers" --memory "${CAD_EVAL_MEMORY:-16g}" --pids-limit 512 \
  --env CADGENBENCH_DATA_DIR=/benchmark-data \
  --mount "type=bind,src=$eval_root,dst=/evaluation" \
  --mount "type=bind,src=$gt_root,dst=/benchmark-data/gt,readonly" \
  "${CAD_EVAL_IMAGE:-cadgenbench-pi-agent:baseline-evaluator-20261105-v1}" \
  cadgenbench evaluate /evaluation --workers "$workers"
