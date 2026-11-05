# Copyright 2026 Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Isolated VTK renderer for candidate-only evaluation.

This module is intentionally executed in a fresh interpreter by
`evaluate_candidate_only`. Keeping VTK/OpenGL out of the process that owns
the CAD validity multiprocessing pool avoids a native-runtime deadlock.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def render_candidate(candidate_step: Path, renders_dir: Path) -> None:
    """Write the canonical four PNG views and rotating WebP."""
    from cadgenbench.common.viewer import (
        render_step,
        render_step_turntable_webp,
    )

    renders_dir.mkdir(parents=True, exist_ok=True)
    for image in render_step(candidate_step):
        (renders_dir / f"{image.name}.png").write_bytes(image.data)
    (renders_dir / "rotating.webp").write_bytes(
        render_step_turntable_webp(candidate_step)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_step", type=Path)
    parser.add_argument("renders_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    render_candidate(args.candidate_step, args.renders_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
