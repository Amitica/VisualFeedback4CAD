# Copyright 2026 Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CLI adapter for the agent-facing drawing comparison tool."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "drawing-view",
        help="Project a STEP model and align it to a reference engineering drawing.",
    )
    parser.add_argument("model", type=Path, help="Candidate STEP model.")
    parser.add_argument("reference", type=Path, help="Reference drawing PNG/JPEG.")
    view_selection = parser.add_mutually_exclusive_group(required=True)
    view_selection.add_argument("--view", help="One named view to validate.")
    view_selection.add_argument(
        "--candidates",
        nargs="+",
        metavar="VIEW",
        help="Two or more named LLM-proposed views to register and rank.",
    )
    parser.add_argument(
        "--direction",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        help="Override the preset observer direction for an auxiliary/isometric view.",
    )
    parser.add_argument(
        "--up",
        type=float,
        nargs=3,
        metavar=("UX", "UY", "UZ"),
        help="Override the preset screen-up vector.",
    )
    parser.add_argument(
        "--section-origin",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Cut at this model-space point, normal to the view, before HLR.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will contain only comparison.png.",
    )
    image_scope = parser.add_mutually_exclusive_group(required=True)
    image_scope.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Rough part ROI in reference-image pixels.",
    )
    image_scope.add_argument(
        "--full-image",
        action="store_true",
        help="Confirm that the reference is already tightly cropped to the part.",
    )
    parser.add_argument(
        "--allow-reflection",
        action="store_true",
        help="Try an additional left/right reflected registration.",
    )
    parser.add_argument(
        "--max-rotation",
        type=float,
        default=6.0,
        help="Maximum 2-D roll correction in degrees (default: 6).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        help="Dark-pixel threshold (default: Otsu with an antialiasing floor).",
    )
    parser.add_argument(
        "--min-roi-fill",
        type=float,
        default=0.70,
        help="Minimum projection/ROI fit-scale ratio (default: 0.70).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=45,
        help="Differential-evolution iterations (default: 45).",
    )
    parser.set_defaults(handler=run)
    return parser


def add_sheet_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the one-call, multi-view comparison-sheet command."""

    parser = subparsers.add_parser(
        "drawing-sheet",
        help="Compare several selected drawing views and return one mean score/image.",
    )
    parser.add_argument("model", type=Path, help="Candidate STEP model.")
    parser.add_argument(
        "manifest",
        type=Path,
        help="JSON manifest containing a non-empty 'views' list.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will contain only comparison.png.",
    )
    parser.add_argument("--threshold", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=45)
    parser.set_defaults(handler=run_sheet)
    return parser


def run(args: argparse.Namespace) -> int:
    _reexec_with_environment_libraries()
    from cadgenbench.tools.drawing_view import (
        compare_step_to_drawing,
        compare_view_candidates,
    )

    try:
        if args.candidates:
            if args.direction or args.up or args.section_origin:
                raise ValueError(
                    "--direction, --up, and --section-origin apply only to "
                    "single --view mode; use the Python ViewCandidate API for "
                    "multiple custom views"
                )
            result = compare_view_candidates(
                args.model,
                args.reference,
                args.output_dir,
                candidates=args.candidates,
                roi=tuple(args.roi) if args.roi else None,
                allow_reflection=args.allow_reflection,
                max_rotation_deg=args.max_rotation,
                threshold=args.threshold,
                min_roi_fill=args.min_roi_fill,
                seed=args.seed,
                max_iterations=args.max_iterations,
            )
            payload = {
                "score": result.score,
                "comparison_png": str(result.comparison_png.resolve()),
            }
            print(json.dumps(payload, indent=2))
            return 0

        result = compare_step_to_drawing(
            args.model,
            args.reference,
            args.output_dir,
            view=args.view,
            direction=tuple(args.direction) if args.direction else None,
            up=tuple(args.up) if args.up else None,
            section_origin=(
                tuple(args.section_origin) if args.section_origin else None
            ),
            roi=tuple(args.roi) if args.roi else None,
            allow_reflection=args.allow_reflection,
            max_rotation_deg=args.max_rotation,
            threshold=args.threshold,
            min_roi_fill=args.min_roi_fill,
            seed=args.seed,
            max_iterations=args.max_iterations,
        )
    except (OSError, RuntimeError, ValueError) as error:
        payload = {
            "status": "error",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 2
    payload = {
        "score": result.score,
        "comparison_png": str(result.comparison_png.resolve()),
    }
    print(json.dumps(payload, indent=2))
    return 0


def run_sheet(args: argparse.Namespace) -> int:
    """Run the compact multi-view interface."""

    _reexec_with_environment_libraries()
    from cadgenbench.tools.drawing_view import compare_step_to_drawings

    try:
        views = _load_drawing_view_manifest(args.manifest)
        result = compare_step_to_drawings(
            args.model,
            views,
            args.output_dir,
            threshold=args.threshold,
            seed=args.seed,
            max_iterations=args.max_iterations,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        payload = {
            "status": "error",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "score": result.score,
                "comparison_png": str(result.comparison_png.resolve()),
            },
            indent=2,
        )
    )
    return 0


def _load_drawing_view_manifest(path: Path):
    """Load selected views; relative references are manifest-relative."""

    from cadgenbench.tools.drawing_view import DrawingViewSpec

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("views")
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must contain a non-empty 'views' list")

    views = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"views[{index}] must be an object")
        reference = Path(item["reference"])
        if not reference.is_absolute():
            reference = path.parent / reference
        views.append(
            DrawingViewSpec(
                name=str(item["name"]),
                reference_image=reference,
                view=str(item["view"]),
                direction=_optional_tuple(item.get("direction"), 3),
                up=_optional_tuple(item.get("up"), 3),
                section_origin=_optional_tuple(item.get("section_origin"), 3),
                roi=_optional_tuple(item.get("roi"), 4),
                allow_reflection=bool(item.get("allow_reflection", False)),
                max_rotation_deg=float(item.get("max_rotation_deg", 6.0)),
                min_roi_fill=float(item.get("min_roi_fill", 0.70)),
            )
        )
    return tuple(views)


def _optional_tuple(value, length: int):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"Expected a JSON array of length {length}, got {value!r}")
    return tuple(value)


def _reexec_with_environment_libraries() -> None:
    """Prefer the active environment's libexpat before importing OCCT."""

    if not sys.platform.startswith("linux") or os.environ.get("CGB_DRAWING_REEXEC"):
        return
    environment_lib = str(Path(sys.prefix) / "lib")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if current.split(":", 1)[0] == environment_lib:
        return
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = (
        f"{environment_lib}:{current}" if current else environment_lib
    )
    environment["CGB_DRAWING_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


__all__ = ["add_sheet_subparser", "add_subparser", "run", "run_sheet"]
