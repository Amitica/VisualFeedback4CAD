"""Locate material edits by projecting BRep differences in a frozen CAD frame.

This module never reads Target ink. Colors encode added/removed material, not
agreement with a drawing. Difference boundaries include occluded/deleted parts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

ADDED_COLOR = (28, 145, 86)
REMOVED_COLOR = (215, 45, 55)
CAD_COLOR = (105, 113, 122)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _difference(minuend: Any, subtrahend: Any) -> Any | None:
    from build123d import Compound
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

    operation = BRepAlgoAPI_Cut(minuend.wrapped, subtrahend.wrapped)
    if not operation.IsDone():
        raise RuntimeError("BRep Boolean difference did not complete")
    wrapped = operation.Shape()
    if wrapped.IsNull():
        return None
    shape = Compound.cast(wrapped)
    solids = list(shape.solids())
    if not solids:
        return None
    if not all(solid.is_valid for solid in solids):
        raise RuntimeError("BRep Boolean difference produced invalid solids")
    return solids[0] if len(solids) == 1 else Compound(children=solids)


def _render_effect(
    step_path: Path,
    *,
    image_size: tuple[int, int],
    view_spec: dict[str, Any],
    transform: dict[str, Any],
    projection_origin: list[float],
) -> Image.Image:
    from cadgenbench.tools.drawing_view import (
        SimilarityTransform,
        _render_cad_view,
        project_step,
    )

    projection = project_step(
        step_path,
        view=view_spec["name"],
        direction=tuple(view_spec["direction"]),
        up=tuple(view_spec["up"]),
        look_at=tuple(projection_origin),
    )
    return _render_cad_view(
        image_size,
        projection,
        SimilarityTransform(**transform),
        show_hidden_lines=True,
    )


def _safe_destination(directory: Path, name: str) -> Path:
    if any(parent.is_symlink() for parent in (directory, *directory.parents)):
        raise ValueError("BRep output directory must not contain symlinks")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("BRep output must be a regular file")
    return path


def highlight_brep_changes(
    *,
    current_step_path: Path,
    candidate_step_path: Path,
    preview_image: Image.Image,
    view_spec: dict[str, Any],
    transform: dict[str, Any],
    projection_origin: list[float],
    output_dir: Path,
) -> tuple[Image.Image, dict[str, Any], dict[str, Any]]:
    """Return Preview plus two material-difference boundary projections.

    Compute both cuts before publishing any success. A successful pair of empty
    cuts means no solid-material change. Failures are handled by the caller;
    they must never be interpreted as an unchanged candidate.
    """
    from build123d import export_step, import_step

    if (
        view_spec.get("section_offset_fraction") is not None
        or view_spec.get("section_keep") is not None
    ):
        raise ValueError("BRep highlights require a non-section frozen view")
    if len(projection_origin) != 3 or not np.all(np.isfinite(projection_origin)):
        raise ValueError("projection_origin must contain three finite coordinates")
    current_step_path, candidate_step_path = (
        Path(current_step_path),
        Path(candidate_step_path),
    )
    current, preview = (
        import_step(str(current_step_path)),
        import_step(str(candidate_step_path)),
    )
    if (
        not current.solids()
        or not preview.solids()
        or not current.is_valid
        or not preview.is_valid
    ):
        raise ValueError("BRep highlights require valid solid Current and Preview CAD")
    effects = {
        "added": _difference(preview, current),
        "removed": _difference(current, preview),
    }
    evidence: dict[str, Any] = {
        "status": "ok",
        "current_step_sha256": _sha(current_step_path),
        "candidate_step_sha256": _sha(candidate_step_path),
    }
    files, masks = {}, {}
    for name, shape in effects.items():
        mask = np.zeros((preview_image.height, preview_image.width), dtype=bool)
        record: dict[str, Any] = {
            "solid_count": 0,
            "bbox_mm": None,
            "projection_pixel_count": 0,
        }
        if shape is not None:
            bounds = shape.bounding_box()
            record.update(
                solid_count=len(shape.solids()),
                bbox_mm={
                    f"{axis}_{end}": float(getattr(getattr(bounds, end), axis.upper()))
                    for axis in "xyz"
                    for end in ("min", "max")
                },
            )
            step_path = _safe_destination(Path(output_dir), f"{name}.step")
            export_step(shape, str(step_path))
            rendered = _render_effect(
                step_path,
                image_size=preview_image.size,
                view_spec=view_spec,
                transform=transform,
                projection_origin=projection_origin,
            )
            if rendered.size != preview_image.size:
                raise ValueError(
                    "Difference projection does not use the frozen image dimensions"
                )
            render_path = _safe_destination(Path(output_dir), f"{name}.png")
            rendered.save(render_path)
            record.update(
                step_path=str(step_path.resolve()),
                render_path=str(render_path.resolve()),
            )
            files[f"{name}:step"] = {
                "path": str(step_path.resolve()),
                "sha256": _sha(step_path),
                "mime_type": "model/step",
            }
            files[f"{name}:render"] = {
                "path": str(render_path.resolve()),
                "sha256": _sha(render_path),
                "mime_type": "image/png",
                "width": rendered.width,
                "height": rendered.height,
            }
            # Widen the boundary for readable location cues at card scale.
            stroke = Image.fromarray(
                np.where(np.asarray(rendered.convert("L")) <= 220, 255, 0).astype(
                    np.uint8
                )
            )
            mask = np.asarray(stroke.filter(ImageFilter.MaxFilter(3))) > 0
            record["projection_pixel_count"] = int(np.count_nonzero(mask))
        masks[name] = mask
        evidence[name] = record
    overlap = masks["added"] & masks["removed"]
    evidence["overlap_pixels"] = int(np.count_nonzero(overlap))
    if not np.any(masks["added"] | masks["removed"]):
        return preview_image.copy(), evidence, files
    pixels = np.asarray(preview_image.convert("RGB")).copy()
    gray = np.asarray(preview_image.convert("L"), dtype=float) / 255
    # Preserve antialiasing and the whole Preview context while de-emphasizing it.
    pixels[:] = np.rint(
        np.asarray(CAD_COLOR) + gray[:, :, None] * (255 - np.asarray(CAD_COLOR))
    ).astype(np.uint8)
    pixels[masks["added"]] = ADDED_COLOR
    pixels[masks["removed"]] = REMOVED_COLOR
    yy, xx = np.indices(overlap.shape)
    pixels[overlap & (((xx + yy) // 4) % 2 == 0)] = ADDED_COLOR
    return Image.fromarray(pixels), evidence, files
