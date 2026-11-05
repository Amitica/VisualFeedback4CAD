"""Show Target, Current and Preview separately in one aligned three-panel row.

The caller owns the shared camera, projection origin and image transform. This
module never aligns images. Each panel retains the complete input frame and
shares one display scale; no matching or correctness scoring is performed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

INK_THRESHOLD = 200
DISPLAY_PADDING_PX = 12
TARGET_COLOR = (65, 70, 78)
CAD_COLOR = (25, 105, 210)
CAD_OPACITY = 0.78


def _projection_contours(
    render_path: Path, size: tuple[int, int]
) -> tuple[list[np.ndarray] | None, dict[str, Any]]:
    """Load optional native curves only when bound to this exact raster.

    Older saved projections use explicit raster extraction as a fallback.
    Sidecars never name another file, and symlinks are not followed.
    """
    sidecar = render_path.with_suffix(".contours.json")
    fallback = {"source": "raster", "reason": "Native projection curves not saved."}
    if not sidecar.exists() and not sidecar.is_symlink():
        return None, fallback
    try:
        if sidecar.is_symlink() or not sidecar.is_file():
            raise ValueError("Projection curve sidecar must be a regular file")
        if sidecar.stat().st_size > 32_000_000:
            raise ValueError("Projection curve sidecar exceeds size limit")
        data = json.loads(sidecar.read_text())
        if (
            data.get("schema_version") != 1
            or data.get("kind") != "projected_cad_contours"
            or data.get("render_sha256") != _sha256(render_path)
            or (data.get("width"), data.get("height")) != size
        ):
            raise ValueError("Projection curves do not match the rendered image")
        raw = data.get("visible_polylines")
        if not isinstance(raw, list) or len(raw) > 20_000:
            raise ValueError("Invalid visible projection curves")
        curves = []
        points = 0
        for line in raw:
            curve = np.asarray(line, dtype=np.float64)
            if (
                curve.ndim != 2
                or curve.shape[1] != 2
                or len(curve) < 2
                or not np.all(np.isfinite(curve))
                or np.any(np.abs(curve) > 1e8)
            ):
                raise ValueError("Invalid projection curve coordinates")
            points += len(curve)
            if points > 500_000:
                raise ValueError("Projection curves exceed point limit")
            curves.append(curve)
        return curves, {"source": "native_hlr", "sha256": _sha256(sidecar)}
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return None, {"source": "raster", "reason": str(exc)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return _white_rgb(source)


def _white_rgb(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    return background.convert("RGB")


def overlay_projection(reference: Image.Image, projection: Image.Image) -> Image.Image:
    """Return a native-size source overlay without fitting or matching curves.

    Continuous grayscale alpha preserves antialiasing. Target ink is dark gray;
    CAD ink is translucent blue, with the same encoding for every candidate.
    """
    if reference.size != projection.size:
        raise ValueError("Target and CAD projection must have identical dimensions")
    target_alpha = (
        255.0 - np.asarray(_white_rgb(reference).convert("L"), dtype=float)
    ) / 255.0
    cad_alpha = (
        CAD_OPACITY
        * (255.0 - np.asarray(_white_rgb(projection).convert("L"), dtype=float))
        / 255.0
    )
    target = (
        255.0 * (1.0 - target_alpha[..., None])
        + np.asarray(TARGET_COLOR) * target_alpha[..., None]
    )
    overlay = (
        target * (1.0 - cad_alpha[..., None])
        + np.asarray(CAD_COLOR) * cad_alpha[..., None]
    )
    return Image.fromarray(np.clip(np.rint(overlay), 0, 255).astype(np.uint8))


def _ink(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L")) <= INK_THRESHOLD


def _roi_pixels(
    roi: list[int] | None, size: tuple[int, int]
) -> tuple[list[int], tuple[int, int, int, int]]:
    normalized = [0, 0, 1000, 1000] if roi is None else list(roi)
    if len(normalized) != 4 or any(
        isinstance(value, bool) or not isinstance(value, int) for value in normalized
    ):
        raise ValueError("target_roi must be four integer boundaries in 0..1000")
    left, top, right, bottom = normalized
    if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
        raise ValueError(
            "target_roi must satisfy 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000"
        )
    if right - left < 50 or bottom - top < 50:
        raise ValueError("target_roi width and height must each be at least 50")
    width, height = size
    return normalized, (
        math.floor(left * width / 1000),
        math.floor(top * height / 1000),
        math.ceil(right * width / 1000),
        math.ceil(bottom * height / 1000),
    )


def _font(
    size: int, *, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _card(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    if len(panels) != 3 or len({panel.size for _, panel in panels}) != 1:
        raise ValueError(
            "The edit card requires three equally sized Target, Current and Preview panels"
        )
    margin, gap = 20, 20
    panel_width = 480
    width = 2 * margin + 3 * panel_width + 2 * gap
    native_width, native_height = panels[0][1].size
    ratio = min(panel_width / native_width, 540 / native_height)
    image_size = (
        max(1, round(native_width * ratio)),
        max(1, round(native_height * ratio)),
    )
    title_font = _font(20, bold=True)
    title_height = 40
    panel_height = title_height + image_size[1]
    canvas = Image.new("RGB", (width, 2 * margin + panel_height), (245, 247, 249))
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        x = margin + index * (panel_width + gap)
        y = margin
        draw.rectangle((x, y, x + panel_width - 1, y + panel_height - 1), fill="white")
        draw.text((x + 10, y + 8), title, fill=(25, 32, 40), font=title_font)
        canvas.paste(
            panel.resize(image_size, Image.Resampling.LANCZOS),
            (x + (panel_width - image_size[0]) // 2, y + title_height),
        )
    return canvas


def compose_shape_feedback(
    reference_path: Path,
    current_render_path: Path,
    candidate_render_path: Path | None = None,
    output_dir: Path | None = None,
    *,
    region_id: str,
    candidate_id: str | None = None,
    hypothesis: str = "",
    target_roi: list[int] | None = None,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write one Target | Current | Preview row per edit.

    The Target panel shows the original drawing; CAD panels are monochrome.
    No matching or scoring is performed.
    Raw images remain separate saved artifacts. Candidates share the caller's
    frozen frame and one display scale. The ROI
    only selects supplementary native crops and diagnostics. The positional
    single-candidate form remains available for historical worker callers.
    """
    legacy = candidates is None
    if legacy:
        if candidate_render_path is None or candidate_id is None:
            raise ValueError("Provide a candidate render and identifier")
        candidates = [
            {
                "candidate_id": candidate_id,
                "render_path": candidate_render_path,
                "hypothesis": hypothesis,
            }
        ]
    elif candidate_render_path is not None or candidate_id is not None or hypothesis:
        raise ValueError("Do not mix a candidates list with single-candidate fields")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
        raise ValueError("Provide one to three candidates")
    if output_dir is None:
        raise ValueError("output_dir is required")
    normalized = []
    for candidate in candidates:
        identifier = (
            candidate.get("candidate_id") if isinstance(candidate, dict) else None
        )
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", identifier) is None
        ):
            raise ValueError("Each candidate_id must be a safe identifier")
        label = candidate.get("hypothesis", "")
        if not isinstance(label, str) or len(label) > 2000:
            raise ValueError("hypothesis must contain at most 2000 characters")
        normalized.append(
            {
                "candidate_id": identifier,
                "hypothesis": label.strip(),
                "render_path": Path(candidate["render_path"]),
            }
        )
    if len({candidate["candidate_id"] for candidate in normalized}) != len(normalized):
        raise ValueError("Candidate identifiers must be unique")
    paths = [
        Path(reference_path),
        Path(current_render_path),
        *(candidate["render_path"] for candidate in normalized),
    ]
    images = [_open_rgb(path) for path in paths]
    reference = images[0]
    if any(image.size != reference.size for image in images):
        raise ValueError(
            "Reference, current and candidate renders must have identical image dimensions"
        )
    normalized_roi, bounds = _roi_pixels(target_roi, reference.size)
    masks = [_ink(image) for image in images]
    if not np.any(masks[0]):
        raise ValueError("Reference image contains no ink at the fixed threshold")
    titles = [
        "Target",
        "Current",
        *(f"Preview Candidate {candidate['candidate_id']}" for candidate in normalized),
    ]
    keys = [
        "original",
        "current",
        *(f"candidate_{candidate['candidate_id']}" for candidate in normalized),
    ]
    left, top, right, bottom = bounds
    selection = np.s_[top:bottom, left:right]
    warnings = ["Drawing ink includes annotations."]
    for name, mask in zip(titles[1:], masks[1:], strict=True):
        if not np.any(mask):
            warnings.append(f"{name} CAD image is empty.")
        if (
            np.any(mask[0])
            or np.any(mask[-1])
            or np.any(mask[:, 0])
            or np.any(mask[:, -1])
        ):
            warnings.append(
                f"{name} projection touches the input image boundary; geometry outside that raster cannot be recovered by display scaling."
            )
    if not np.any(masks[0][selection]):
        warnings.append("Selected ROI contains no reference ink.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for index, (key, original) in enumerate(zip(keys, images, strict=True)):
        for suffix, panel in (("", original.crop(bounds)), ("_full", original)):
            artifact_key = key + suffix
            if index >= 2:
                # IDs may themselves end in '_full'. Separate their file paths
                # and use a delimiter forbidden in IDs for the full-image key.
                directory = (
                    output_dir / "candidates" / normalized[index - 2]["candidate_id"]
                )
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / ("full.png" if suffix else "crop.png")
                artifact_key = key + (":full" if suffix else "")
            else:
                path = output_dir / f"{key}{suffix}.png"
            panel.save(path, format="PNG", optimize=True)
            files[artifact_key] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "width": panel.width,
                "height": panel.height,
            }

    def padded(image: Image.Image) -> Image.Image:
        panel = Image.new(
            "RGB",
            (
                image.width + 2 * DISPLAY_PADDING_PX,
                image.height + 2 * DISPLAY_PADDING_PX,
            ),
            "white",
        )
        panel.paste(image, (DISPLAY_PADDING_PX, DISPLAY_PADDING_PX))
        return panel

    target_panel = padded(reference)
    current_panel = padded(images[1].convert("L").convert("RGB"))
    cards = []
    for index, candidate in enumerate(normalized):
        panels = [
            ("Target", target_panel),
            ("Current", current_panel),
            ("Preview", padded(images[index + 2].convert("L").convert("RGB"))),
        ]
        card = _card(panels)
        card_dir = (
            output_dir
            if index == 0
            else output_dir / "candidates" / candidate["candidate_id"]
        )
        card_path = card_dir / "shape-feedback.png"
        card.save(card_path, format="PNG", optimize=True)
        cards.append(
            {
                "candidate_id": candidate["candidate_id"],
                "card_path": str(card_path.resolve()),
                "sha256": _sha256(card_path),
                "width": card.width,
                "height": card.height,
                "mime_type": "image/png",
                "card_layout": "three_panel_row",
                "panel_count": 3,
                "panel_titles": [title for title, _ in panels],
            }
        )
    candidate_records = [
        {
            "candidate_id": candidate["candidate_id"],
            "hypothesis": candidate["hypothesis"],
            "render_sha256": _sha256(candidate["render_path"]),
            "cad_ink_pixels": int(np.count_nonzero(mask[selection])),
            "cad_empty_full_frame": not bool(np.any(mask)),
        }
        for candidate, mask in zip(normalized, masks[2:], strict=True)
    ]
    result = {
        "schema_version": 1,
        "kind": "aligned_cad_candidate_comparison",
        "region_id": region_id,
        "candidates": candidate_records,
        **{key: value for key, value in cards[0].items() if key != "candidate_id"},
        "card_count": len(cards),
        "cards": cards,
        "files": files,
        "reference_sha256": _sha256(paths[0]),
        "current_render_sha256": _sha256(paths[1]),
        "drawing_size": {"width": reference.width, "height": reference.height},
        "target_roi": normalized_roi,
        "target_pixel_roi": list(bounds),
        "display": {
            "domain": "complete_input_frame",
            "native_panel_domain": "selected_roi",
            "padding_px": DISPLAY_PADDING_PX,
            "shared_fit": True,
            "independent_cad_scaling": False,
        },
        "parameters": {"ink_threshold_inclusive": INK_THRESHOLD},
        "diagnostics": {
            "reference_ink_pixels": int(np.count_nonzero(masks[0][selection])),
            "current_cad_ink_pixels": int(np.count_nonzero(masks[1][selection])),
            "current_cad_empty_full_frame": not bool(np.any(masks[1])),
        },
        "warnings": warnings,
    }
    if legacy:
        result.update(
            candidate_id=normalized[0]["candidate_id"],
            hypothesis=normalized[0]["hypothesis"],
            candidate_render_sha256=candidate_records[0]["render_sha256"],
        )
        result["diagnostics"].update(
            candidate_cad_ink_pixels=candidate_records[0]["cad_ink_pixels"],
            candidate_cad_empty_full_frame=candidate_records[0]["cad_empty_full_frame"],
        )
    metadata_path = output_dir / "shape-feedback.json"
    metadata_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    files["metadata"] = {
        "path": str(metadata_path.resolve()),
        "sha256": _sha256(metadata_path),
        "mime_type": "application/json",
    }
    return result
