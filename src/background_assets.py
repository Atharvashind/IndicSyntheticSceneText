from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT_DIR = Path(__file__).resolve().parents[1]
SUPPORTED_SOURCE_TYPES = {"real_photo", "ai_generated"}
SUPPORTED_SCENE_CATEGORIES = {"shopfront", "outdoor_hoarding"}
SUPPORTED_REVIEW_STATUSES = {"pending", "approved", "rejected"}
SUPPORTED_LICENSE_STATUSES = {"verified", "unverified", "unknown", "not_applicable"}
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "TIFF", "BMP"}


@dataclass(frozen=True)
class BackgroundAsset:
    record: dict
    path: Path


@dataclass(frozen=True)
class BackgroundManifest:
    manifest_path: Path
    assets: Tuple[BackgroundAsset, ...]
    excluded: Tuple[dict, ...]
    warnings: Tuple[dict, ...]


def _project_relative_path(root: Path, value: str) -> Optional[Path]:
    path = Path(value)
    if path.is_absolute():
        return None
    resolved_root = root.resolve()
    resolved_path = (resolved_root / path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved_path


def _validate_surface_polygon(value: Any, width: int, height: int) -> List[str]:
    if not isinstance(value, list) or len(value) != 4:
        return ["signboard_surface_polygon must contain exactly four pixel-edge points."]
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return ["signboard_surface_polygon must contain numeric x/y coordinates."]
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return ["signboard_surface_polygon must contain four finite x/y points."]
    if (
        (points[:, 0] < 0).any()
        or (points[:, 0] > width).any()
        or (points[:, 1] < 0).any()
        or (points[:, 1] > height).any()
    ):
        return ["signboard_surface_polygon lies outside the background image."]
    edges = np.roll(points, -1, axis=0) - points
    crosses = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if (np.abs(crosses) < 1e-6).any() or not (np.all(crosses > 0) or np.all(crosses < 0)):
        return ["signboard_surface_polygon must be non-degenerate and convex."]
    area = abs(
        sum(
            points[index, 0] * points[(index + 1) % 4, 1]
            - points[(index + 1) % 4, 0] * points[index, 1]
            for index in range(4)
        )
    ) / 2
    if area < 1:
        return ["signboard_surface_polygon has negligible area."]
    signed_area = sum(
        points[index, 0] * points[(index + 1) % 4, 1]
        - points[(index + 1) % 4, 0] * points[index, 1]
        for index in range(4)
    ) / 2
    if signed_area <= 0:
        return [
            "signboard_surface_polygon must use top-left, top-right, bottom-right, bottom-left order."
        ]
    return []


def load_background_image(path: Path, expected_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    try:
        with Image.open(path) as source:
            if source.format not in SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f"Unsupported background image format: {source.format!r}.")
            source.verify()
        with Image.open(path) as source:
            if source.format not in SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f"Unsupported background image format: {source.format!r}.")
            if source.width <= 0 or source.height <= 0:
                raise ValueError("Background image dimensions must be positive.")
            if expected_size is not None and source.size != expected_size:
                raise ValueError(
                    f"Image dimensions {source.size} do not match manifest dimensions {expected_size}."
                )
            return source.convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise ValueError(f"Cannot read background image '{path}': {error}") from error


def _validate_asset_record(
    record: Any,
    project_root: Path,
    policy: dict,
) -> Tuple[Optional[BackgroundAsset], List[str], List[str]]:
    errors: List[str] = []
    reasons: List[str] = []
    if not isinstance(record, dict):
        return None, ["Asset record must be a JSON object."], reasons

    background_id = record.get("background_id")
    if not isinstance(background_id, str) or not background_id.strip():
        errors.append("background_id must be a non-empty string.")
    asset_id = record.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id.strip():
        errors.append("asset_id must be a non-empty string.")

    relative_path = record.get("relative_path")
    path = (
        _project_relative_path(project_root, relative_path)
        if isinstance(relative_path, str) and relative_path.strip()
        else None
    )
    if path is None:
        errors.append("relative_path must be a project-relative path that stays within the project.")
    elif not path.is_file():
        errors.append(f"Background file is missing: {relative_path}.")
    elif policy.get("asset_directory"):
        asset_directory = (project_root / policy["asset_directory"]).resolve()
        try:
            path.relative_to(asset_directory)
        except ValueError:
            errors.append(
                f"Background file must be inside the configured asset directory "
                f"'{policy['asset_directory']}'."
            )

    source_type = record.get("source_type")
    if not isinstance(source_type, str) or source_type not in SUPPORTED_SOURCE_TYPES:
        errors.append(f"source_type must be one of {sorted(SUPPORTED_SOURCE_TYPES)}.")
    scene_category = record.get("scene_category")
    if not isinstance(scene_category, str) or scene_category not in SUPPORTED_SCENE_CATEGORIES:
        errors.append(f"scene_category must be one of {sorted(SUPPORTED_SCENE_CATEGORIES)}.")
    original_filename = record.get("original_filename")
    if not isinstance(original_filename, str) or not original_filename.strip():
        errors.append("original_filename must be recorded.")
    elif Path(original_filename).name != original_filename:
        errors.append("original_filename must contain only the original file name, not a path.")
    provenance = record.get("provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        reasons.append("provenance is incomplete.")

    licensing = record.get("licensing")
    if not isinstance(licensing, dict):
        reasons.append("licensing information is missing.")
        licensing = {}
    license_status = licensing.get("verification_status")
    if not isinstance(license_status, str) or license_status not in SUPPORTED_LICENSE_STATUSES:
        reasons.append("licensing verification_status is missing or invalid.")
    if policy.get("require_verified_license", True):
        if license_status != "verified":
            reasons.append("license has not been verified.")
    if licensing.get("research_use_allowed") is not True:
        reasons.append("permission for research use has not been verified.")
    if not isinstance(licensing.get("terms"), str) or not licensing["terms"].strip():
        reasons.append("license terms or their known status must be recorded.")

    width, height = record.get("image_width"), record.get("image_height")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or isinstance(height, bool)
        or width <= 0
        or height <= 0
    ):
        errors.append("image_width and image_height must be positive integers.")
        width, height = 0, 0
    elif (
        width < policy.get("minimum_image_width", 1)
        or height < policy.get("minimum_image_height", 1)
    ):
        reasons.append(
            "image is below the configured minimum resolution "
            f"({policy.get('minimum_image_width', 1)}x{policy.get('minimum_image_height', 1)})."
        )

    if path is not None and path.is_file() and width > 0 and height > 0:
        try:
            load_background_image(path, (width, height))
        except ValueError as error:
            errors.append(str(error))

    surface = record.get("signboard_surface_polygon")
    errors.extend(_validate_surface_polygon(surface, width, height) if width and height else [])
    if (
        isinstance(surface, list)
        and len(surface) == 4
        and width > 0
        and height > 0
        and all(isinstance(point, (list, tuple)) and len(point) == 2 for point in surface)
    ):
        try:
            points = np.asarray(surface, dtype=np.float64)
            lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
            surface_width = float((lengths[0] + lengths[2]) / 2)
            surface_height = float((lengths[1] + lengths[3]) / 2)
            if (
                surface_width < policy.get("minimum_surface_width", 1)
                or surface_height < policy.get("minimum_surface_height", 1)
            ):
                reasons.append(
                    "verified signboard surface is below the configured minimum dimensions."
                )
        except (TypeError, ValueError):
            pass
    surface_status = record.get("surface_verification_status")
    if surface_status != "verified":
        reasons.append("blank signboard surface geometry has not been verified.")
    if policy.get("require_manual_review_approved", True):
        if record.get("manual_review_status") != "approved":
            reasons.append("manual review is not approved.")
    review_status = record.get("manual_review_status")
    if not isinstance(review_status, str) or review_status not in SUPPORTED_REVIEW_STATUSES:
        errors.append(f"manual_review_status must be one of {sorted(SUPPORTED_REVIEW_STATUSES)}.")

    warnings = record.get("validation_warnings", [])
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        errors.append("validation_warnings must be a list of strings.")
    elif warnings:
        reasons.append("asset has unresolved validation warnings.")

    if source_type == "ai_generated" and record.get("generation_prompt") is not None and not isinstance(
        record.get("generation_prompt"), str
    ):
        errors.append("generation_prompt must be a string or null when applicable.")
    elif source_type == "real_photo" and record.get("generation_prompt") not in (None, ""):
        errors.append("generation_prompt is only valid for ai_generated assets.")
    created_date = record.get("created_date")
    if created_date is not None:
        if not isinstance(created_date, str):
            errors.append("created_date must be an ISO date string or null.")
        else:
            try:
                date.fromisoformat(created_date)
            except ValueError:
                errors.append("created_date must be an ISO date string or null.")

    if errors or path is None:
        return None, errors, reasons
    return BackgroundAsset(dict(record), path), errors, reasons


def load_background_manifest(
    manifest_path: Path,
    project_root: Path = ROOT_DIR,
    policy: Optional[dict] = None,
) -> BackgroundManifest:
    if policy is None:
        policy = {}
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read background manifest '{manifest_path}': {error}") from error
    if not isinstance(manifest_data, dict) or manifest_data.get("schema_version") != "1.0":
        raise ValueError("Background manifest must be a JSON object with schema_version '1.0'.")
    if manifest_data.get("coordinate_convention") != (
        "pixel_edges; top-left origin; x right; y down; polygon order "
        "top-left, top-right, bottom-right, bottom-left"
    ):
        raise ValueError("Background manifest has an unsupported coordinate_convention.")
    records = manifest_data.get("assets")
    if not isinstance(records, list):
        raise ValueError("Background manifest assets must be a list.")

    seen_ids = set()
    id_counts = Counter(
        record.get("background_id")
        for record in records
        if isinstance(record, dict) and isinstance(record.get("background_id"), str)
    )
    asset_id_counts = Counter(
        record.get("asset_id")
        for record in records
        if isinstance(record, dict) and isinstance(record.get("asset_id"), str)
    )
    eligible: List[BackgroundAsset] = []
    excluded: List[dict] = []
    warnings: List[dict] = []
    for index, record in enumerate(records):
        asset, errors, reasons = _validate_asset_record(record, project_root, policy)
        asset_id = record.get("background_id") if isinstance(record, dict) else None
        if isinstance(asset_id, str) and id_counts[asset_id] > 1:
            errors.append(f"Duplicate background_id: {asset_id}.")
        elif isinstance(asset_id, str) and asset_id not in seen_ids:
            seen_ids.add(asset_id)
        if isinstance(record, dict):
            manifest_asset_id = record.get("asset_id")
            if isinstance(manifest_asset_id, str) and asset_id_counts[manifest_asset_id] > 1:
                errors.append(f"Duplicate asset_id: {manifest_asset_id}.")
        if errors or reasons:
            excluded.append(
                {
                    "background_id": asset_id,
                    "manifest_index": index,
                    "errors": errors,
                    "exclusion_reasons": reasons,
                }
            )
        elif asset is not None:
            eligible.append(asset)
        if isinstance(record, dict):
            record_warnings = record.get("validation_warnings", [])
            if isinstance(record_warnings, list) and record_warnings:
                warnings.append(
                    {"background_id": asset_id, "warnings": list(record_warnings)}
                )
    return BackgroundManifest(
        manifest_path=manifest_path,
        assets=tuple(eligible),
        excluded=tuple(excluded),
        warnings=tuple(warnings),
    )


def transform_polygon(points: List[List[float]], homography: np.ndarray) -> List[List[float]]:
    matrix = np.asarray(homography, dtype=np.float64)
    polygon = np.asarray(points, dtype=np.float64)
    if matrix.shape != (3, 3) or polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("A polygon and a 3x3 homography are required.")
    if not np.isfinite(matrix).all() or not np.isfinite(polygon).all():
        raise ValueError("Polygon and homography coordinates must be finite.")
    homogeneous = np.column_stack((polygon, np.ones(len(polygon))))
    mapped = (matrix @ homogeneous.T).T
    if (np.abs(mapped[:, 2]) < 1e-10).any():
        raise ValueError("Homography maps a polygon point to infinity.")
    result = mapped[:, :2] / mapped[:, 2:3]
    if not np.isfinite(result).all():
        raise ValueError("Homography produced non-finite polygon coordinates.")
    return result.tolist()


def polygon_bbox(points: List[List[float]]) -> dict:
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError("Polygon must contain finite x/y points.")
    return {
        "x_min": float(np.min(coordinates[:, 0])),
        "y_min": float(np.min(coordinates[:, 1])),
        "x_max": float(np.max(coordinates[:, 0])),
        "y_max": float(np.max(coordinates[:, 1])),
    }


def polygon_is_in_frame(points: List[List[float]], width: int, height: int) -> bool:
    bbox = polygon_bbox(points)
    return (
        bbox["x_min"] >= 0
        and bbox["y_min"] >= 0
        and bbox["x_max"] <= width
        and bbox["y_max"] <= height
        and bbox["x_max"] > bbox["x_min"]
        and bbox["y_max"] > bbox["y_min"]
        and math.isfinite(bbox["x_min"])
    )
