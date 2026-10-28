from __future__ import annotations

import copy
import random
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .content_engine import load_config

PROFILE_NAMES = (
    "D0_clean",
    "D2_distance",
    "D2_perspective",
    "D2_lighting",
    "D2_blur",
    "D2_occlusion",
)


def _ensure_transform_metadata(annotation: dict, profile: str, seed: int) -> None:
    if "scene_difficulty" not in annotation:
        annotation["scene_difficulty"] = {
            "profile": profile,
            "random_seed": seed,
            "transformations": [],
            "visibility_policy": "bbox_unoccluded_area_fraction",
        }


def _get_polygons(annotation: dict) -> List[np.ndarray]:
    polygons = []
    for index, region in enumerate(annotation.get("text_regions", [])):
        points = np.asarray(region.get("polygon"), dtype=np.float64)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError(f"text_regions[{index}] polygon must contain four finite points.")
        polygons.append(points)
    if not polygons:
        raise ValueError("Annotation must contain at least one text polygon.")
    return polygons


def _validate_polygons(polygons: Sequence[np.ndarray], width: int, height: int) -> None:
    for index, polygon in enumerate(polygons):
        if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
            raise ValueError(f"Transformed polygon {index} is invalid.")
        if (
            (polygon[:, 0] < -1e-6).any()
            or (polygon[:, 0] > width + 1e-6).any()
            or (polygon[:, 1] < -1e-6).any()
            or (polygon[:, 1] > height + 1e-6).any()
        ):
            raise ValueError(f"Transformed polygon {index} lies outside the output image.")
        contour = polygon.astype(np.float32)
        if abs(cv2.contourArea(contour)) < 1.0 or not cv2.isContourConvex(contour):
            raise ValueError(f"Transformed polygon {index} is degenerate or non-convex.")


def _set_final_polygons(annotation: dict, polygons: Sequence[np.ndarray]) -> None:
    for region, points in zip(annotation["text_regions"], polygons):
        region["polygon"] = [[float(x), float(y)] for x, y in points]
        region["bbox"] = {
            "x_min": float(points[:, 0].min()),
            "y_min": float(points[:, 1].min()),
            "x_max": float(points[:, 0].max()),
            "y_max": float(points[:, 1].max()),
        }


def _append_transformation(annotation: dict, transformation: dict) -> None:
    annotation["scene_difficulty"]["transformations"].append(transformation)


def _set_full_visibility(annotation: dict) -> None:
    for region in annotation.get("text_regions", []):
        region["visibility"] = {
            "estimated_fraction": 1.0,
            "status": "visible",
            "policy": "bbox_unoccluded_area_fraction",
            "note": "Estimate uses axis-aligned text bbox area, not glyph-mask pixels.",
        }


def _visibility_status(fraction: float, thresholds: dict) -> str:
    if fraction >= thresholds["visible"]:
        return "visible"
    if fraction >= thresholds["partial"]:
        return "partially_occluded"
    return "fully_occluded"


def apply_distance_scale(
    image: Image.Image,
    annotation: dict,
    scale_x: float,
    scale_y: Optional[float] = None,
    *,
    seed: int = 0,
) -> Tuple[Image.Image, dict]:
    if scale_y is None:
        scale_y = scale_x
    if not np.isfinite([scale_x, scale_y]).all() or scale_x <= 0 or scale_y <= 0:
        raise ValueError("Distance scale factors must be finite and positive.")
    width, height = image.size
    output_width = max(1, round(width * scale_x))
    output_height = max(1, round(height * scale_y))
    actual_x = output_width / width
    actual_y = output_height / height
    polygons = [polygon * np.array([actual_x, actual_y]) for polygon in _get_polygons(annotation)]
    _validate_polygons(polygons, output_width, output_height)

    transformed = copy.deepcopy(annotation)
    _set_final_polygons(transformed, polygons)
    transformed["image_width"] = output_width
    transformed["image_height"] = output_height
    _ensure_transform_metadata(
        transformed,
        transformed.get("scene_difficulty", {}).get("profile", "D2_distance"),
        seed,
    )
    _append_transformation(
        transformed,
        {
            "type": "distance_downscale",
            "seed": seed,
            "requested_scale_x": scale_x,
            "requested_scale_y": scale_y,
            "scale_x": actual_x,
            "scale_y": actual_y,
            "input_size": [width, height],
            "output_size": [output_width, output_height],
        },
    )
    return image.resize((output_width, output_height), Image.Resampling.LANCZOS), transformed


def _edge_homography_to_cv(matrix: np.ndarray) -> np.ndarray:
    edge_to_center = np.array(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    center_to_edge = np.array(
        [[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return center_to_edge @ matrix @ edge_to_center


def apply_perspective(
    image: Image.Image,
    annotation: dict,
    homography: np.ndarray,
    *,
    seed: int = 0,
) -> Tuple[Image.Image, dict]:
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Perspective homography must be a finite 3x3 matrix.")
    if abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError("Perspective homography is singular.")
    width, height = image.size
    source_polygons = _get_polygons(annotation)
    transformed_points = []
    for polygon in source_polygons:
        homogeneous = np.column_stack((polygon, np.ones(len(polygon))))
        mapped = (matrix @ homogeneous.T).T
        if (np.abs(mapped[:, 2]) < 1e-10).any():
            raise ValueError("Perspective transformation maps a polygon to infinity.")
        transformed_points.append(mapped[:, :2] / mapped[:, 2:3])
    _validate_polygons(transformed_points, width, height)

    cv_matrix = _edge_homography_to_cv(matrix)
    array = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    warped = cv2.warpPerspective(
        array,
        cv_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    transformed = copy.deepcopy(annotation)
    _set_final_polygons(transformed, transformed_points)
    _ensure_transform_metadata(
        transformed,
        transformed.get("scene_difficulty", {}).get("profile", "D2_perspective"),
        seed,
    )
    _append_transformation(
        transformed,
        {
            "type": "perspective",
            "seed": seed,
            "coordinate_system": "pixel_edges",
            "homography": matrix.tolist(),
            "input_size": [width, height],
            "output_size": [width, height],
            "interpolation": "opencv_inter_linear",
            "border_mode": "replicate",
        },
    )
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)), transformed


def make_seeded_perspective_homography(
    width: int,
    height: int,
    inset_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, List[List[float]], List[List[float]]]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not np.isfinite(inset_fraction) or not 0 <= inset_fraction < 0.2:
        raise ValueError("Perspective inset fraction must be in [0, 0.2).")
    rng = random.Random(seed)
    inset_x = width * inset_fraction
    inset_y = height * inset_fraction
    variation_x = inset_x * 0.45
    variation_y = inset_y * 0.45
    source = np.asarray(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float32,
    )
    destination = np.asarray(
        [
            [rng.uniform(0, variation_x), rng.uniform(0, variation_y)],
            [width - rng.uniform(0, variation_x), rng.uniform(0, variation_y)],
            [width - rng.uniform(0, variation_x), height - rng.uniform(0, variation_y)],
            [rng.uniform(0, variation_x), height - rng.uniform(0, variation_y)],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination).astype(np.float64)
    return matrix, source.astype(float).tolist(), destination.astype(float).tolist()


def apply_lighting(
    image: Image.Image,
    annotation: dict,
    brightness: float,
    contrast: float,
    *,
    shadow_opacity: float = 0.0,
    shadow_sigma: float = 0.0,
    seed: int = 0,
) -> Tuple[Image.Image, dict]:
    if not np.isfinite([brightness, contrast, shadow_opacity, shadow_sigma]).all():
        raise ValueError("Lighting parameters must be finite.")
    if brightness < 0 or contrast < 0 or not 0 <= shadow_opacity <= 1 or shadow_sigma < 0:
        raise ValueError("Lighting parameters are outside their valid ranges.")
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    lit = np.clip((image_array - 127.5) * contrast + 127.5, 0, 255)
    lit = np.clip(lit * brightness, 0, 255).astype(np.uint8)
    output = Image.fromarray(lit, mode="RGB")
    shadow_metadata = None
    if shadow_opacity > 0:
        width, height = output.size
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        shadow_box = (
            round(width * 0.43),
            round(height * 0.02),
            round(width * 0.98),
            round(height * 0.53),
        )
        draw.ellipse(shadow_box, fill=round(255 * shadow_opacity))
        if shadow_sigma > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=shadow_sigma))
        darkness = Image.new("RGB", (width, height), (0, 0, 0))
        output = Image.composite(darkness, output, mask)
        shadow_metadata = {
            "shape": "soft_ellipse",
            "bbox_pixel_edges": list(shadow_box),
            "opacity": shadow_opacity,
            "blur_sigma_px": shadow_sigma,
        }

    transformed = copy.deepcopy(annotation)
    _get_polygons(transformed)
    _ensure_transform_metadata(
        transformed,
        transformed.get("scene_difficulty", {}).get("profile", "D2_lighting"),
        seed,
    )
    _append_transformation(
        transformed,
        {
            "type": "lighting",
            "seed": seed,
            "input_size": list(image.size),
            "output_size": list(image.size),
            "brightness_factor": brightness,
            "contrast_factor": contrast,
            "shadow": shadow_metadata,
            "geometry_changed": False,
        },
    )
    return output, transformed


def apply_blur_degradation(
    image: Image.Image,
    annotation: dict,
    sigma: float,
    *,
    downsample_scale: Optional[float] = None,
    seed: int = 0,
) -> Tuple[Image.Image, dict]:
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Gaussian blur sigma must be finite and non-negative.")
    if downsample_scale is not None and (
        not np.isfinite(downsample_scale) or not 0 < downsample_scale < 1
    ):
        raise ValueError("Downsample scale must be in (0, 1).")
    width, height = image.size
    polygons = _get_polygons(annotation)
    transformed_image = image.convert("RGB")
    if downsample_scale is not None:
        small_size = (
            max(1, round(width * downsample_scale)),
            max(1, round(height * downsample_scale)),
        )
        transformed_image = transformed_image.resize(small_size, Image.Resampling.BILINEAR)
        transformed_image = transformed_image.resize((width, height), Image.Resampling.BILINEAR)
    if sigma > 0:
        transformed_image = transformed_image.filter(ImageFilter.GaussianBlur(radius=sigma))

    transformed = copy.deepcopy(annotation)
    _set_final_polygons(transformed, polygons)
    _ensure_transform_metadata(
        transformed,
        transformed.get("scene_difficulty", {}).get("profile", "D2_blur"),
        seed,
    )
    _append_transformation(
        transformed,
        {
            "type": "blur_degradation",
            "seed": seed,
            "input_size": [width, height],
            "output_size": [width, height],
            "gaussian_sigma_px": sigma,
            "downsample_scale": downsample_scale,
            "upsample_size": [width, height] if downsample_scale is not None else None,
            "geometry_changed": False,
        },
    )
    return transformed_image, transformed


def apply_occlusion(
    image: Image.Image,
    annotation: dict,
    *,
    seed: int,
    width_fraction: float,
    height_fraction: float,
) -> Tuple[Image.Image, dict]:
    if not 0 < width_fraction < 1 or not 0 < height_fraction < 1:
        raise ValueError("Occlusion size fractions must be in (0, 1).")
    width, height = image.size
    regions = annotation.get("text_regions", [])
    if not regions:
        raise ValueError("Occlusion requires at least one annotated text region.")
    rng = random.Random(seed)
    target = rng.choice(regions)["bbox"]
    box_width = max(1, round(width * width_fraction))
    box_height = max(1, round(height * height_fraction))
    center_x = (target["x_min"] + target["x_max"]) / 2
    center_y = (target["y_min"] + target["y_max"]) / 2
    x_min = max(0, min(width - box_width, round(center_x - box_width / 2)))
    y_min = max(0, min(height - box_height, round(center_y - box_height / 2)))
    occlusion = {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_min + box_width,
        "y_max": y_min + box_height,
    }

    output = image.convert("RGBA")
    draw = ImageDraw.Draw(output)
    rng_color = tuple(rng.randint(35, 115) for _ in range(3))
    draw.rectangle(
        (x_min, y_min, x_min + box_width - 1, y_min + box_height - 1),
        fill=(*rng_color, 255),
        outline=(30, 30, 30, 255),
        width=max(1, min(width, height) // 500),
    )
    transformed = copy.deepcopy(annotation)
    _get_polygons(transformed)
    difficulty = transformed.get("scene_difficulty", {})
    _ensure_transform_metadata(transformed, difficulty.get("profile", "D2_occlusion"), seed)
    visibility_thresholds = difficulty.get(
        "visibility_thresholds", {"visible": 0.99, "partial": 0.01}
    )
    _append_transformation(
        transformed,
        {
            "type": "occlusion",
            "seed": seed,
            "input_size": [width, height],
            "output_size": [width, height],
            "occlusion_region": occlusion,
            "color_rgb": list(rng_color),
            "policy": "opaque_rectangular_foreground",
            "visibility_policy": "bbox_unoccluded_area_fraction",
        },
    )

    for region in transformed["text_regions"]:
        bbox = region["bbox"]
        area = max(0.0, bbox["x_max"] - bbox["x_min"]) * max(
            0.0, bbox["y_max"] - bbox["y_min"]
        )
        if area <= 0:
            raise ValueError("Cannot estimate visibility for degenerate text bounding box.")
        overlap_width = max(
            0.0,
            min(bbox["x_max"], occlusion["x_max"])
            - max(bbox["x_min"], occlusion["x_min"]),
        )
        overlap_height = max(
            0.0,
            min(bbox["y_max"], occlusion["y_max"])
            - max(bbox["y_min"], occlusion["y_min"]),
        )
        visible_fraction = max(0.0, min(1.0, 1.0 - overlap_width * overlap_height / area))
        status = _visibility_status(visible_fraction, visibility_thresholds)
        region["visibility"] = {
            "estimated_fraction": visible_fraction,
            "status": status,
            "policy": "bbox_unoccluded_area_fraction",
            "note": "Estimate uses axis-aligned text bbox area, not glyph-mask pixels.",
        }
    return output.convert("RGB"), transformed


def apply_profile(
    image: Image.Image,
    annotation: dict,
    profile: str,
    seed: int,
    config: Optional[dict] = None,
) -> Tuple[Image.Image, dict]:
    if config is None:
        config = load_config()
    difficulty = config["shop_board_generation"]["scene_difficulty"]
    if profile not in difficulty["profiles"]:
        raise ValueError(f"Unknown scene difficulty profile '{profile}'.")
    transformed = copy.deepcopy(annotation)
    transformed["scene_difficulty"] = {
        "profile": profile,
        "random_seed": seed,
        "transformations": [],
        "visibility_policy": difficulty["visibility_policy"],
        "visibility_thresholds": dict(difficulty["visibility_thresholds"]),
    }
    _set_full_visibility(transformed)
    output = image.copy()
    rng = random.Random(seed)

    if profile == "D0_clean":
        _append_transformation(
            transformed,
            {"type": "identity", "seed": seed, "geometry_changed": False},
        )
        return output, transformed
    if profile == "D2_distance":
        low, high = difficulty["distance_scale_range"]
        scale = rng.uniform(low, high)
        return apply_distance_scale(output, transformed, scale, seed=seed)
    if profile == "D2_perspective":
        low, high = difficulty["perspective_inset_fraction_range"]
        inset = rng.uniform(low, high)
        matrix, source, destination = make_seeded_perspective_homography(
            output.width, output.height, inset, seed
        )
        output, transformed = apply_perspective(output, transformed, matrix, seed=seed)
        transformed["scene_difficulty"]["transformations"][-1].update(
            {
                "inset_fraction": inset,
                "source_corners": source,
                "destination_corners": destination,
            }
        )
        return output, transformed
    if profile == "D2_lighting":
        brightness = rng.uniform(*difficulty["brightness_factor_range"])
        contrast = rng.uniform(*difficulty["contrast_factor_range"])
        opacity = rng.uniform(*difficulty["shadow_opacity_range"])
        sigma = rng.uniform(*difficulty["shadow_blur_sigma_range"])
        return apply_lighting(
            output,
            transformed,
            brightness,
            contrast,
            shadow_opacity=opacity,
            shadow_sigma=sigma,
            seed=seed,
        )
    if profile == "D2_blur":
        sigma = rng.uniform(*difficulty["gaussian_blur_sigma_range"])
        downsample = rng.uniform(*difficulty["degradation_scale_range"])
        return apply_blur_degradation(
            output,
            transformed,
            sigma,
            downsample_scale=downsample,
            seed=seed,
        )
    if profile == "D2_occlusion":
        width_fraction = rng.uniform(*difficulty["occlusion_width_fraction_range"])
        height_fraction = rng.uniform(*difficulty["occlusion_height_fraction_range"])
        return apply_occlusion(
            output,
            transformed,
            seed=seed,
            width_fraction=width_fraction,
            height_fraction=height_fraction,
        )
    raise ValueError(f"Profile '{profile}' does not have an implementation.")
