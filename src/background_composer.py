from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .background_assets import (
    BackgroundAsset,
    load_background_image,
    polygon_bbox,
    polygon_is_in_frame,
    transform_polygon,
)
from .signboard_generator import render_shop_board_sample


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


def _validate_quad(points: List[List[float]], width: int, height: int, label: str) -> None:
    try:
        polygon = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must contain numeric x/y coordinates.") from error
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError(f"{label} must contain four finite x/y points.")
    if not polygon_is_in_frame(points, width, height):
        raise ValueError(f"{label} is degenerate or lies outside image boundaries.")
    if not cv2.isContourConvex(polygon.astype(np.float32)):
        raise ValueError(f"{label} must be convex.")
    signed_area = sum(
        polygon[index, 0] * polygon[(index + 1) % 4, 1]
        - polygon[(index + 1) % 4, 0] * polygon[index, 1]
        for index in range(4)
    ) / 2
    if abs(signed_area) < 1:
        raise ValueError(f"{label} has negligible area.")
    if signed_area < 0:
        raise ValueError(
            f"{label} must use top-left, top-right, bottom-right, bottom-left order."
        )


def _warp_rgba(image: Image.Image, matrix: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    source = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = source[:, :, 3:4] / 255.0
    premultiplied = source[:, :, :3] * alpha
    cv_matrix = _edge_homography_to_cv(matrix)
    output_width, output_height = size
    warped_alpha = cv2.warpPerspective(
        alpha[:, :, 0],
        cv_matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_rgb = cv2.warpPerspective(
        premultiplied,
        cv_matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rgb = np.zeros_like(warped_rgb)
    visible = warped_alpha > 1e-6
    rgb[visible] = warped_rgb[visible] / warped_alpha[visible, None]
    rgba = np.dstack((np.clip(rgb, 0, 255), np.clip(warped_alpha * 255, 0, 255)))
    return Image.fromarray(np.rint(rgba).astype(np.uint8), mode="RGBA")


def _validate_transformed_text_geometry(annotation: dict, width: int, height: int) -> None:
    for index, region in enumerate(annotation.get("text_regions", [])):
        polygon = region.get("polygon")
        _validate_quad(polygon, width, height, f"text_regions[{index}].polygon")
        expected = polygon_bbox(polygon)
        if any(abs(float(region["bbox"][key]) - expected[key]) > 1e-5 for key in expected):
            raise ValueError(f"text_regions[{index}].bbox does not match the transformed polygon.")


def compose_background_scene(
    asset: BackgroundAsset,
    *,
    language: str,
    sample_seed: int,
    sample_id: str,
    board_seed: int,
    layout_id: str,
    layer_relative_path: str,
    minimum_surface_width: int = 1,
    minimum_surface_height: int = 1,
    font_path: Optional[str] = None,
    font_id: Optional[str] = None,
    font_weight: Optional[int] = None,
) -> Tuple[Image.Image, dict, Image.Image]:
    record = asset.record
    width, height = record["image_width"], record["image_height"]
    background = load_background_image(asset.path, (width, height))
    surface = record.get("signboard_surface_polygon")
    _validate_quad(surface, width, height, "signboard_surface_polygon")

    corners = np.asarray(surface, dtype=np.float32)
    edge_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    surface_width = float((edge_lengths[0] + edge_lengths[2]) / 2)
    surface_height = float((edge_lengths[1] + edge_lengths[3]) / 2)
    if surface_width < minimum_surface_width or surface_height < minimum_surface_height:
        raise ValueError(
            "Verified signboard surface is below the configured minimum dimensions "
            f"({surface_width:.1f}x{surface_height:.1f}px)."
        )

    content_category = (
        "shop_board" if record["scene_category"] == "shopfront" else "advertising_hoarding"
    )
    board, board_annotation = render_shop_board_sample(
        language,
        board_seed,
        sample_id,
        layout_id,
        content_category=content_category,
        font_path=font_path,
        font_id=font_id,
        font_weight=font_weight,
    )
    board_width, board_height = board.size
    source_corners = np.asarray(
        [[0, 0], [board_width, 0], [board_width, board_height], [0, board_height]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source_corners, corners)
    if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-10:
        raise ValueError("Signboard-to-surface homography is invalid or singular.")
    warped_board = _warp_rgba(board, homography, (width, height))
    scene = background.convert("RGBA")
    scene.alpha_composite(warped_board)

    annotation = copy.deepcopy(board_annotation)
    annotation["sample_id"] = sample_id
    annotation["image"] = f"{sample_id}.png"
    annotation["image_width"] = width
    annotation["image_height"] = height
    annotation["random_seed"] = sample_seed
    annotation["board_layer"] = layer_relative_path
    annotation["source_scene_category"] = record["scene_category"]
    for region in annotation["text_regions"]:
        region["board_bbox"] = dict(region["bbox"])
        region["board_polygon"] = copy.deepcopy(region["polygon"])
        region["polygon"] = transform_polygon(region["board_polygon"], homography)
        region["bbox"] = polygon_bbox(region["polygon"])
    annotation["signboard_surface_polygon"] = copy.deepcopy(surface)
    annotation["background_composition"] = {
        "background_id": record["background_id"],
        "source_type": record["source_type"],
        "scene_category": record["scene_category"],
        "relative_path": record["relative_path"],
        "original_filename": record["original_filename"],
        "provenance": record["provenance"],
        "licensing": copy.deepcopy(record["licensing"]),
        "generation_prompt": record.get("generation_prompt"),
        "created_date": record.get("created_date"),
        "asset_id": record.get("asset_id", record["background_id"]),
        "surface_verification_status": record["surface_verification_status"],
        "manual_review_status": record["manual_review_status"],
        "source_surface_polygon": copy.deepcopy(surface),
        "composed_surface_polygon": copy.deepcopy(surface),
        "surface_polygon": copy.deepcopy(surface),
        "board_layer": layer_relative_path,
        "board_size": [board_width, board_height],
        "board_corners": source_corners.astype(float).tolist(),
        "homography_coordinate_system": "pixel_edges",
        "board_to_background_homography": homography.tolist(),
        "background_size": [width, height],
        "random_seeds": {"sample": sample_seed, "board": board_seed},
        "clipping": "rejected_if_text_or_surface_geometry_is_out_of_frame",
    }
    annotation["generation_parameters"] = copy.deepcopy(annotation["generation_parameters"])
    annotation["generation_parameters"]["board_random_seed"] = board_seed
    annotation["generation_parameters"]["background_layout_seed"] = sample_seed
    _validate_transformed_text_geometry(annotation, width, height)
    if not polygon_is_in_frame(annotation["signboard_surface_polygon"], width, height):
        raise ValueError("Composed signboard surface is outside the final image.")
    annotation["validation"] = {
        "status": "pending",
        "warnings": ["visual_review_required"],
        "visual_review_required": True,
    }
    return scene.convert("RGB"), annotation, board


def update_composed_surface_for_transform(annotation: dict) -> None:
    composition = annotation.get("background_composition")
    if not isinstance(composition, dict):
        return
    surface = copy.deepcopy(composition["composed_surface_polygon"])
    difficulty = annotation.get("scene_difficulty", {})
    for transformation in difficulty.get("transformations", []):
        transform_type = transformation.get("type")
        if transform_type == "distance_downscale":
            surface = [
                [
                    point[0] * transformation["scale_x"],
                    point[1] * transformation["scale_y"],
                ]
                for point in surface
            ]
        elif transform_type == "perspective":
            surface = transform_polygon(surface, transformation["homography"])
    composition["surface_polygon"] = surface
    annotation["signboard_surface_polygon"] = copy.deepcopy(surface)


def validate_background_composition(annotation: dict) -> List[str]:
    errors: List[str] = []
    composition = annotation.get("background_composition")
    if not isinstance(composition, dict):
        return ["background_composition must be an object."]
    if not isinstance(composition.get("background_id"), str) or not composition["background_id"]:
        errors.append("background_composition.background_id must be a non-empty string.")
    source_type = composition.get("source_type")
    if not isinstance(source_type, str) or source_type not in {
        "real_photo",
        "ai_generated",
        "test_fixture",
    }:
        errors.append("background_composition.source_type is unsupported.")
    if not isinstance(composition.get("scene_category"), str) or composition.get(
        "scene_category"
    ) not in {"shopfront", "outdoor_hoarding"}:
        errors.append("background_composition.scene_category is unsupported.")
    if not isinstance(composition.get("asset_id"), str) or not composition["asset_id"]:
        errors.append("background asset_id is missing.")
    if not isinstance(composition.get("original_filename"), str) or not composition[
        "original_filename"
    ]:
        errors.append("original background filename is missing.")
    if not isinstance(composition.get("provenance"), str) or not composition["provenance"].strip():
        errors.append("background provenance is missing.")
    licensing = composition.get("licensing")
    if (
        not isinstance(licensing, dict)
        or licensing.get("verification_status") != "verified"
        or licensing.get("research_use_allowed") is not True
    ):
        errors.append("background license is not verified for research use.")
    if composition.get("manual_review_status") != "approved":
        errors.append("background asset has not passed manual review.")
    if composition.get("surface_verification_status") != "verified":
        errors.append("background signboard surface has not been verified.")
    if composition.get("homography_coordinate_system") != "pixel_edges":
        errors.append("background homography must use pixel-edge coordinates.")

    width, height = annotation.get("image_width"), annotation.get("image_height")
    polygon = annotation.get("signboard_surface_polygon")
    if not isinstance(width, int) or not isinstance(height, int) or not isinstance(polygon, list):
        errors.append("final signboard surface polygon or image dimensions are missing.")
        return errors
    try:
        _validate_quad(polygon, width, height, "signboard_surface_polygon")
    except (TypeError, ValueError, IndexError, OverflowError) as error:
        errors.append(str(error))
    if composition.get("surface_polygon") != polygon:
        errors.append("top-level signboard surface polygon differs from composition metadata.")

    try:
        matrix = np.asarray(composition["board_to_background_homography"], dtype=np.float64)
        board_corners = composition["board_corners"]
        composed_surface = transform_polygon(board_corners, matrix)
        source_surface = composition["source_surface_polygon"]
        recorded_surface = composition["composed_surface_polygon"]
        if not np.allclose(composed_surface, source_surface, atol=1e-4):
            errors.append("board-to-background homography does not map to the verified surface.")
        if not np.allclose(source_surface, recorded_surface, atol=1e-4):
            errors.append("recorded composed surface does not match the board homography.")
    except (KeyError, TypeError, ValueError):
        errors.append("background composition is missing valid source geometry or homography.")
    return errors


def validate_transformed_surface(parent: dict, annotation: dict) -> List[str]:
    parent_composition = parent.get("background_composition")
    current_composition = annotation.get("background_composition")
    if not isinstance(parent_composition, dict) or not isinstance(current_composition, dict):
        return ["Transformed background sample is missing parent composition metadata."]
    errors: List[str] = []
    if parent_composition.get("background_id") != current_composition.get("background_id"):
        errors.append("Transformed sample changed its parent background asset.")
    stable_fields = (
        "source_type",
        "scene_category",
        "relative_path",
        "original_filename",
        "provenance",
        "licensing",
        "generation_prompt",
        "created_date",
        "asset_id",
        "surface_verification_status",
        "manual_review_status",
        "source_surface_polygon",
        "composed_surface_polygon",
        "board_layer",
        "board_size",
        "board_corners",
        "homography_coordinate_system",
        "board_to_background_homography",
        "background_size",
    )
    for field in stable_fields:
        if parent_composition.get(field) != current_composition.get(field):
            errors.append(f"Transformed sample changed background composition field {field}.")
    expected = copy.deepcopy(parent_composition.get("surface_polygon"))
    if not isinstance(expected, list):
        return errors + ["Parent signboard surface polygon is missing."]
    difficulty = annotation.get("scene_difficulty", {})
    for transformation in difficulty.get("transformations", []):
        try:
            if transformation.get("type") == "distance_downscale":
                expected = [
                    [
                        point[0] * transformation["scale_x"],
                        point[1] * transformation["scale_y"],
                    ]
                    for point in expected
                ]
            elif transformation.get("type") == "perspective":
                expected = transform_polygon(expected, transformation["homography"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"Cannot replay signboard surface geometry: {error}")
            return errors
    actual = current_composition.get("surface_polygon")
    try:
        expected_array = np.asarray(expected, dtype=np.float64)
        actual_array = np.asarray(actual, dtype=np.float64)
        matches = (
            expected_array.shape == actual_array.shape
            and expected_array.shape == (4, 2)
            and np.isfinite(actual_array).all()
            and np.allclose(expected_array, actual_array, atol=1e-4)
        )
    except (TypeError, ValueError):
        matches = False
    if not matches:
        errors.append("Transformed signboard surface does not match the recorded homography.")
    if annotation.get("signboard_surface_polygon") != actual:
        errors.append("Transformed top-level signboard surface differs from its metadata.")
    return errors
