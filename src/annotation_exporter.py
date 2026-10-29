from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

from PIL import Image

from .content_engine import CATEGORY_NAMES, SUPPORTED_LANGUAGES
from .text_renderer import render_multiline_text

ROOT_DIR = Path(__file__).resolve().parents[1]

SCRIPT_BY_LANGUAGE = {
    "marathi": "Devanagari",
    "hindi": "Devanagari",
    "gujarati": "Gujarati",
    "bengali": "Bengali",
}


def validate_annotation(annotation: dict, image_path: Path) -> List[str]:
    errors: List[str] = []
    image = None
    try:
        image = Image.open(image_path)
        image.verify()
    except (OSError, ValueError) as error:
        errors.append(f"Image is missing or unreadable: {error}")
    finally:
        if image is not None:
            image.close()

    try:
        with Image.open(image_path) as checked_image:
            image_width, image_height = checked_image.size
    except (OSError, ValueError):
        return errors

    if not isinstance(annotation.get("sample_id"), str) or not annotation["sample_id"]:
        errors.append("sample_id must be a non-empty string.")
    if annotation.get("image") != image_path.name:
        errors.append("image filename does not match the annotation image file.")
    if annotation.get("image_width") != image_width or annotation.get("image_height") != image_height:
        errors.append("annotation image dimensions do not match the image.")
    if annotation.get("language") not in SUPPORTED_LANGUAGES:
        errors.append(f"Unsupported language label: {annotation.get('language')!r}.")
    elif annotation.get("script") != SCRIPT_BY_LANGUAGE[annotation["language"]]:
        errors.append("script does not match the language label.")
    if annotation.get("category") not in CATEGORY_NAMES:
        errors.append(f"Unsupported category label: {annotation.get('category')!r}.")
    if not isinstance(annotation.get("random_seed"), int):
        errors.append("random_seed must be an integer.")

    regions = annotation.get("text_regions")
    if not isinstance(regions, list) or not regions:
        errors.append("text_regions must be a non-empty list.")
        return errors

    reading_orders = []
    for index, region in enumerate(regions):
        prefix = f"text_regions[{index}]"
        transcription = region.get("transcription")
        if not isinstance(transcription, str) or not transcription.strip():
            errors.append(f"{prefix}.transcription must be a non-empty string.")
        elif any(0xD800 <= ord(char) <= 0xDFFF for char in transcription):
            errors.append(f"{prefix}.transcription contains an invalid Unicode surrogate.")
        elif any(ord(char) < 32 and char not in {"\n", "\t"} for char in transcription):
            errors.append(f"{prefix}.transcription contains an unsupported control character.")
        if region.get("language") not in SUPPORTED_LANGUAGES:
            errors.append(f"{prefix} has an unsupported language label.")
        elif region.get("script") != SCRIPT_BY_LANGUAGE[region["language"]]:
            errors.append(f"{prefix}.script does not match its language.")

        reading_order = region.get("reading_order")
        if not isinstance(reading_order, int):
            errors.append(f"{prefix}.reading_order must be an integer.")
        else:
            reading_orders.append(reading_order)

        bbox = region.get("bbox")
        if not isinstance(bbox, dict):
            errors.append(f"{prefix}.bbox must be an object.")
            continue
        try:
            x_min, y_min, x_max, y_max = (
                float(bbox[key]) for key in ("x_min", "y_min", "x_max", "y_max")
            )
        except (KeyError, TypeError, ValueError):
            errors.append(f"{prefix}.bbox must contain finite numeric edge coordinates.")
            continue
        if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
            errors.append(f"{prefix}.bbox coordinates must be finite.")
        elif x_min < 0 or y_min < 0 or x_max > image_width or y_max > image_height:
            errors.append(f"{prefix}.bbox lies outside image bounds.")
        elif x_max <= x_min or y_max <= y_min:
            errors.append(f"{prefix}.bbox must have positive width and height.")

        polygon = region.get("polygon")
        if not isinstance(polygon, list) or len(polygon) != 4:
            errors.append(f"{prefix}.polygon must contain four points.")
        else:
            try:
                coordinates = [(float(point[0]), float(point[1])) for point in polygon]
                if not all(math.isfinite(value) for point in coordinates for value in point):
                    raise ValueError
                if any(
                    x < 0 or y < 0 or x > image_width or y > image_height
                    for x, y in coordinates
                ):
                    errors.append(f"{prefix}.polygon lies outside image bounds.")
            except (TypeError, ValueError, IndexError):
                errors.append(f"{prefix}.polygon must contain finite numeric x/y coordinates.")
            else:
                polygon_x = [point[0] for point in coordinates]
                polygon_y = [point[1] for point in coordinates]
                epsilon = 1e-5
                if (
                    abs(min(polygon_x) - x_min) > epsilon
                    or abs(max(polygon_x) - x_max) > epsilon
                    or abs(min(polygon_y) - y_min) > epsilon
                    or abs(max(polygon_y) - y_max) > epsilon
                ):
                    errors.append(f"{prefix}.bbox does not enclose its polygon bounds.")
                twice_area = abs(
                    sum(
                        coordinates[index][0] * coordinates[(index + 1) % 4][1]
                        - coordinates[(index + 1) % 4][0] * coordinates[index][1]
                        for index in range(4)
                    )
                )
                if twice_area <= epsilon:
                    errors.append(f"{prefix}.polygon is degenerate.")
                if any(
                    region.get("bbox", {}).get(key) is not None
                    and abs(float(region["bbox"][key]) - expected) > epsilon
                    for key, expected in (
                        ("x_min", min(polygon_x)),
                        ("x_max", max(polygon_x)),
                        ("y_min", min(polygon_y)),
                        ("y_max", max(polygon_y)),
                    )
                ):
                    errors.append(f"{prefix}.bbox is not the polygon's enclosing axis-aligned box.")

        if not isinstance(region.get("font"), dict) or not region["font"].get("family"):
            errors.append(f"{prefix}.font must include its family.")
        elif region["font"].get("id") is not None:
            font = region["font"]
            if (
                not isinstance(font.get("path"), str)
                or not font["path"]
                or not isinstance(font.get("style"), str)
                or not font["style"]
                or not isinstance(font.get("weight"), int)
            ):
                errors.append(f"{prefix}.font registry metadata is incomplete.")

    background_source_type = annotation.get("background_source_type")
    if background_source_type is not None and background_source_type not in {
        "procedural_illustration",
        "real_photo",
        "ai_generated",
    }:
        errors.append("background_source_type is not a supported provenance label.")
    composition = annotation.get("background_composition")
    if (
        isinstance(composition, dict)
        and background_source_type is not None
        and background_source_type != composition.get("source_type")
    ):
        errors.append("background_source_type does not match background_composition.source_type.")

    if sorted(reading_orders) != list(range(len(regions))):
        errors.append("reading_order values must be unique and contiguous from zero.")

    scene_transform = annotation.get("scene_transform")
    if scene_transform is not None:
        board_layer = annotation.get("board_layer")
        if not isinstance(board_layer, str) or not (image_path.parent / board_layer).is_file():
            errors.append("scene annotation is missing its separate board_layer image.")
        if annotation.get("scene_difficulty") is not None:
            pass
        elif not isinstance(scene_transform, dict):
            errors.append("scene_transform must be an object.")
        else:
            dx, dy = scene_transform.get("translate_x"), scene_transform.get("translate_y")
            if not isinstance(dx, int) or not isinstance(dy, int):
                errors.append("scene translation must contain integer offsets.")
            else:
                for index, region in enumerate(regions):
                    source_bbox = region.get("board_bbox")
                    bbox = region.get("bbox")
                    if not isinstance(source_bbox, dict) or not isinstance(bbox, dict):
                        errors.append(f"text_regions[{index}] is missing source board geometry.")
                        continue
                    if any(
                        bbox.get(key) != source_bbox.get(key) + offset
                        for key, offset in (("x_min", dx), ("x_max", dx), ("y_min", dy), ("y_max", dy))
                    ):
                        errors.append(f"text_regions[{index}] does not match the recorded scene translation.")

    if annotation.get("scene_difficulty") is not None:
        difficulty_errors = validate_visibility_metadata(annotation)
        errors.extend(difficulty_errors)
    return errors


def validate_visibility_metadata(annotation: dict) -> List[str]:
    errors = []
    difficulty = annotation.get("scene_difficulty")
    if not isinstance(difficulty, dict):
        return ["scene_difficulty must be an object."]
    if not isinstance(difficulty.get("transformations"), list):
        errors.append("scene_difficulty.transformations must be a list.")
    if difficulty.get("visibility_policy") != "bbox_unoccluded_area_fraction":
        errors.append("Unsupported or missing text visibility policy.")
    thresholds = difficulty.get("visibility_thresholds", {"visible": 0.99, "partial": 0.01})
    if (
        not isinstance(thresholds, dict)
        or not isinstance(thresholds.get("visible"), (int, float))
        or not isinstance(thresholds.get("partial"), (int, float))
        or not 0 <= thresholds.get("partial", -1) < thresholds.get("visible", -1) <= 1
    ):
        errors.append("Visibility thresholds must satisfy 0 <= partial < visible <= 1.")
        thresholds = {"visible": 0.99, "partial": 0.01}

    occlusion = next(
        (
            transformation.get("occlusion_region")
            for transformation in difficulty.get("transformations", [])
            if transformation.get("type") == "occlusion"
        ),
        None,
    )
    for index, region in enumerate(annotation.get("text_regions", [])):
        visibility = region.get("visibility")
        if not isinstance(visibility, dict):
            errors.append(f"text_regions[{index}].visibility is missing.")
            continue
        if occlusion is None:
            expected_fraction = 1.0
        else:
            bbox = region.get("bbox", {})
            area = max(0.0, bbox.get("x_max", 0) - bbox.get("x_min", 0)) * max(
                0.0, bbox.get("y_max", 0) - bbox.get("y_min", 0)
            )
            if area <= 0:
                errors.append(f"text_regions[{index}] cannot estimate visibility for a degenerate bbox.")
                continue
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
            expected_fraction = max(0.0, min(1.0, 1.0 - overlap_width * overlap_height / area))
        if not isinstance(visibility.get("estimated_fraction"), (int, float)):
            errors.append(f"text_regions[{index}].visibility fraction must be numeric.")
            continue
        if abs(visibility["estimated_fraction"] - expected_fraction) > 1e-5:
            errors.append(f"text_regions[{index}].visibility fraction does not match the policy.")
        expected_status = (
            "visible"
            if expected_fraction >= thresholds["visible"]
            else "partially_occluded"
            if expected_fraction >= thresholds["partial"]
            else "fully_occluded"
        )
        if visibility.get("status") != expected_status:
            errors.append(f"text_regions[{index}].visibility status does not match its fraction.")
        if visibility.get("policy") != "bbox_unoccluded_area_fraction":
            errors.append(f"text_regions[{index}].visibility policy is missing or inconsistent.")
    return errors


def write_annotation(annotation: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(annotation, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def validate_transformed_geometry(parent: dict, annotation: dict) -> List[str]:
    errors: List[str] = []
    difficulty = annotation.get("scene_difficulty", {})
    transformations = difficulty.get("transformations")
    if not isinstance(transformations, list) or not transformations:
        return ["Transformed sample must record at least one transformation."]
    if (
        not isinstance(difficulty.get("random_seed"), int)
        or annotation.get("random_seed") != difficulty.get("random_seed")
    ):
        errors.append("Sample and scene-difficulty random seeds do not match.")
    if (
        annotation.get("language") != parent.get("language")
        or annotation.get("script") != parent.get("script")
        or annotation.get("category") != parent.get("category")
    ):
        errors.append("Transformed sample language, script, or category differs from its parent.")
    if len(annotation.get("text_regions", [])) != len(parent.get("text_regions", [])):
        return errors + ["Transformed sample changed the parent text-region count."]

    parent_regions = {
        region.get("region_id"): region for region in parent.get("text_regions", [])
    }
    variant_regions = {
        region.get("region_id"): region for region in annotation.get("text_regions", [])
    }
    if parent_regions.keys() != variant_regions.keys():
        return errors + ["Transformed sample text-region IDs differ from its parent."]
    for region_id, parent_region in parent_regions.items():
        transformed_region = variant_regions[region_id]
        for field in ("transcription", "language", "script", "reading_order"):
            if transformed_region.get(field) != parent_region.get(field):
                errors.append(
                    f"Transformed text region {region_id!r} changed its {field}."
                )

    polygons = {
        region_id: [[float(point[0]), float(point[1])] for point in region["polygon"]]
        for region_id, region in parent_regions.items()
    }
    width = parent.get("image_width")
    height = parent.get("image_height")
    if not isinstance(width, int) or not isinstance(height, int):
        return errors + ["Parent annotation dimensions are invalid."]

    expected_types = {
        "D0_clean": "identity",
        "D2_distance": "distance_downscale",
        "D2_perspective": "perspective",
        "D2_lighting": "lighting",
        "D2_blur": "blur_degradation",
        "D2_occlusion": "occlusion",
    }
    expected_type = expected_types.get(difficulty.get("profile"))
    if expected_type is None or len(transformations) != 1 or transformations[0].get("type") != expected_type:
        errors.append("Difficulty profile must contain exactly its configured independent transform.")

    for index, transformation in enumerate(transformations):
        if transformation.get("seed") != difficulty.get("random_seed"):
            errors.append(f"Transformation {index} seed does not match the profile seed.")
        if transformation.get("type") == "distance_downscale":
            if transformation.get("input_size") != [width, height]:
                errors.append("Distance transform input dimensions do not match its parent image.")
            scale_x = transformation.get("scale_x")
            scale_y = transformation.get("scale_y")
            if not isinstance(scale_x, (int, float)) or not isinstance(scale_y, (int, float)):
                errors.append("Distance transform is missing actual scale factors.")
                continue
            polygons = {
                region_id: [[x * scale_x, y * scale_y] for x, y in points]
                for region_id, points in polygons.items()
            }
            output_size = transformation.get("output_size")
            if not isinstance(output_size, list) or len(output_size) != 2:
                errors.append("Distance transform is missing output dimensions.")
                continue
            if (
                output_size[0] != round(width * scale_x)
                or output_size[1] != round(height * scale_y)
            ):
                errors.append("Distance transform output dimensions do not match its scale factors.")
            width, height = output_size
        elif transformation.get("type") == "perspective":
            if transformation.get("input_size") != [width, height]:
                errors.append("Perspective transform input dimensions do not match the current image.")
            if transformation.get("output_size") != [width, height]:
                errors.append("Perspective transform must preserve the image dimensions.")
            matrix = transformation.get("homography")
            if (
                not isinstance(matrix, list)
                or len(matrix) != 3
                or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
            ):
                errors.append("Perspective transform has an invalid homography.")
                continue
            mapped_polygons = {}
            for region_id, points in polygons.items():
                mapped_points = []
                for x, y in points:
                    hx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]
                    hy = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]
                    hw = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
                    if not math.isfinite(hw) or abs(hw) < 1e-10:
                        errors.append("Perspective homography maps geometry to infinity.")
                        continue
                    mapped_points.append([hx / hw, hy / hw])
                if len(mapped_points) == 4:
                    mapped_polygons[region_id] = mapped_points
            polygons = mapped_polygons
        elif transformation.get("type") in {"lighting", "blur_degradation", "occlusion"}:
            if (
                transformation.get("input_size") != [width, height]
                or transformation.get("output_size") != [width, height]
                or transformation.get("geometry_changed") is True
            ):
                errors.append(
                    f"{transformation.get('type')} must preserve dimensions and annotation geometry."
                )
        elif transformation.get("type") != "identity":
            errors.append(f"Unknown scene transformation type: {transformation.get('type')!r}.")

    if width != annotation.get("image_width") or height != annotation.get("image_height"):
        errors.append("Final image dimensions do not match the replayed transformation dimensions.")
    for region_id, points in polygons.items():
        if len(points) != 4:
            errors.append(f"Region {region_id} transformed polygon is not a quadrilateral.")
            continue
        crosses = []
        for point_index in range(4):
            first = points[point_index]
            second = points[(point_index + 1) % 4]
            third = points[(point_index + 2) % 4]
            crosses.append(
                (second[0] - first[0]) * (third[1] - second[1])
                - (second[1] - first[1]) * (third[0] - second[0])
            )
        if any(abs(value) < 1e-8 for value in crosses) or not (
            all(value > 0 for value in crosses) or all(value < 0 for value in crosses)
        ):
            errors.append(f"Region {region_id} transformed polygon is degenerate or non-convex.")
        region = variant_regions[region_id]
        actual_points = region.get("polygon", [])
        if len(actual_points) != len(points):
            errors.append(f"Region {region_id} polygon does not match replayed transformation.")
            continue
        if any(
            not math.isfinite(float(actual[axis]))
            or abs(float(actual[axis]) - expected[axis]) > 1e-4
            for actual, expected in zip(actual_points, points)
            for axis in (0, 1)
        ):
            errors.append(f"Region {region_id} polygon does not match replayed transformation.")
            continue
        bbox = region.get("bbox", {})
        expected_bbox = {
            "x_min": min(point[0] for point in points),
            "y_min": min(point[1] for point in points),
            "x_max": max(point[0] for point in points),
            "y_max": max(point[1] for point in points),
        }
        if any(abs(float(bbox.get(key, float("nan"))) - value) > 1e-4 for key, value in expected_bbox.items()):
            errors.append(f"Region {region_id} bbox does not match its replayed polygon.")

        source_region = parent_regions[region_id]
        if (
            region.get("transcription") != source_region.get("transcription")
            or region.get("language") != source_region.get("language")
            or region.get("script") != source_region.get("script")
        ):
            errors.append(f"Region {region_id} transcription or language differs from its parent.")
        if any(
            point[0] < -1e-5
            or point[0] > annotation["image_width"] + 1e-5
            or point[1] < -1e-5
            or point[1] > annotation["image_height"] + 1e-5
            for point in points
        ):
            errors.append(f"Region {region_id} transformed polygon lies outside the image.")
    return errors


def validate_rendered_geometry(annotation: dict) -> List[str]:
    errors: List[str] = []
    generation = annotation.get("generation_parameters", {})
    layouts = generation.get("region_layout", [])
    layouts_by_id = {layout.get("region_id"): layout for layout in layouts}
    foreground = tuple(generation.get("text_rgb", []))
    for index, region in enumerate(annotation.get("text_regions", [])):
        layout = layouts_by_id.get(region.get("region_id"))
        if not isinstance(layout, dict):
            errors.append(f"text_regions[{index}] has no corresponding rendering layout.")
            continue
        try:
            font_path = region["font"]["path"]
            if not Path(font_path).is_absolute():
                font_path = str(ROOT_DIR / font_path)
            rendered = render_multiline_text(
                region["transcription"],
                region["language"],
                width=layout["canvas_width"],
                height=layout["canvas_height"],
                font_size=layout["font_size_px"],
                foreground=foreground,
                background=None,
                padding=layout["padding_px"],
                line_spacing=layout["line_spacing"],
                align=layout.get("alignment", generation["alignment"]),
                shadow_offset=(0, 0),
                font_path=font_path,
            )
        except (KeyError, TypeError, ValueError, OSError) as error:
            errors.append(f"text_regions[{index}] could not be rerendered: {error}")
            continue

        if rendered["text"] != region["transcription"]:
            errors.append(f"text_regions[{index}] transcription changed during rendering.")
        ink = rendered["ink_bbox"]
        origin_x, origin_y = layout["canvas_x"], layout["canvas_y"]
        expected = {
            "x_min": origin_x + ink["x_min"],
            "y_min": origin_y + ink["y_min"],
            "x_max": origin_x + ink["x_max"],
            "y_max": origin_y + ink["y_max"],
        }
        source_bbox = region.get("board_bbox", region.get("bbox"))
        if source_bbox != expected:
            errors.append(f"text_regions[{index}] bbox does not match the rendered glyph-mask geometry.")
    return errors
