from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List

from PIL import Image, ImageChops

from .background_composer import (
    validate_background_composition,
    validate_transformed_surface,
)
from .annotation_exporter import (
    validate_annotation,
    validate_rendered_geometry,
    validate_transformed_geometry,
    validate_visibility_metadata,
)


def validate_dataset(image_dir: Path, annotation_dir: Path) -> dict:
    image_paths = sorted(
        path for path in image_dir.glob("*.png") if not path.stem.endswith("_board")
    )
    annotation_paths = sorted(annotation_dir.glob("*.json"))
    errors: List[dict] = []
    records: List[dict] = []
    annotations_by_image: dict[str, Path] = {}
    seen_sample_ids = set()

    for annotation_path in annotation_paths:
        try:
            with annotation_path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            errors.append({"file": annotation_path.name, "error": f"Invalid annotation JSON: {error}"})
            continue
        if not isinstance(record, dict):
            errors.append({"file": annotation_path.name, "error": "Annotation root must be a JSON object."})
            continue

        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append({"file": annotation_path.name, "error": "Missing sample_id."})
        elif sample_id in seen_sample_ids:
            errors.append({"file": annotation_path.name, "error": f"Duplicate sample_id: {sample_id}"})
        else:
            seen_sample_ids.add(sample_id)
        if annotation_path.stem != sample_id:
            errors.append({"file": annotation_path.name, "error": "Annotation filename does not match sample_id."})

        image_name = record.get("image")
        if not isinstance(image_name, str):
            errors.append({"file": annotation_path.name, "error": "Missing image filename."})
            continue
        if image_name in annotations_by_image:
            errors.append({"file": annotation_path.name, "error": f"Duplicate annotation for image {image_name}."})
        annotations_by_image[image_name] = annotation_path
        image_path = image_dir / image_name
        for error in validate_annotation(record, image_path):
            errors.append({"file": annotation_path.name, "error": error})
        for error in validate_rendered_geometry(record):
            errors.append({"file": annotation_path.name, "error": error})
        if record.get("background_composition") is not None:
            for error in validate_background_composition(record):
                errors.append({"file": annotation_path.name, "error": error})
        if record.get("scene_difficulty") is not None:
            for error in validate_visibility_metadata(record):
                errors.append({"file": annotation_path.name, "error": error})
            parent = record.get("parent", {})
            parent_image = parent.get("image")
            if not isinstance(parent_image, str) or not (image_dir / parent_image).is_file():
                errors.append(
                    {"file": annotation_path.name, "error": "Transformed sample has a missing parent image."}
                )
            parent_annotation = parent.get("annotation")
            if not isinstance(parent_annotation, str):
                errors.append(
                    {"file": annotation_path.name, "error": "Transformed sample has no parent annotation link."}
                )
            else:
                parent_path = annotation_dir / parent_annotation
                try:
                    with parent_path.open("r", encoding="utf-8") as handle:
                        parent_record = json.load(handle)
                    if parent.get("sample_id") != parent_record.get("sample_id"):
                        errors.append(
                            {
                                "file": annotation_path.name,
                                "error": "Parent annotation sample_id does not match the recorded parent link.",
                            }
                        )
                    for error in validate_transformed_geometry(parent_record, record):
                        errors.append({"file": annotation_path.name, "error": error})
                    if record.get("background_composition") is not None:
                        for error in validate_transformed_surface(parent_record, record):
                            errors.append({"file": annotation_path.name, "error": error})
                except (OSError, json.JSONDecodeError) as error:
                    errors.append(
                        {"file": annotation_path.name, "error": f"Cannot read parent annotation: {error}"}
                    )

        if record.get("validation", {}).get("status") != "passed":
            errors.append({"file": annotation_path.name, "error": "Sample is not marked validation status passed."})
        if (
            image_path.is_file()
            and record.get("scene_transform") is not None
            and record.get("scene_difficulty") is None
        ):
            board_path = image_dir / record.get("board_layer", "")
            try:
                with Image.open(image_path) as scene, Image.open(board_path) as board:
                    dx = record["scene_transform"]["translate_x"]
                    dy = record["scene_transform"]["translate_y"]
                    region = scene.crop((dx, dy, dx + board.width, dy + board.height))
                    if ImageChops.difference(region.convert("RGBA"), board.convert("RGBA")).getbbox():
                        errors.append(
                            {
                                "file": annotation_path.name,
                                "error": "Composited scene board pixels differ from the retained board layer.",
                            }
                        )
            except (OSError, KeyError, TypeError) as error:
                errors.append({"file": annotation_path.name, "error": f"Cannot compare board layer: {error}"})
        records.append(record)

    image_names = {path.name for path in image_paths}
    annotated_names = set(annotations_by_image)
    for image_name in sorted(image_names - annotated_names):
        errors.append({"file": image_name, "error": "Image has no corresponding annotation."})
    for image_name in sorted(annotated_names - image_names):
        errors.append({"file": image_name, "error": "Annotation references a missing image."})

    language_counts = Counter(
        record.get("language", "unknown") for record in records
    )
    category_counts = Counter(
        record.get("category", "unknown") for record in records
    )
    background_counts = Counter(
        record.get("background_composition", {}).get("source_type", "unknown")
        for record in records
        if record.get("background_composition") is not None
    )
    scene_category_counts = Counter(
        record.get("background_composition", {}).get("scene_category", "unknown")
        for record in records
        if record.get("background_composition") is not None
    )
    difficulty_counts = Counter(
        record.get("scene_difficulty", {}).get("profile", "unknown")
        for record in records
        if record.get("scene_difficulty") is not None
    )
    font_counts = Counter(
        region.get("font", {}).get("id", region.get("font", {}).get("family", "unknown"))
        for record in records
        for region in record.get("text_regions", [])
    )
    font_script_counts = Counter(
        (
            region.get("font", {}).get("id", region.get("font", {}).get("family", "unknown")),
            region.get("script", "unknown"),
        )
        for record in records
        for region in record.get("text_regions", [])
    )
    background_source_type_counts = Counter(
        record.get("background_source_type")
        or record.get("background_composition", {}).get("source_type", "unknown")
        for record in records
        if record.get("background_source_type") is not None
        or record.get("background_composition") is not None
    )
    return {
        "schema_version": "1.0",
        "image_dir": str(image_dir),
        "annotation_dir": str(annotation_dir),
        "image_count": len(image_paths),
        "annotation_count": len(annotation_paths),
        "sample_id_count": len(seen_sample_ids),
        "language_counts": dict(sorted(language_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "background_source_counts": dict(sorted(background_counts.items())),
        "background_source_type_counts": dict(sorted(background_source_type_counts.items())),
        "background_scene_category_counts": dict(sorted(scene_category_counts.items())),
        "difficulty_profile_counts": dict(sorted(difficulty_counts.items())),
        "font_counts": dict(sorted(font_counts.items())),
        "font_script_counts": {
            f"{font_id}|{script}": count
            for (font_id, script), count in sorted(font_script_counts.items())
        },
        "passed": not errors,
        "errors": errors,
    }
