from __future__ import annotations

import copy
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont

from .annotation_exporter import (
    validate_annotation,
    validate_rendered_geometry,
    validate_transformed_geometry,
    write_annotation,
)
from .content_engine import load_config
from .scene_transformer import PROFILE_NAMES, apply_profile

ROOT_DIR = Path(__file__).resolve().parents[1]


def _draw_annotation_overlay(image: Image.Image, annotation: dict, output_path: Path) -> None:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    label_font = ImageFont.load_default()
    for region in annotation["text_regions"]:
        polygon = [(round(x), round(y)) for x, y in region["polygon"]]
        draw.line(polygon + [polygon[0]], fill=(255, 0, 0), width=3)
        bbox = region["bbox"]
        label = f"{region['region_id']} {region['visibility']['estimated_fraction']:.2f}"
        draw.text(
            (round(bbox["x_min"]), max(0, round(bbox["y_min"]) - 13)),
            label,
            fill=(255, 255, 0),
            font=label_font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path, format="PNG")


def _write_profile_contact_sheet(image_paths: List[Path], records: List[dict], output_path: Path) -> None:
    columns = 3
    cell_width, cell_height = 540, 430
    rows = (len(image_paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (image_path, record) in enumerate(zip(image_paths, records)):
        with Image.open(image_path) as source:
            sample = source.convert("RGB")
            sample.thumbnail((cell_width - 20, cell_height - 40), Image.Resampling.LANCZOS)
        cell_x = (index % columns) * cell_width
        cell_y = (index // columns) * cell_height
        x = cell_x + (cell_width - sample.width) // 2
        y = cell_y + 5
        sheet.paste(sample, (x, y))
        draw.text(
            (cell_x + 8, cell_y + cell_height - 25),
            f"{record['sample_id']} | {record['language']}",
            font=font,
            fill=(20, 20, 20),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")


def generate_difficulty_variants(
    source_image_dir: Path,
    source_annotation_dir: Path,
    output_root: Path,
    preview_root: Path,
    seed: int,
    profiles: tuple[str, ...] = PROFILE_NAMES,
) -> dict:
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if not profiles or any(profile not in PROFILE_NAMES for profile in profiles):
        raise ValueError("Requested profiles must be selected from PROFILE_NAMES.")
    annotation_paths = sorted(source_annotation_dir.glob("*.json"))
    if not annotation_paths:
        raise FileNotFoundError(f"No source annotations found in {source_annotation_dir}.")

    source_records = []
    seen_ids = set()
    for annotation_path in annotation_paths:
        with annotation_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("sample_id") in seen_ids:
            raise ValueError(f"Duplicate source sample ID: {record.get('sample_id')}")
        seen_ids.add(record.get("sample_id"))
        image_path = source_image_dir / record["image"]
        errors = validate_annotation(record, image_path)
        errors.extend(validate_rendered_geometry(record))
        if errors:
            raise ValueError(f"Invalid source sample {record.get('sample_id')}: {'; '.join(errors)}")
        source_records.append((record, image_path, annotation_path))

    config = load_config()
    report: Dict[str, dict] = {
        "schema_version": "1.0",
        "source_image_dir": str(source_image_dir),
        "source_annotation_dir": str(source_annotation_dir),
        "seed": seed,
        "visibility_policy": config["shop_board_generation"]["scene_difficulty"]["visibility_policy"],
        "profiles": {},
    }

    for profile_index, profile in enumerate(profiles):
        image_dir = output_root / profile / "images"
        annotation_dir = output_root / profile / "annotations"
        overlay_dir = preview_root / "overlays" / profile
        contact_sheet_path = preview_root / f"{profile}_contact_sheet.png"
        planned_paths = [
            path
            for record, _, _ in source_records
            for path in (
                image_dir / f"{record['sample_id']}__{profile}.png",
                annotation_dir / f"{record['sample_id']}__{profile}.json",
                overlay_dir / f"{record['sample_id']}__{profile}_overlay.png",
            )
        ] + [contact_sheet_path]
        existing = [path for path in planned_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing difficulty outputs: "
                + ", ".join(str(path) for path in existing)
            )
        image_dir.mkdir(parents=True, exist_ok=True)
        annotation_dir.mkdir(parents=True, exist_ok=True)
        generated_records = []
        generated_image_paths = []
        visibility_counts = Counter()

        for sample_index, (source_record, source_image_path, source_annotation_path) in enumerate(source_records):
            variant_seed = (
                random.Random(seed ^ (profile_index << 16) ^ sample_index).getrandbits(64)
            )
            sample_id = f"{source_record['sample_id']}__{profile}"
            with Image.open(source_image_path) as source_image:
                image = source_image.copy()
            source_annotation = copy.deepcopy(source_record)
            source_annotation["sample_id"] = sample_id
            source_annotation["image"] = f"{sample_id}.png"
            source_annotation["random_seed"] = variant_seed
            source_annotation["source_random_seed"] = source_record["random_seed"]
            source_annotation["parent"] = {
                "sample_id": source_record["sample_id"],
                "image": Path(os.path.relpath(source_image_path.resolve(), image_dir.resolve())).as_posix(),
                "annotation": Path(
                    os.path.relpath(source_annotation_path.resolve(), annotation_dir.resolve())
                ).as_posix(),
            }
            if source_annotation.get("board_layer"):
                source_annotation["board_layer"] = Path(
                    os.path.relpath(
                        (source_image_path.parent / source_annotation["board_layer"]).resolve(),
                        image_dir.resolve(),
                    )
                ).as_posix()

            transformed_image, transformed_annotation = apply_profile(
                image,
                source_annotation,
                profile,
                variant_seed,
                config,
            )
            transformed_annotation["validation"] = {
                "status": "pending",
                "warnings": ["visual_review_required"],
                "visual_review_required": True,
            }
            image_path = image_dir / transformed_annotation["image"]
            annotation_path = annotation_dir / f"{sample_id}.json"
            if profile == "D0_clean":
                shutil.copyfile(source_image_path, image_path)
            else:
                transformed_image.save(image_path, format="PNG")

            validation_errors = validate_annotation(transformed_annotation, image_path)
            validation_errors.extend(validate_rendered_geometry(transformed_annotation))
            validation_errors.extend(validate_transformed_geometry(source_record, transformed_annotation))
            if validation_errors:
                transformed_annotation["validation"]["status"] = "failed"
                transformed_annotation["validation"]["errors"] = validation_errors
                write_annotation(transformed_annotation, annotation_path)
                raise ValueError(
                    f"Difficulty sample {sample_id} failed validation: "
                    + "; ".join(validation_errors)
                )
            transformed_annotation["validation"]["status"] = "passed"
            transformed_annotation["validation"]["errors"] = []
            write_annotation(transformed_annotation, annotation_path)
            _draw_annotation_overlay(
                transformed_image,
                transformed_annotation,
                overlay_dir / f"{sample_id}_overlay.png",
            )
            generated_records.append(transformed_annotation)
            generated_image_paths.append(image_path)
            for region in transformed_annotation["text_regions"]:
                visibility_counts[region["visibility"]["status"]] += 1

        _write_profile_contact_sheet(generated_image_paths, generated_records, contact_sheet_path)
        report["profiles"][profile] = {
            "sample_count": len(generated_records),
            "language_counts": dict(sorted(Counter(r["language"] for r in generated_records).items())),
            "visibility_status_counts": dict(sorted(visibility_counts.items())),
            "image_dir": str(image_dir),
            "annotation_dir": str(annotation_dir),
            "contact_sheet": str(contact_sheet_path),
            "overlay_dir": str(overlay_dir),
            "passed": True,
        }

    report_path = preview_root / "scene_difficulty_generation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
