from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SOURCE_ROOT = ROOT_DIR / "data" / "generated" / "multilingual_variety_seed_20261030"
PREVIEW_ROOT = ROOT_DIR / "previews" / "multilingual_variety_seed_20261030"
STAGING_ROOT = ROOT_DIR / "hf_upload" / "indic-synthetic-scene-text-1000"
PROFILE_NAMES = (
    "D0_clean",
    "D2_blur",
    "D2_distance",
    "D2_lighting",
    "D2_occlusion",
    "D2_perspective",
)


class DatasetPreparationError(Exception):
    pass


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetPreparationError(f"Cannot read JSON file {path}: {error}") from error


def _safe_source_path(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise DatasetPreparationError(f"Expected a relative source path, got: {relative_path}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise DatasetPreparationError(f"Source path escapes the batch directory: {relative_path}") from error
    if not path.is_file():
        raise DatasetPreparationError(f"Source file does not exist: {path}")
    return path


def _copy_without_overwrite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256(source) != _sha256(destination):
            raise DatasetPreparationError(
                f"Refusing to overwrite a different staged file: {destination}"
            )
        return
    destination.write_bytes(source.read_bytes())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stage_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise DatasetPreparationError(f"Staging path escapes the dataset directory: {relative_path}") from error
    return path


def _copy_source_artifacts(
    source_root: Path,
    preview_root: Path,
    stage_root: Path,
) -> dict[str, Any]:
    manifest_path = preview_root / "generation_manifest.json"
    generation_report_path = preview_root / "generation_report.json"
    if not manifest_path.is_file() or not generation_report_path.is_file():
        raise DatasetPreparationError("The batch manifest or generation report is missing.")

    manifest = _read_json(manifest_path)
    generation_report = _read_json(generation_report_path)
    records = manifest.get("scenes")
    if not isinstance(records, list) or not records:
        raise DatasetPreparationError("The source manifest has no scene records.")
    if generation_report.get("generated_scene_count") != len(records):
        raise DatasetPreparationError("Manifest and generation report scene counts do not match.")

    seen_ids: set[str] = set()
    jsonl_records: list[dict[str, Any]] = []
    primary_images: set[str] = set()
    primary_annotations: set[str] = set()
    parent_images: set[str] = set()
    parent_annotations: set[str] = set()
    board_layers: set[str] = set()

    for record in records:
        sample_id = record.get("sample_id")
        profile = record.get("profile")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
            raise DatasetPreparationError(f"Missing or duplicate sample ID: {sample_id!r}")
        if profile not in PROFILE_NAMES:
            raise DatasetPreparationError(f"Unknown difficulty profile for {sample_id}: {profile!r}")
        seen_ids.add(sample_id)

        source_image = _safe_source_path(source_root, record["image"])
        source_annotation = _safe_source_path(source_root, record["annotation"])
        annotation = _read_json(source_annotation)
        if not isinstance(annotation, dict) or annotation.get("sample_id") != sample_id:
            raise DatasetPreparationError(f"Annotation sample ID mismatch: {source_annotation}")
        if annotation.get("image") != source_image.name:
            raise DatasetPreparationError(f"Annotation image name mismatch: {source_annotation}")

        staged_image = f"images/{profile}/{source_image.name}"
        staged_annotation = f"annotations/{profile}/{source_annotation.name}"
        _copy_without_overwrite(source_image, _safe_stage_path(stage_root, staged_image))
        _copy_without_overwrite(
            source_annotation,
            _safe_stage_path(stage_root, staged_annotation),
        )
        primary_images.add(staged_image)
        primary_annotations.add(staged_annotation)

        parent = annotation.get("parent")
        if isinstance(parent, dict):
            parent_image_ref = parent.get("image")
            parent_annotation_ref = parent.get("annotation")
            if not isinstance(parent_image_ref, str) or not isinstance(parent_annotation_ref, str):
                raise DatasetPreparationError(f"Incomplete parent references in {source_annotation}")
            parent_image = _safe_source_path(
                source_image.parent,
                parent_image_ref,
            )
            parent_annotation_path = _safe_source_path(
                source_annotation.parent,
                parent_annotation_ref,
            )
            parent_record = _read_json(parent_annotation_path)
            if parent_record.get("sample_id") != parent.get("sample_id"):
                raise DatasetPreparationError(f"Parent annotation ID mismatch in {source_annotation}")
            staged_parent_image = f"images/{profile}/{Path(parent_image_ref).as_posix()}"
            staged_parent_annotation = (
                f"annotations/{profile}/{Path(parent_annotation_ref).as_posix()}"
            )
            _copy_without_overwrite(
                parent_image,
                _safe_stage_path(stage_root, staged_parent_image),
            )
            _copy_without_overwrite(
                parent_annotation_path,
                _safe_stage_path(stage_root, staged_parent_annotation),
            )
            parent_images.add(staged_parent_image)
            parent_annotations.add(staged_parent_annotation)

        board_layer_ref = annotation.get("board_layer")
        if isinstance(board_layer_ref, str) and board_layer_ref:
            board_layer = _safe_source_path(source_image.parent, board_layer_ref)
            staged_board_layer = f"images/{profile}/{Path(board_layer_ref).as_posix()}"
            _copy_without_overwrite(
                board_layer,
                _safe_stage_path(stage_root, staged_board_layer),
            )
            board_layers.add(staged_board_layer)

        jsonl_records.append(
            {
                **record,
                "source_image": record["image"],
                "source_annotation": record["annotation"],
                "image": staged_image,
                "annotation": staged_annotation,
            }
        )

    metadata_dir = stage_root / "metadata"
    report_dir = stage_root / "validation_reports"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_jsonl = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in jsonl_records
    )
    _write_without_overwrite(metadata_dir / "manifest.jsonl", manifest_jsonl.encode("utf-8"))

    dataset_metadata = {
        "name": "indic-synthetic-scene-text-1000",
        "schema_version": "1.0",
        "seed": manifest.get("seed"),
        "primary_sample_count": len(records),
        "primary_image_count": len(primary_images),
        "primary_annotation_count": len(primary_annotations),
        "supporting_parent_image_count": len(parent_images),
        "supporting_parent_annotation_count": len(parent_annotations),
        "supporting_board_layer_count": len(board_layers),
        "language_counts": generation_report.get("coverage", {}).get("language_counts", {}),
        "scene_category_counts": generation_report.get("coverage", {}).get(
            "scene_category_counts", {}
        ),
        "difficulty_profile_counts": generation_report.get("coverage", {}).get(
            "difficulty_profile_counts", {}
        ),
        "background_source_counts": generation_report.get("coverage", {}).get(
            "background_source_counts", {}
        ),
        "photorealistic": False,
        "licensing_status": "unresolved; repository should remain private until reviewed",
        "annotation_format": "One UTF-8 JSON object per sample; exact transcription is in text_regions[].transcription.",
    }
    _write_without_overwrite(
        metadata_dir / "dataset_metadata.json",
        (json.dumps(dataset_metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )

    report_sources = {
        manifest_path: metadata_dir / "source_generation_manifest.json",
        generation_report_path: report_dir / "generation_report.json",
    }
    consolidated_report = preview_root / "validation_report.json"
    if consolidated_report.is_file():
        report_sources[consolidated_report] = report_dir / "source_validation_report.json"
    for profile in PROFILE_NAMES:
        profile_report = preview_root / f"{profile}_validation.json"
        if profile_report.is_file():
            report_sources[profile_report] = report_dir / profile_report.name
    font_report = ROOT_DIR / "previews" / "font_validation_seed_20261029" / "font_validation_report.json"
    if font_report.is_file():
        report_sources[font_report] = report_dir / "font_validation_report.json"
    for source, destination in report_sources.items():
        _copy_without_overwrite(source, destination)

    return {
        "sample_count": len(records),
        "primary_images": sorted(primary_images),
        "primary_annotations": sorted(primary_annotations),
        "parent_images": sorted(parent_images),
        "parent_annotations": sorted(parent_annotations),
        "board_layers": sorted(board_layers),
        "manifest_records": jsonl_records,
        "dataset_metadata": dataset_metadata,
    }


def _write_without_overwrite(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise DatasetPreparationError(
                f"Refusing to overwrite a different staged file: {path}"
            )
        return
    path.write_bytes(content)


def _audit_stage(stage_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings = [
        "Dataset licensing and font/asset redistribution rights are unresolved; keep the Hub repository private."
    ]
    manifest_path = stage_root / "metadata" / "manifest.jsonl"
    if not manifest_path.is_file():
        raise DatasetPreparationError(f"Staging manifest is missing: {manifest_path}")

    records: list[dict[str, Any]] = []
    try:
        manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DatasetPreparationError(f"Cannot read staging manifest {manifest_path}: {error}") from error
    for line_number, line in enumerate(manifest_lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"Manifest line {line_number} is invalid JSON: {error}")
            continue
        if not isinstance(record, dict):
            errors.append(f"Manifest line {line_number} is not a JSON object.")
            continue
        records.append(record)

    ids = [record.get("sample_id") for record in records]
    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(ids).items() if count > 1
    )
    if duplicate_ids:
        errors.append(f"Duplicate sample IDs: {duplicate_ids[:10]}")

    expected_images: set[str] = set()
    expected_annotations: set[str] = set()
    expected_parents: set[tuple[str, str]] = set()
    expected_board_layers: set[str] = set()
    language_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    background_counts: Counter[str] = Counter()
    transcription_count = 0

    for record in records:
        sample_id = record.get("sample_id")
        image_ref = record.get("image")
        annotation_ref = record.get("annotation")
        if not isinstance(image_ref, str) or not isinstance(annotation_ref, str):
            errors.append(f"Manifest record {sample_id!r} has no image/annotation paths.")
            continue
        if image_ref in expected_images or annotation_ref in expected_annotations:
            errors.append(f"Duplicate manifest file path for sample {sample_id!r}.")
        expected_images.add(image_ref)
        expected_annotations.add(annotation_ref)
        try:
            image_path = _safe_stage_path(stage_root, image_ref)
            annotation_path = _safe_stage_path(stage_root, annotation_ref)
        except DatasetPreparationError as error:
            errors.append(str(error))
            continue
        if not image_path.is_file():
            errors.append(f"Missing staged image: {image_ref}")
        if not annotation_path.is_file():
            errors.append(f"Missing staged annotation: {annotation_ref}")
            continue
        try:
            annotation = _read_json(annotation_path)
        except DatasetPreparationError as error:
            errors.append(str(error))
            continue
        if not isinstance(annotation, dict):
            errors.append(f"Annotation root is not an object: {annotation_ref}")
            continue
        if annotation.get("sample_id") != sample_id:
            errors.append(f"Annotation sample ID mismatch: {annotation_ref}")
        if not image_path.is_file():
            continue
        if annotation.get("image") != image_path.name:
            errors.append(f"Annotation image filename mismatch: {annotation_ref}")
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
        except (OSError, UnidentifiedImageError) as error:
            errors.append(f"Invalid image {image_ref}: {error}")
            continue
        if (annotation.get("image_width"), annotation.get("image_height")) != (width, height):
            errors.append(f"Image dimensions differ from annotation: {annotation_ref}")
        if annotation.get("language") != record.get("language"):
            errors.append(f"Language mismatch between manifest and annotation: {annotation_ref}")
        if annotation.get("category") != record.get("category"):
            errors.append(f"Category mismatch between manifest and annotation: {annotation_ref}")
        profile = annotation.get("scene_difficulty", {}).get("profile")
        if profile != record.get("profile"):
            errors.append(f"Difficulty profile mismatch: {annotation_ref}")
        if annotation.get("background_source_type") != record.get("background_source_type"):
            errors.append(f"Background source mismatch: {annotation_ref}")

        regions = annotation.get("text_regions")
        if not isinstance(regions, list) or not regions:
            errors.append(f"Annotation has no text regions: {annotation_ref}")
        else:
            for region in regions:
                transcription = region.get("transcription") if isinstance(region, dict) else None
                if not isinstance(transcription, str):
                    errors.append(f"Invalid transcription in {annotation_ref}")
                else:
                    transcription_count += 1

        language_counts[str(record.get("language", "unknown"))] += 1
        category_counts[str(record.get("scene_category", "unknown"))] += 1
        difficulty_counts[str(record.get("profile", "unknown"))] += 1
        background_counts[str(record.get("background_source_type", "unknown"))] += 1

        parent = annotation.get("parent")
        if isinstance(parent, dict):
            parent_image_ref = parent.get("image")
            parent_annotation_ref = parent.get("annotation")
            if not isinstance(parent_image_ref, str) or not isinstance(parent_annotation_ref, str):
                errors.append(f"Incomplete parent links in {annotation_ref}")
            else:
                profile_root = Path(image_ref).parent
                parent_image_path = (profile_root / parent_image_ref).as_posix()
                annotation_profile_root = Path(annotation_ref).parent
                parent_annotation_path = (
                    annotation_profile_root / parent_annotation_ref
                ).as_posix()
                expected_parents.add((parent_image_path, parent_annotation_path))
                for relative in (parent_image_path, parent_annotation_path):
                    try:
                        dependency = _safe_stage_path(stage_root, relative)
                        if not dependency.is_file():
                            errors.append(f"Missing parent dependency: {relative}")
                        elif relative == parent_image_path:
                            try:
                                with Image.open(dependency) as image:
                                    image.verify()
                            except (OSError, UnidentifiedImageError) as error:
                                errors.append(f"Invalid parent image {relative}: {error}")
                        else:
                            parent_record = _read_json(dependency)
                            if parent_record.get("sample_id") != parent.get("sample_id"):
                                errors.append(f"Parent annotation ID mismatch: {relative}")
                    except DatasetPreparationError as error:
                        errors.append(str(error))

        board_layer_ref = annotation.get("board_layer")
        if isinstance(board_layer_ref, str) and board_layer_ref:
            board_layer_path = (Path(image_ref).parent / board_layer_ref).as_posix()
            expected_board_layers.add(board_layer_path)
            try:
                board_layer = _safe_stage_path(stage_root, board_layer_path)
                if not board_layer.is_file():
                    errors.append(f"Missing retained board layer: {board_layer_path}")
                else:
                    try:
                        with Image.open(board_layer) as image:
                            image.verify()
                    except (OSError, UnidentifiedImageError) as error:
                        errors.append(f"Invalid retained board layer {board_layer_path}: {error}")
            except DatasetPreparationError as error:
                errors.append(str(error))

    actual_primary_images = {
        path.relative_to(stage_root).as_posix()
        for profile in PROFILE_NAMES
        for path in (stage_root / "images" / profile).glob("*.png")
        if path.is_file()
    }
    actual_primary_annotations = {
        path.relative_to(stage_root).as_posix()
        for profile in PROFILE_NAMES
        for path in (stage_root / "annotations" / profile).glob("*.json")
        if path.is_file()
    }
    if actual_primary_images != expected_images:
        errors.append(
            f"Primary image/manifest mismatch (unlisted={len(actual_primary_images - expected_images)}, "
            f"missing={len(expected_images - actual_primary_images)})."
        )
    if actual_primary_annotations != expected_annotations:
        errors.append(
            f"Primary annotation/manifest mismatch (unlisted={len(actual_primary_annotations - expected_annotations)}, "
            f"missing={len(expected_annotations - actual_primary_annotations)})."
        )

    actual_parent_images = {
        path.relative_to(stage_root).as_posix()
        for profile in PROFILE_NAMES
        for path in (stage_root / "images" / profile / "parents").glob("*.png")
        if path.is_file()
    }
    actual_parent_annotations = {
        path.relative_to(stage_root).as_posix()
        for profile in PROFILE_NAMES
        for path in (stage_root / "annotations" / profile / "parents").glob("*.json")
        if path.is_file()
    }
    expected_parent_images = {image for image, _ in expected_parents}
    expected_parent_annotations = {annotation for _, annotation in expected_parents}
    if actual_parent_images != expected_parent_images:
        errors.append("Parent image files do not match the annotation parent links.")
    if actual_parent_annotations != expected_parent_annotations:
        errors.append("Parent annotation files do not match the annotation parent links.")

    actual_board_layers = {
        path.relative_to(stage_root).as_posix()
        for profile in PROFILE_NAMES
        for path in (stage_root / "images" / profile / "board_layers").glob("*.png")
        if path.is_file()
    }
    if actual_board_layers != expected_board_layers:
        errors.append("Retained board-layer files do not match annotation references.")

    all_image_names = [
        Path(path).name
        for path in expected_images | expected_parent_images | actual_board_layers
    ]
    duplicate_image_names = sorted(
        name for name, count in Counter(all_image_names).items() if count > 1
    )
    all_annotation_names = [
        Path(path).name
        for path in expected_annotations | expected_parent_annotations
    ]
    duplicate_annotation_names = sorted(
        name for name, count in Counter(all_annotation_names).items() if count > 1
    )
    if duplicate_image_names:
        errors.append(f"Duplicate image filenames: {duplicate_image_names[:10]}")
    if duplicate_annotation_names:
        errors.append(f"Duplicate annotation filenames: {duplicate_annotation_names[:10]}")

    report = {
        "schema_version": "1.0",
        "passed": not errors,
        "primary_sample_count": len(records),
        "primary_image_count": len(actual_primary_images),
        "primary_annotation_count": len(actual_primary_annotations),
        "supporting_parent_image_count": len(actual_parent_images),
        "supporting_parent_annotation_count": len(actual_parent_annotations),
        "supporting_board_layer_count": len(actual_board_layers),
        "text_region_transcription_count": transcription_count,
        "language_counts": dict(sorted(language_counts.items())),
        "scene_category_counts": dict(sorted(category_counts.items())),
        "difficulty_profile_counts": dict(sorted(difficulty_counts.items())),
        "background_source_counts": dict(sorted(background_counts.items())),
        "errors": errors,
        "warnings": warnings,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely stage and audit the verified 1,000-image Hugging Face dataset."
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Copy verified source files into the staging directory without overwriting differences.",
    )
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--preview-root", type=Path, default=PREVIEW_ROOT)
    parser.add_argument("--stage-root", type=Path, default=STAGING_ROOT)
    args = parser.parse_args()

    try:
        if args.prepare:
            _copy_source_artifacts(args.source_root, args.preview_root, args.stage_root)
        report = _audit_stage(args.stage_root)
        if args.prepare:
            _write_without_overwrite(
                args.stage_root / "validation_reports" / "hf_upload_validation_report.json",
                (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
            )
        print(
            f"Upload staging audit: {'PASSED' if report['passed'] else 'FAILED'}\n"
            f"Primary images/annotations: {report['primary_image_count']}/"
            f"{report['primary_annotation_count']} of {report['primary_sample_count']} samples\n"
            f"Linked parent images/annotations: {report['supporting_parent_image_count']}/"
            f"{report['supporting_parent_annotation_count']}; board layers: "
            f"{report['supporting_board_layer_count']}\n"
            f"Languages: {report['language_counts']}\n"
            f"Scene categories: {report['scene_category_counts']}\n"
            f"Difficulty profiles: {report['difficulty_profile_counts']}"
        )
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        return 0 if report["passed"] else 1
    except DatasetPreparationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
