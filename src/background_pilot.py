from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .annotation_exporter import (
    validate_annotation,
    validate_rendered_geometry,
    validate_transformed_geometry,
    write_annotation,
)
from .background_assets import BackgroundAsset, BackgroundManifest, load_background_manifest
from .background_composer import (
    compose_background_scene,
    update_composed_surface_for_transform,
    validate_background_composition,
    validate_transformed_surface,
)
from .content_engine import SUPPORTED_LANGUAGES, load_config
from .quality_checker import validate_dataset
from .scene_transformer import PROFILE_NAMES, apply_profile
from .signboard_generator import SHOP_BOARD_LAYOUTS

ROOT_DIR = Path(__file__).resolve().parents[1]


def _select_assets(
    assets: Sequence[BackgroundAsset],
    maximum: int,
    seed: int,
) -> List[BackgroundAsset]:
    candidates = sorted(
        assets,
        key=lambda asset: (asset.record["asset_id"], asset.record["background_id"]),
    )
    random.Random(seed).shuffle(candidates)
    selected: List[BackgroundAsset] = []
    source_counts: Counter = Counter()
    category_counts: Counter = Counter()
    while candidates and len(selected) < maximum:
        candidate = min(
            candidates,
            key=lambda asset: (
                source_counts[asset.record["source_type"]],
                category_counts[asset.record["scene_category"]],
                candidates.index(asset),
            ),
        )
        candidates.remove(candidate)
        selected.append(candidate)
        source_counts[candidate.record["source_type"]] += 1
        category_counts[candidate.record["scene_category"]] += 1
    return selected


def _schedule_language_profiles(count: int, seed: int) -> List[Tuple[str, str]]:
    available = [
        (language, profile)
        for language in SUPPORTED_LANGUAGES
        for profile in PROFILE_NAMES
    ]
    rng = random.Random(seed)
    rng.shuffle(available)
    language_counts: Counter = Counter()
    profile_counts: Counter = Counter()
    schedule: List[Tuple[str, str]] = []
    while available and len(schedule) < count:
        best_score = min(
            (language_counts[language], profile_counts[profile])
            for language, profile in available
        )
        options = [
            pair
            for pair in available
            if (language_counts[pair[0]], profile_counts[pair[1]]) == best_score
        ]
        selected = rng.choice(options)
        available.remove(selected)
        schedule.append(selected)
        language_counts[selected[0]] += 1
        profile_counts[selected[1]] += 1
    return schedule


def _draw_overlay(image: Image.Image, annotation: dict, output_path: Path) -> None:
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    for region in annotation["text_regions"]:
        polygon = [(round(x), round(y)) for x, y in region["polygon"]]
        draw.line(polygon + [polygon[0]], fill=(255, 0, 0), width=3)
        bbox = region["bbox"]
        draw.text(
            (max(0, round(bbox["x_min"])), max(0, round(bbox["y_min"]) - 12)),
            str(region["reading_order"]),
            fill=(255, 255, 0),
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path, format="PNG")


def _write_contact_sheet(image_paths: List[Path], records: List[dict], output_path: Path) -> None:
    if not image_paths:
        sheet = Image.new("RGB", (640, 180), (238, 238, 238))
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()
        draw.text((20, 45), "No eligible background assets.", fill=(20, 20, 20), font=font)
        draw.text(
            (20, 75),
            "No samples generated; validation is not applicable.",
            fill=(20, 20, 20),
            font=font,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path, format="PNG")
        return
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
        sheet.paste(sample, (cell_x + (cell_width - sample.width) // 2, cell_y + 5))
        draw.text(
            (cell_x + 8, cell_y + cell_height - 25),
            f"{record['sample_id']} | {record['language']} | {record['source_scene_category']}",
            font=font,
            fill=(20, 20, 20),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")


def _coverage(records: Sequence[dict]) -> dict:
    return {
        "language_counts": dict(sorted(Counter(record["language"] for record in records).items())),
        "scene_category_counts": dict(
            sorted(Counter(record["source_scene_category"] for record in records).items())
        ),
        "source_type_counts": dict(
            sorted(
                Counter(
                    record["background_composition"]["source_type"] for record in records
                ).items()
            )
        ),
        "difficulty_profile_counts": dict(
            sorted(
                Counter(record["scene_difficulty"]["profile"] for record in records).items()
            )
        ),
    }


def _planned_paths(
    assignments: Sequence[dict],
    output_root: Path,
    preview_root: Path,
    profiles: Sequence[str],
) -> List[Path]:
    paths: List[Path] = []
    for item in assignments:
        profile = item["profile"]
        sample_id = item["sample_id"]
        profile_root = output_root / profile
        paths.extend(
            [
                profile_root / "images" / f"{sample_id}.png",
                profile_root / "images" / "parents" / f"{sample_id}__base.png",
                profile_root / "images" / "board_layers" / f"{sample_id}_board.png",
                profile_root / "annotations" / f"{sample_id}.json",
                profile_root / "annotations" / "parents" / f"{sample_id}__base.json",
                preview_root / "overlays" / profile / f"{sample_id}_overlay.png",
            ]
        )
    for profile in profiles:
        paths.extend(
            [
                preview_root / f"{profile}_contact_sheet.png",
                preview_root / f"{profile}_validation.json",
            ]
        )
    paths.extend(
        [
            preview_root / "pilot_manifest.json",
            preview_root / "validation_report.json",
            preview_root / "generation_report.json",
        ]
    )
    return paths


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate_background_pilot(
    manifest_path: Path,
    output_root: Path,
    preview_root: Path,
    seed: int,
    *,
    project_root: Path = ROOT_DIR,
    maximum_scenes: int = 24,
) -> dict:
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if maximum_scenes < 0:
        raise ValueError("maximum_scenes must not be negative.")
    config = load_config()
    generation = config["shop_board_generation"]
    background_policy = config["background_pipeline"]
    target_scene_count = background_policy["pilot_scene_count"]
    if maximum_scenes > target_scene_count:
        raise ValueError(f"maximum_scenes cannot exceed the configured pilot target of {target_scene_count}.")
    configured_profiles = tuple(generation["scene_difficulty"]["profiles"])
    if configured_profiles != PROFILE_NAMES:
        raise ValueError(
            "The configured difficulty profiles differ from the implemented profile set; "
            "refusing to alter their semantics."
        )
    manifest: BackgroundManifest = load_background_manifest(
        manifest_path,
        project_root=project_root,
        policy=background_policy,
    )
    assets = _select_assets(manifest.assets, maximum_scenes, seed ^ 0xBACC)
    schedule = _schedule_language_profiles(len(assets), seed ^ 0x1D1C)
    layout_rng = random.Random(seed ^ 0x1A70)
    root_rng = random.Random(seed)

    assignments = []
    for asset, (language, profile) in zip(assets, schedule):
        asset_token = hashlib.sha256(
            asset.record["asset_id"].encode("utf-8")
        ).hexdigest()[:10]
        sample_id = (
            f"background-pilot-{seed & 0xFFFFFFFF:08x}-{asset_token}-"
            f"{language}-{profile.lower()}"
        )
        assignments.append(
            {
                "asset": asset,
                "language": language,
                "profile": profile,
                "sample_id": sample_id,
                "sample_seed": root_rng.getrandbits(64),
                "board_seed": root_rng.getrandbits(64),
                "layout_id": layout_rng.choice(SHOP_BOARD_LAYOUTS),
            }
        )

    planned_paths = _planned_paths(assignments, output_root, preview_root, PROFILE_NAMES)
    existing = [path for path in planned_paths if path.exists()]
    if output_root.exists():
        existing.extend(path for path in output_root.rglob("*") if path.is_file())
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing background-pilot outputs: "
            + ", ".join(str(path) for path in existing)
        )

    output_records: Dict[str, List[dict]] = {profile: [] for profile in PROFILE_NAMES}
    profile_images: Dict[str, List[Path]] = {profile: [] for profile in PROFILE_NAMES}
    for item in assignments:
        asset = item["asset"]
        profile = item["profile"]
        profile_root = output_root / profile
        image_dir = profile_root / "images"
        annotation_dir = profile_root / "annotations"
        layer_relative = f"board_layers/{item['sample_id']}_board.png"
        base_image, base_annotation, board_layer = compose_background_scene(
            asset,
            language=item["language"],
            sample_seed=item["sample_seed"],
            sample_id=item["sample_id"],
            board_seed=item["board_seed"],
            layout_id=item["layout_id"],
            layer_relative_path=layer_relative,
            minimum_surface_width=background_policy["minimum_surface_width"],
            minimum_surface_height=background_policy["minimum_surface_height"],
        )
        base_id = f"{item['sample_id']}__base"
        base_annotation["sample_id"] = base_id
        base_annotation["image"] = f"{base_id}.png"
        base_annotation["validation"] = {
            "status": "passed",
            "warnings": ["visual_review_required"],
            "visual_review_required": True,
        }
        parent_image_path = image_dir / "parents" / base_annotation["image"]
        parent_annotation_path = annotation_dir / "parents" / f"{base_id}.json"
        parent_image_path.parent.mkdir(parents=True, exist_ok=True)
        parent_annotation_path.parent.mkdir(parents=True, exist_ok=True)
        base_image.save(parent_image_path, format="PNG")
        layer_path = image_dir / layer_relative
        layer_path.parent.mkdir(parents=True, exist_ok=True)
        board_layer.save(layer_path, format="PNG")
        validation_errors = validate_annotation(base_annotation, parent_image_path)
        validation_errors.extend(validate_rendered_geometry(base_annotation))
        validation_errors.extend(validate_background_composition(base_annotation))
        if validation_errors:
            raise ValueError(
                f"Base composition {base_id} failed validation: {'; '.join(validation_errors)}"
            )
        write_annotation(base_annotation, parent_annotation_path)

        variant_input = copy.deepcopy(base_annotation)
        variant_input["sample_id"] = item["sample_id"]
        variant_input["image"] = f"{item['sample_id']}.png"
        variant_input["random_seed"] = item["sample_seed"]
        variant_input["parent"] = {
            "sample_id": base_id,
            "image": f"parents/{base_annotation['image']}",
            "annotation": f"parents/{base_id}.json",
        }
        transformed_image, annotation = apply_profile(
            base_image,
            variant_input,
            profile,
            item["sample_seed"],
            config,
        )
        annotation["validation"] = {
            "status": "pending",
            "warnings": ["visual_review_required"],
            "visual_review_required": True,
        }
        update_composed_surface_for_transform(annotation)
        final_image_path = image_dir / annotation["image"]
        final_annotation_path = annotation_dir / f"{item['sample_id']}.json"
        final_image_path.parent.mkdir(parents=True, exist_ok=True)
        annotation["image_width"], annotation["image_height"] = transformed_image.size
        transformed_image.save(final_image_path, format="PNG")
        validation_errors = validate_annotation(annotation, final_image_path)
        validation_errors.extend(validate_rendered_geometry(annotation))
        validation_errors.extend(validate_transformed_geometry(base_annotation, annotation))
        validation_errors.extend(validate_background_composition(annotation))
        validation_errors.extend(validate_transformed_surface(base_annotation, annotation))
        if validation_errors:
            raise ValueError(
                f"Difficulty sample {item['sample_id']} failed validation: "
                + "; ".join(validation_errors)
            )
        annotation["validation"]["status"] = "passed"
        annotation["validation"]["errors"] = []
        write_annotation(annotation, final_annotation_path)
        overlay_path = preview_root / "overlays" / profile / f"{item['sample_id']}_overlay.png"
        _draw_overlay(transformed_image, annotation, overlay_path)
        output_records[profile].append(annotation)
        profile_images[profile].append(final_image_path)

    profile_validation = {}
    for profile in PROFILE_NAMES:
        profile_root = output_root / profile
        image_dir = profile_root / "images"
        annotation_dir = profile_root / "annotations"
        image_dir.mkdir(parents=True, exist_ok=True)
        annotation_dir.mkdir(parents=True, exist_ok=True)
        validation = validate_dataset(image_dir, annotation_dir)
        applicable = validation["image_count"] > 0
        profile_report = dict(validation)
        profile_report["applicable"] = applicable
        profile_report["passed"] = validation["passed"] if applicable else None
        if not applicable:
            profile_report["note"] = "No eligible sample was generated; validation is not applicable."
        profile_validation[profile] = profile_report
        _write_json(preview_root / f"{profile}_validation.json", profile_report)
        _write_contact_sheet(
            profile_images[profile],
            output_records[profile],
            preview_root / f"{profile}_contact_sheet.png",
        )

    records = [record for profile in PROFILE_NAMES for record in output_records[profile]]
    coverage = _coverage(records)
    sample_manifest = {
        "schema_version": "1.0",
        "seed": seed,
        "requested_scene_count": maximum_scenes,
        "generated_scene_count": len(records),
        "scenes": [
            {
                "sample_id": record["sample_id"],
                "image": f"{record['scene_difficulty']['profile']}/images/{record['image']}",
                "annotation": f"{record['scene_difficulty']['profile']}/annotations/{record['sample_id']}.json",
                "language": record["language"],
                "script": record["script"],
                "category": record["category"],
                "scene_category": record["source_scene_category"],
                "background_id": record["background_composition"]["background_id"],
                "source_type": record["background_composition"]["source_type"],
                "difficulty_profile": record["scene_difficulty"]["profile"],
                "random_seed": record["random_seed"],
            }
            for record in records
        ],
    }
    source_target = background_policy["pilot_target_per_source_type"]
    category_target = background_policy["pilot_target_per_scene_category"]
    language_target = target_scene_count // len(SUPPORTED_LANGUAGES)
    profile_target = target_scene_count // len(PROFILE_NAMES)
    full_target_counts_met = (
        len(records) == target_scene_count
        and all(
            coverage["source_type_counts"].get(source_type, 0) == source_target
            for source_type in ("real_photo", "ai_generated")
        )
        and all(
            coverage["scene_category_counts"].get(category, 0) == category_target
            for category in ("shopfront", "outdoor_hoarding")
        )
        and all(
            coverage["language_counts"].get(language, 0) == language_target
            for language in SUPPORTED_LANGUAGES
        )
        and all(
            coverage["difficulty_profile_counts"].get(profile, 0) == profile_target
            for profile in PROFILE_NAMES
        )
    )
    coverage_shortfalls = {
        "source_type_counts": {
            source_type: max(
                0,
                source_target - coverage["source_type_counts"].get(source_type, 0),
            )
            for source_type in ("real_photo", "ai_generated")
        },
        "scene_category_counts": {
            category: max(
                0,
                category_target - coverage["scene_category_counts"].get(category, 0),
            )
            for category in ("shopfront", "outdoor_hoarding")
        },
        "language_counts": {
            language: max(
                0,
                language_target - coverage["language_counts"].get(language, 0),
            )
            for language in SUPPORTED_LANGUAGES
        },
        "difficulty_profile_counts": {
            profile: max(
                0,
                profile_target - coverage["difficulty_profile_counts"].get(profile, 0),
            )
            for profile in PROFILE_NAMES
        },
    }
    validation_applicable = bool(records)
    passed = (
        all(report["passed"] for report in profile_validation.values() if report["applicable"])
        if validation_applicable
        else None
    )
    report = {
        "schema_version": "1.0",
        "seed": seed,
        "manifest_path": str(manifest_path),
        "target_scene_count": target_scene_count,
        "requested_scene_limit": maximum_scenes,
        "eligible_background_count": len(manifest.assets),
        "excluded_backgrounds": list(manifest.excluded),
        "manifest_warnings": list(manifest.warnings),
        "generated_scene_count": len(records),
        "coverage": coverage,
        "coverage_shortfalls": coverage_shortfalls,
        "target_counts": {
            "real_photo": source_target,
            "ai_generated": source_target,
            "shopfront": category_target,
            "outdoor_hoarding": category_target,
            "per_language": language_target,
            "per_profile": profile_target,
        },
        "coverage_targets_met": full_target_counts_met,
        "validation_applicable": validation_applicable,
        "validation_passed": passed,
        "complete": full_target_counts_met and passed,
        "shortfall": max(0, target_scene_count - len(records)),
        "asset_instructions": [
            "Add suitable local images under assets/backgrounds/real_photos/ or assets/backgrounds/ai_generated/.",
            "For each image, add a manifest record with its true source/provenance, original filename, dimensions, licensing status, and a manually verified blank-surface polygon in pixel-edge coordinates.",
            "Mark manual_review_status and surface_verification_status as approved/verified only after checking the actual image; unverified assets are excluded.",
            "For AI-created images, record the prompt and generator/terms when known; do not claim a service or license that was not actually used or verified.",
            "Re-run scripts/generate_photorealistic_pilot.py with a new output directory if prior outputs already exist.",
        ],
        "validation_reports": {
            profile: {
                "passed": value["passed"],
                "applicable": value["applicable"],
                "image_count": value["image_count"],
                "annotation_count": value["annotation_count"],
                "errors": value["errors"],
            }
            for profile, value in profile_validation.items()
        },
        "limitations": [
            "No image-generation service is called by this pipeline.",
            "Source backgrounds are used only when provenance, research-use permission, review, and blank-surface geometry pass the configured gates.",
            "Occlusion visibility is the inherited bbox-unoccluded-area estimate, not pixel-accurate glyph visibility.",
            "Contact sheets and overlays require manual visual review; automated validation does not establish photorealism.",
        ],
    }
    _write_json(preview_root / "pilot_manifest.json", sample_manifest)
    _write_json(preview_root / "validation_report.json", {
        "schema_version": "1.0",
        "passed": passed,
        "applicable": validation_applicable,
        "generated_scene_count": len(records),
        "profiles": report["validation_reports"],
    })
    _write_json(preview_root / "generation_report.json", report)
    return report
