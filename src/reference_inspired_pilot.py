from __future__ import annotations

import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .background_assets import BackgroundAsset, load_background_manifest
from .background_composer import compose_background_scene, update_composed_surface_for_transform
from .annotation_exporter import (
    validate_annotation,
    validate_rendered_geometry,
    validate_transformed_geometry,
    write_annotation,
)
from .content_engine import SUPPORTED_LANGUAGES, load_config
from .quality_checker import validate_dataset
from .scene_composer import compose_storefront_scene
from .scene_transformer import PROFILE_NAMES, apply_profile
from .signboard_generator import SHOP_BOARD_LAYOUTS, render_shop_board_sample

ROOT_DIR = Path(__file__).resolve().parents[1]
SCENE_CATEGORIES = ("shopfront", "outdoor_hoarding")


def _draw_hoarding_background(
    size: Tuple[int, int],
    seed: int,
    board_size: Tuple[int, int],
) -> Tuple[Image.Image, Tuple[int, int], dict]:
    width, height = size
    board_width, board_height = board_size
    rng = random.Random(seed)
    if board_width + 160 > width or board_height + 250 > height:
        raise ValueError("Configured signboard does not fit on the outdoor hoarding scene.")
    board_x = (width - board_width) // 2
    board_y = 150
    sky_top = tuple(rng.choice([(116, 165, 196), (157, 184, 197), (192, 168, 139)]))
    sky_horizon = tuple(
        min(255, channel + rng.randint(8, 24)) for channel in sky_top
    )
    image = Image.new("RGB", size, sky_top)
    draw = ImageDraw.Draw(image)
    for y in range(height):
        blend = y / max(height - 1, 1)
        color = tuple(
            round(sky_top[channel] * (1 - blend) + sky_horizon[channel] * blend)
            for channel in range(3)
        )
        draw.line((0, y, width, y), fill=color)

    horizon_y = board_y + board_height + 100
    for _ in range(11):
        building_width = rng.randint(90, 260)
        building_height = rng.randint(130, 390)
        left = rng.randint(-40, width - 40)
        top = max(0, horizon_y - building_height)
        color = tuple(rng.randint(104, 170) for _ in range(3))
        draw.rectangle(
            (left, top, left + building_width, horizon_y + 70),
            fill=color,
            outline=tuple(max(0, channel - 25) for channel in color),
            width=5,
        )
        window_color = tuple(min(255, channel + 38) for channel in color)
        for window_y in range(top + 20, horizon_y, 58):
            for window_x in range(left + 16, left + building_width - 12, 48):
                draw.rectangle(
                    (window_x, window_y, window_x + 23, window_y + 32),
                    fill=window_color,
                )
    draw.rectangle((0, horizon_y + 70, width, height), fill=(117, 112, 99))
    draw.polygon(
        [
            (0, horizon_y + 70),
            (width, horizon_y + 70),
            (width, height),
            (0, height),
        ],
        fill=(111, 108, 99),
    )
    for y in range(horizon_y + 110, height, 90):
        draw.line((0, y, width, y - 25), fill=(126, 121, 110), width=3)

    support_color = tuple(rng.randint(58, 91) for _ in range(3))
    support_width = max(22, board_width // 38)
    support_top = board_y + board_height - 5
    support_bottom = min(height - 35, support_top + 290)
    for support_x in (board_x + board_width // 5, board_x + board_width * 4 // 5):
        draw.rectangle(
            (
                support_x - support_width // 2,
                support_top,
                support_x + support_width // 2,
                support_bottom,
            ),
            fill=support_color,
            outline=tuple(max(0, channel - 20) for channel in support_color),
            width=4,
        )
        draw.polygon(
            [
                (support_x, support_top + 20),
                (support_x + 80, support_bottom),
                (support_x - 80, support_bottom),
            ],
            outline=support_color,
            fill=None,
        )
    draw.rectangle(
        (
            board_x - 18,
            board_y - 18,
            board_x + board_width + 18,
            board_y + board_height + 18,
        ),
        outline=(55, 53, 48),
        width=16,
    )
    parameters = {
        "template_id": "procedural_outdoor_hoarding_reference_inspired_v1",
        "sky_rgb": list(sky_top),
        "horizon_rgb": list(sky_horizon),
        "support_rgb": list(support_color),
        "horizon_y": horizon_y,
        "board_position": {
            "x": board_x,
            "y": board_y,
            "width": board_width,
            "height": board_height,
        },
        "background_source": "procedural_shapes",
        "visual_review_required": True,
    }
    return image, (board_x, board_y), parameters


def _compose_hoarding_scene(
    board: Image.Image,
    board_annotation: dict,
    scene_seed: int,
    scene_id: str,
    config: dict,
) -> Tuple[Image.Image, dict, Image.Image]:
    scene_sizes = config["shop_board_generation"]["storefront_scene"]["scene_sizes"]
    rng = random.Random(scene_seed)
    scene_size = tuple(rng.choice(scene_sizes))
    scene, (board_x, board_y), parameters = _draw_hoarding_background(
        scene_size,
        scene_seed,
        board.size,
    )
    scene_rgba = scene.convert("RGBA")
    scene_rgba.alpha_composite(board.convert("RGBA"), dest=(board_x, board_y))
    width, height = scene_size
    annotation = copy.deepcopy(board_annotation)
    annotation["sample_id"] = scene_id
    annotation["image"] = f"{scene_id}.png"
    annotation["image_width"] = width
    annotation["image_height"] = height
    annotation["board_layer"] = f"{scene_id}_board.png"
    annotation["source_board_sample_id"] = board_annotation["sample_id"]
    annotation["scene_category"] = "outdoor_hoarding"
    annotation["scene_composition"] = {
        "type": "translation_only",
        "coordinate_system": "pixel_edges",
        "board_to_scene_translation": [board_x, board_y],
        "scene_size": [width, height],
        "board_size": list(board.size),
        "board_polygon": [
            [board_x, board_y],
            [board_x + board.width, board_y],
            [board_x + board.width, board_y + board.height],
            [board_x, board_y + board.height],
        ],
        "random_seed": scene_seed,
        **parameters,
    }
    annotation["generation_parameters"] = copy.deepcopy(
        board_annotation["generation_parameters"]
    )
    annotation["generation_parameters"]["board_position_in_scene"] = {
        "x": board_x,
        "y": board_y,
        "width": board.width,
        "height": board.height,
    }
    for region in annotation["text_regions"]:
        region["board_bbox"] = dict(region["bbox"])
        region["board_polygon"] = [list(point) for point in region["polygon"]]
        region["bbox"] = {
            "x_min": region["board_bbox"]["x_min"] + board_x,
            "y_min": region["board_bbox"]["y_min"] + board_y,
            "x_max": region["board_bbox"]["x_max"] + board_x,
            "y_max": region["board_bbox"]["y_max"] + board_y,
        }
        region["polygon"] = [
            [x + board_x, y + board_y] for x, y in region["board_polygon"]
        ]
    annotation["scene_parameters"] = parameters
    annotation["validation"] = {
        "status": "pending",
        "warnings": ["visual_review_required"],
        "visual_review_required": True,
    }
    return scene_rgba, annotation, board


def _make_overlay(image: Image.Image, annotation: dict, path: Path) -> None:
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    for region in annotation["text_regions"]:
        polygon = [(round(x), round(y)) for x, y in region["polygon"]]
        draw.line(polygon + [polygon[0]], fill=(255, 0, 0), width=3)
        box = region["bbox"]
        draw.text(
            (max(0, round(box["x_min"])), max(0, round(box["y_min"]) - 13)),
            str(region["reading_order"]),
            fill=(255, 255, 0),
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(path, format="PNG")


def _write_contact_sheet(
    image_paths: Sequence[Path],
    records: Sequence[dict],
    destination: Path,
    *,
    group_key: Optional[str] = None,
) -> None:
    columns = 4
    cell_width, cell_height = 400, 300
    rows = max(1, (len(image_paths) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (image_path, record) in enumerate(zip(image_paths, records)):
        with Image.open(image_path) as source:
            sample = source.convert("RGB")
            sample.thumbnail((cell_width - 16, cell_height - 38), Image.Resampling.LANCZOS)
        cell_x = (index % columns) * cell_width
        cell_y = (index // columns) * cell_height
        sheet.paste(sample, (cell_x + (cell_width - sample.width) // 2, cell_y + 4))
        group_value = (
            record["language"]
            if group_key == "language"
            else record["generation_parameters"]["font_id"]
            if group_key == "font"
            else record["scene_difficulty"]["profile"]
            if group_key == "profile"
            else record["scene_category"]
        )
        draw.text(
            (cell_x + 6, cell_y + cell_height - 28),
            f"{group_value} | {record['language']} | {record['sample_id'][-12:]}",
            font=font,
            fill=(20, 20, 20),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG")


def _make_schedule(count: int, seed: int) -> List[Tuple[str, str, str]]:
    if count <= 0:
        raise ValueError("Pilot count must be a positive integer.")
    root_rng = random.Random(seed)
    if count == 12 or count == 24:
        schedule = [
            (
                language,
                SCENE_CATEGORIES[(profile_index + language_index) % len(SCENE_CATEGORIES)],
                profile,
            )
            for profile_index, profile in enumerate(PROFILE_NAMES)
            for language_index, language in enumerate(SUPPORTED_LANGUAGES)
        ]
        root_rng.shuffle(schedule)
        if count == 24:
            return schedule
        category_pairs = [
            (language, category)
            for language in SUPPORTED_LANGUAGES
            for category in SCENE_CATEGORIES
        ]
        category_pairs.extend(
            (SUPPORTED_LANGUAGES[index], SCENE_CATEGORIES[index % 2])
            for index in range(4)
        )
        root_rng.shuffle(category_pairs)
        profiles = [
            profile
            for profile in PROFILE_NAMES
            for _ in range(count // len(PROFILE_NAMES))
        ]
        root_rng.shuffle(profiles)
        return [
            (language, category, profile)
            for (language, category), profile in zip(category_pairs, profiles)
        ]

    excluded_extras = (
        {("marathi", "shopfront")},
        {("hindi", "outdoor_hoarding")},
        {("gujarati", "shopfront")},
        {("bengali", "outdoor_hoarding")},
        {("marathi", "outdoor_hoarding"), ("gujarati", "outdoor_hoarding")},
        {("hindi", "shopfront"), ("bengali", "shopfront")},
    )
    base_cycle: List[Tuple[str, str, str]] = []
    omitted: List[Tuple[str, str, str]] = []
    for profile_index, profile in enumerate(PROFILE_NAMES):
        excluded = excluded_extras[profile_index]
        profile_items = [
            (language, category, profile)
            for language in SUPPORTED_LANGUAGES
            for category in SCENE_CATEGORIES
            if (language, category) not in excluded
        ]
        base_cycle.extend(profile_items)
        omitted.extend(
            (language, category, profile)
            for language, category in sorted(excluded)
        )
    base_cycle.extend(omitted)
    if len(base_cycle) != len(SUPPORTED_LANGUAGES) * len(SCENE_CATEGORIES) * len(PROFILE_NAMES):
        raise RuntimeError("The deterministic coverage cycle has an invalid number of slots.")

    full_cycles, remainder = divmod(count, len(base_cycle))
    schedule: List[Tuple[str, str, str]] = []
    for _ in range(full_cycles):
        cycle = list(base_cycle)
        root_rng.shuffle(cycle)
        schedule.extend(cycle)
    schedule.extend(base_cycle[:remainder])
    return schedule


def _load_generation_fonts(
    font_validation_report_path: Optional[Path],
    sample_count: int,
) -> Tuple[Dict[str, List[dict]], Optional[str]]:
    config = load_config()
    registry = config.get("font_registry", {}).get("fonts", [])
    if font_validation_report_path is None:
        if sample_count > 24:
            raise ValueError(
                "Large batches require a completed font-validation report; "
                "run scripts/validate_font_registry.py first."
            )
        options: Dict[str, List[dict]] = {language: [] for language in SUPPORTED_LANGUAGES}
        for language, configured_path in config["font_paths"].items():
            path = Path(configured_path)
            registry_entry = next(
                (entry for entry in registry if entry.get("path") == configured_path),
                {},
            )
            options[language].append(
                {
                    "id": registry_entry.get("id", f"default-{language}"),
                    "path": str((ROOT_DIR / path).resolve()),
                    "weight": 400,
                    "style": "Regular",
                }
            )
        return options, None

    report_path = Path(font_validation_report_path)
    if not report_path.is_absolute():
        report_path = ROOT_DIR / report_path
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read font-validation report '{report_path}': {error}") from error
    fonts_by_id = {entry.get("id"): entry for entry in registry}
    options = {language: [] for language in SUPPORTED_LANGUAGES}
    for font in report.get("fonts", []):
        if font.get("eligible_for_generation") is not True:
            continue
        configured = fonts_by_id.get(font.get("id"))
        if (
            configured is None
            or configured.get("license_status") != "verified"
            or font.get("license_status") != "verified"
        ):
            continue
        path = (ROOT_DIR / configured["path"]).resolve()
        try:
            path.relative_to(ROOT_DIR.resolve())
        except ValueError as error:
            raise ValueError(f"Validated font path escapes the project: {path}") from error
        if not path.is_file() or font.get("path") != configured["path"]:
            continue
        passed_languages = {
            entry["language"]
            for entry in font.get("language_validation", [])
            if entry.get("passed") is True
        }
        for language in configured.get("target_languages", []):
            if language in options and language in passed_languages:
                options[language].append(
                    {
                        "id": font["id"],
                        "path": str(path),
                        "weight": font["weight"],
                        "style": font["style"],
                    }
                )
    missing = [language for language, fonts in options.items() if not fonts]
    if missing:
        raise ValueError(
            "No licensed, validated font is available for: " + ", ".join(missing)
        )
    try:
        report_reference = report_path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        report_reference = str(report_path.resolve())
    return options, report_reference


def _assign_fonts(
    schedule: Sequence[Tuple[str, str, str]],
    fonts_by_language: Dict[str, List[dict]],
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    font_orders = {}
    font_counts = {}
    for language, candidates in fonts_by_language.items():
        order = list(candidates)
        rng.shuffle(order)
        font_orders[language] = order
        font_counts[language] = {font["id"]: 0 for font in order}

    assignments = []
    for language, category, profile in schedule:
        candidates = font_orders[language]
        minimum_count = min(font_counts[language].values())
        least_used = [
            font for font in candidates
            if font_counts[language][font["id"]] == minimum_count
        ]
        font = rng.choice(least_used)
        font_counts[language][font["id"]] += 1
        assignments.append(
            {
                "language": language,
                "scene_category": category,
                "profile": profile,
                "font_id": font["id"],
                "font_path": font["path"],
                "font_weight": font["weight"],
                "font_style": font["style"],
            }
        )
    return assignments


def generate_reference_inspired_pilot(
    output_root: Path,
    preview_root: Path,
    seed: int = 20261029,
    *,
    sample_count: int = 24,
    font_validation_report_path: Optional[Path] = None,
    sample_id_prefix: str = "reference-inspired",
) -> dict:
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer.")
    if (
        not isinstance(sample_id_prefix, str)
        or not sample_id_prefix
        or any(not (char.isalnum() or char in "-_") for char in sample_id_prefix)
    ):
        raise ValueError("sample_id_prefix may contain only letters, digits, hyphens, and underscores.")
    config = load_config()
    configured_profiles = tuple(config["shop_board_generation"]["scene_difficulty"]["profiles"])
    if configured_profiles != PROFILE_NAMES:
        raise ValueError("Configured difficulty profiles differ from the supported profile set.")
    schedule = _make_schedule(sample_count, seed ^ 0x51CE)
    fonts_by_language, font_report_path = _load_generation_fonts(
        font_validation_report_path,
        sample_count,
    )
    scheduled_fonts = _assign_fonts(schedule, fonts_by_language, seed ^ 0xF017)
    root_rng = random.Random(seed)
    layout_rng = random.Random(seed ^ 0x1A70)
    background_policy = config["background_pipeline"]
    background_manifest_path = ROOT_DIR / background_policy["manifest_path"]
    background_manifest = load_background_manifest(
        background_manifest_path,
        project_root=ROOT_DIR,
        policy=background_policy,
    )
    assets_by_category: Dict[str, List[BackgroundAsset]] = {
        category: [] for category in SCENE_CATEGORIES
    }
    for asset in background_manifest.assets:
        assets_by_category[asset.record["scene_category"]].append(asset)
    asset_rng = random.Random(seed ^ 0xA55E)
    for assets in assets_by_category.values():
        assets.sort(key=lambda asset: asset.record["background_id"])
        asset_rng.shuffle(assets)
    asset_indices = {category: 0 for category in SCENE_CATEGORIES}
    assignments = []
    for index, (scheduled, font) in enumerate(zip(schedule, scheduled_fonts)):
        language, category, profile = scheduled
        sample_id = (
            f"{sample_id_prefix}-{seed & 0xFFFFFFFF:08x}-{index + 1:04d}-{language}"
        )
        asset_pool = assets_by_category[category]
        background_asset = None
        if asset_pool:
            background_asset = asset_pool[asset_indices[category] % len(asset_pool)]
            asset_indices[category] += 1
        assignment = {
            "sample_id": sample_id,
            "language": language,
            "scene_category": category,
            "profile": profile,
            "sample_seed": root_rng.getrandbits(64),
            "board_seed": root_rng.getrandbits(64),
            "layout_id": layout_rng.choice(SHOP_BOARD_LAYOUTS),
            **font,
            "background_asset": background_asset,
        }
        assignments.append(
            assignment
        )

    planned_paths = []
    for item in assignments:
        profile = item["profile"]
        sample_id = item["sample_id"]
        image_dir = output_root / profile / "images"
        annotation_dir = output_root / profile / "annotations"
        planned_paths.extend(
            [
                image_dir / f"{sample_id}.png",
                image_dir / "parents" / f"{sample_id}__base.png",
                image_dir / "board_layers" / f"{sample_id}_board.png",
                annotation_dir / f"{sample_id}.json",
                annotation_dir / "parents" / f"{sample_id}__base.json",
                preview_root / "overlays" / profile / f"{sample_id}_overlay.png",
            ]
        )
    for language in SUPPORTED_LANGUAGES:
        planned_paths.append(preview_root / "by_language" / f"{language}.png")
    all_font_ids = sorted({item["font_id"] for item in assignments})
    for font_id in all_font_ids:
        planned_paths.append(preview_root / "by_font" / f"{font_id}.png")
    planned_paths.append(preview_root / "generation_manifest.json")
    for profile in PROFILE_NAMES:
        planned_paths.append(preview_root / f"{profile}_contact_sheet.png")
    planned_paths.extend(
        [
            preview_root / "pilot_manifest.json",
            preview_root / "validation_report.json",
            preview_root / "generation_report.json",
        ]
    )
    if output_root.resolve() == preview_root.resolve():
        raise ValueError("Dataset output and preview directories must be different.")
    existing = [path for path in planned_paths if path.exists()]
    if output_root.exists():
        existing.append(output_root)
    if preview_root.exists():
        existing.append(preview_root)
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing reference-inspired pilot outputs: "
            + ", ".join(str(path) for path in existing)
        )

    profile_records: Dict[str, List[dict]] = {profile: [] for profile in PROFILE_NAMES}
    profile_paths: Dict[str, List[Path]] = {profile: [] for profile in PROFILE_NAMES}
    language_records: Dict[str, List[dict]] = {language: [] for language in SUPPORTED_LANGUAGES}
    language_paths: Dict[str, List[Path]] = {language: [] for language in SUPPORTED_LANGUAGES}
    font_records: Dict[str, List[dict]] = {font_id: [] for font_id in all_font_ids}
    font_paths: Dict[str, List[Path]] = {font_id: [] for font_id in all_font_ids}
    for item in assignments:
        sample_id = item["sample_id"]
        board_sample_id = f"{sample_id}__board"
        content_category = (
            "shop_board"
            if item["scene_category"] == "shopfront"
            else "advertising_hoarding"
        )
        image_dir = output_root / item["profile"] / "images"
        layer_relative_path = f"board_layers/{sample_id}_board.png"
        if item["background_asset"] is not None:
            image, base_annotation, board_layer = compose_background_scene(
                item["background_asset"],
                language=item["language"],
                sample_seed=item["sample_seed"],
                sample_id=sample_id,
                board_seed=item["board_seed"],
                layout_id=item["layout_id"],
                layer_relative_path=layer_relative_path,
                minimum_surface_width=background_policy["minimum_surface_width"],
                minimum_surface_height=background_policy["minimum_surface_height"],
                font_path=item["font_path"],
                font_id=item["font_id"],
                font_weight=item["font_weight"],
            )
            asset_record = item["background_asset"].record
            base_annotation["background_source_type"] = asset_record["source_type"]
            base_annotation["background_provenance"] = {
                "source_type": asset_record["source_type"],
                "asset_id": asset_record["asset_id"],
                "background_id": asset_record["background_id"],
                "relative_path": asset_record["relative_path"],
            }
        elif item["scene_category"] == "shopfront":
            board, board_annotation = render_shop_board_sample(
                item["language"],
                item["board_seed"],
                board_sample_id,
                item["layout_id"],
                content_category=content_category,
                font_path=item["font_path"],
                font_id=item["font_id"],
                font_weight=item["font_weight"],
            )
            image, base_annotation, board_layer = compose_storefront_scene(
                board,
                board_annotation,
                item["sample_seed"],
                sample_id,
                config,
            )
            base_annotation["scene_category"] = "shopfront"
            base_annotation["scene_parameters"].update(
                {
                    "background_source": "procedural_shapes",
                    "reference_inspiration": "layout, framing, and surface-context observations only",
                }
            )
            base_annotation["background_source_type"] = "procedural_illustration"
            base_annotation["background_provenance"] = {
                "source_type": "procedural_illustration",
                "asset_id": None,
                "photorealistic": False,
            }
        else:
            board, board_annotation = render_shop_board_sample(
                item["language"],
                item["board_seed"],
                board_sample_id,
                item["layout_id"],
                content_category=content_category,
                font_path=item["font_path"],
                font_id=item["font_id"],
                font_weight=item["font_weight"],
            )
            image, base_annotation, board_layer = _compose_hoarding_scene(
                board,
                board_annotation,
                item["sample_seed"],
                sample_id,
                config,
            )
            base_annotation["background_source_type"] = "procedural_illustration"
            base_annotation["background_provenance"] = {
                "source_type": "procedural_illustration",
                "asset_id": None,
                "photorealistic": False,
            }

        base_id = f"{sample_id}__base"
        base_annotation["sample_id"] = base_id
        base_annotation["image"] = f"{base_id}.png"
        base_annotation["random_seed"] = item["sample_seed"]
        base_annotation["source_random_seed"] = item["sample_seed"]
        base_annotation["scene_category"] = item["scene_category"]
        base_annotation["reference_usage"] = {
            "mode": "visual_reference_only",
            "reference_dataset_path": "reference images/",
            "reference_pixels_used": False,
            "reference_annotations_used_as_ground_truth": False,
            "approved_background_asset_used": item["background_asset"] is not None,
            "notes": "Reference photos informed composition only; locally rendered board text uses exact synthetic transcriptions, and any approved background asset is separately attributed.",
        }
        base_annotation.setdefault("scene_parameters", {})
        base_annotation["scene_parameters"]["reference_usage"] = copy.deepcopy(
            base_annotation["reference_usage"]
        )
        base_annotation["validation"] = {
            "status": "passed",
            "warnings": ["visual_review_required", "procedural_not_photorealistic"],
            "visual_review_required": True,
        }

        profile = item["profile"]
        image_dir = output_root / profile / "images"
        annotation_dir = output_root / profile / "annotations"
        parent_image_path = image_dir / "parents" / base_annotation["image"]
        parent_annotation_path = annotation_dir / "parents" / f"{base_id}.json"
        parent_image_path.parent.mkdir(parents=True, exist_ok=True)
        parent_annotation_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(parent_image_path, format="PNG")
        layer_path = image_dir / layer_relative_path
        layer_path.parent.mkdir(parents=True, exist_ok=True)
        board_layer.save(layer_path, format="PNG")
        base_annotation["board_layer"] = f"../board_layers/{layer_path.name}"
        base_errors = validate_annotation(base_annotation, parent_image_path)
        base_errors.extend(validate_rendered_geometry(base_annotation))
        if base_errors:
            raise ValueError(
                f"Base scene {base_id} failed validation: {'; '.join(base_errors)}"
            )
        write_annotation(base_annotation, parent_annotation_path)

        variant_input = copy.deepcopy(base_annotation)
        variant_input["sample_id"] = sample_id
        variant_input["image"] = f"{sample_id}.png"
        variant_input["random_seed"] = item["sample_seed"]
        variant_input["board_layer"] = f"board_layers/{layer_path.name}"
        variant_input["parent"] = {
            "sample_id": base_id,
            "image": f"parents/{base_annotation['image']}",
            "annotation": f"parents/{base_id}.json",
        }
        transformed, annotation = apply_profile(
            image,
            variant_input,
            profile,
            item["sample_seed"],
            config,
        )
        update_composed_surface_for_transform(annotation)
        annotation["scene_category"] = item["scene_category"]
        annotation["background_source_type"] = base_annotation["background_source_type"]
        annotation["background_provenance"] = copy.deepcopy(
            base_annotation["background_provenance"]
        )
        annotation["reference_usage"] = copy.deepcopy(base_annotation["reference_usage"])
        annotation["validation"] = {
            "status": "pending",
            "warnings": ["visual_review_required", "procedural_not_photorealistic"],
            "visual_review_required": True,
        }
        final_image_path = image_dir / annotation["image"]
        final_annotation_path = annotation_dir / f"{sample_id}.json"
        final_image_path.parent.mkdir(parents=True, exist_ok=True)
        transformed.save(final_image_path, format="PNG")
        errors = validate_annotation(annotation, final_image_path)
        errors.extend(validate_rendered_geometry(annotation))
        errors.extend(validate_transformed_geometry(base_annotation, annotation))
        if errors:
            raise ValueError(f"Variant {sample_id} failed validation: {'; '.join(errors)}")
        annotation["validation"]["status"] = "passed"
        annotation["validation"]["errors"] = []
        write_annotation(annotation, final_annotation_path)
        _make_overlay(
            transformed,
            annotation,
            preview_root / "overlays" / profile / f"{sample_id}_overlay.png",
        )
        profile_records[profile].append(annotation)
        profile_paths[profile].append(final_image_path)
        language_records[item["language"]].append(annotation)
        language_paths[item["language"]].append(final_image_path)
        font_records[item["font_id"]].append(annotation)
        font_paths[item["font_id"]].append(final_image_path)

    profile_reports = {}
    for profile in PROFILE_NAMES:
        image_dir = output_root / profile / "images"
        annotation_dir = output_root / profile / "annotations"
        report = validate_dataset(image_dir, annotation_dir)
        profile_reports[profile] = report
        report_path = preview_root / f"{profile}_validation.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_contact_sheet(
            profile_paths[profile],
            profile_records[profile],
            preview_root / f"{profile}_contact_sheet.png",
            group_key="profile",
        )
    records = [record for profile in PROFILE_NAMES for record in profile_records[profile]]
    for language in SUPPORTED_LANGUAGES:
        _write_contact_sheet(
            language_paths[language],
            language_records[language],
            preview_root / "by_language" / f"{language}.png",
            group_key="language",
        )
    for font_id in all_font_ids:
        _write_contact_sheet(
            font_paths[font_id],
            font_records[font_id],
            preview_root / "by_font" / f"{font_id}.png",
            group_key="font",
        )

    language_counts = Counter(record["language"] for record in records)
    scene_category_counts = Counter(record["scene_category"] for record in records)
    sign_category_counts = Counter(record["category"] for record in records)
    difficulty_counts = Counter(record["scene_difficulty"]["profile"] for record in records)
    font_counts = Counter(record["generation_parameters"]["font_id"] for record in records)
    font_by_language = {
        language: dict(
            sorted(
                Counter(
                    record["generation_parameters"]["font_id"]
                    for record in records
                    if record["language"] == language
                ).items()
            )
        )
        for language in SUPPORTED_LANGUAGES
    }
    background_counts = Counter(
        record["background_source_type"] for record in records
    )
    coverage = {
        "language_counts": dict(sorted(language_counts.items())),
        "scene_category_counts": dict(sorted(scene_category_counts.items())),
        "sign_category_counts": dict(sorted(sign_category_counts.items())),
        "difficulty_profile_counts": dict(sorted(difficulty_counts.items())),
        "font_counts": dict(sorted(font_counts.items())),
        "font_counts_by_language": font_by_language,
        "background_source_counts": dict(sorted(background_counts.items())),
    }
    expected = {
        "language_counts": dict(
            sorted(Counter(language for language, _, _ in schedule).items())
        ),
        "scene_category_counts": dict(
            sorted(Counter(category for _, category, _ in schedule).items())
        ),
        "difficulty_profile_counts": dict(
            sorted(Counter(profile for _, _, profile in schedule).items())
        ),
        "font_counts": dict(sorted(Counter(item["font_id"] for item in assignments).items())),
    }
    font_coverage_balanced = all(
        not counts or max(counts.values()) - min(counts.values()) <= 1
        for counts in font_by_language.values()
    )
    coverage_balanced = (
        coverage["language_counts"] == expected["language_counts"]
        and coverage["scene_category_counts"] == expected["scene_category_counts"]
        and coverage["difficulty_profile_counts"] == expected["difficulty_profile_counts"]
        and coverage["font_counts"] == expected["font_counts"]
        and font_coverage_balanced
    )
    validation_passed = all(report["passed"] for report in profile_reports.values())
    manifest = {
        "schema_version": "1.0",
        "seed": seed,
        "requested_scene_count": sample_count,
        "generated_scene_count": len(records),
        "font_validation_report": font_report_path,
        "background_manifest": str(background_manifest_path.relative_to(ROOT_DIR)),
        "reference_usage": {
            "mode": "visual_reference_only",
            "reference_dataset_path": "reference images/",
            "reference_pixels_used": False,
            "reference_annotations_used_as_ground_truth": False,
        },
        "scenes": [
            {
                "sample_id": record["sample_id"],
                "profile": record["scene_difficulty"]["profile"],
                "language": record["language"],
                "category": record["category"],
                "scene_category": record["scene_category"],
                "font_id": record["generation_parameters"]["font_id"],
                "background_source_type": record["background_source_type"],
                "image": (
                    f"{record['scene_difficulty']['profile']}/images/{record['image']}"
                ),
                "annotation": (
                    f"{record['scene_difficulty']['profile']}/annotations/"
                    f"{record['sample_id']}.json"
                ),
                "random_seed": record["random_seed"],
                "parent_sample_id": record.get("parent", {}).get("sample_id"),
            }
            for record in records
        ],
    }
    background_inventory = Counter(
        asset.record["source_type"] for asset in background_manifest.assets
    )
    selected_background_count = sum(
        background_counts.get(source_type, 0)
        for source_type in ("real_photo", "ai_generated")
    )
    excluded_background_count = len(background_manifest.excluded)
    failed_samples = [
        {
            "profile": profile,
            "file": error["file"],
            "error": error["error"],
        }
        for profile, validation in profile_reports.items()
        for error in validation["errors"]
    ]
    report = {
        "schema_version": "1.0",
        "seed": seed,
        "photorealistic": False,
        "description": "Reference-inspired synthetic dataset; procedural illustrations are not photorealistic.",
        "generated_scene_count": len(records),
        "requested_scene_count": sample_count,
        "coverage": coverage,
        "expected_coverage": expected,
        "coverage_balanced": coverage_balanced,
        "font_coverage_balanced_within_language": font_coverage_balanced,
        "validation_passed": validation_passed,
        "complete": len(records) == sample_count and coverage_balanced and validation_passed,
        "font_validation_report": font_report_path,
        "background_assets": {
            "eligible_asset_count": len(background_manifest.assets),
            "eligible_source_counts": dict(sorted(background_inventory.items())),
            "excluded_asset_count": excluded_background_count,
            "excluded_assets": list(background_manifest.excluded),
            "warnings": list(background_manifest.warnings),
            "selected_source_counts": {
                source: background_counts.get(source, 0)
                for source in ("real_photo", "ai_generated", "procedural_illustration")
            },
            "nonprocedural_background_shortfall": sample_count - selected_background_count,
            "procedural_fallback_used": background_counts.get("procedural_illustration", 0) > 0,
        },
        "failed_samples": failed_samples,
        "contact_sheets": {
            "by_language": {
                language: f"by_language/{language}.png" for language in SUPPORTED_LANGUAGES
            },
            "by_font": {
                font_id: f"by_font/{font_id}.png" for font_id in all_font_ids
            },
            "by_difficulty_profile": {
                profile: f"{profile}_contact_sheet.png" for profile in PROFILE_NAMES
            },
        },
        "profile_validation": {
            profile: {
                "passed": value["passed"],
                "image_count": value["image_count"],
                "annotation_count": value["annotation_count"],
                "errors": value["errors"],
            }
            for profile, value in profile_reports.items()
        },
        "visual_review_required": True,
        "limitations": [
            "Scene backgrounds are procedural 2D illustrations, not photorealistic.",
            "Reference photos were not used as pixels or as annotation ground truth.",
            "Visual inspection of the contact sheets remains necessary; automated geometry checks do not establish visual realism.",
            "Occlusion visibility remains the configured bbox-area estimate, not pixel-accurate glyph visibility.",
        ],
    }
    preview_root.mkdir(parents=True, exist_ok=True)
    (preview_root / "pilot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (preview_root / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (preview_root / "validation_report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "passed": validation_passed,
                "generated_scene_count": len(records),
                "coverage": coverage,
                "profiles": report["profile_validation"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (preview_root / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
