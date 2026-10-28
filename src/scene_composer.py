from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from .content_engine import SUPPORTED_LANGUAGES, load_config
from .signboard_generator import (
    SHOP_BOARD_LAYOUTS,
    _make_contact_sheet,
    render_shop_board_sample,
)
from .annotation_exporter import validate_annotation, validate_rendered_geometry, write_annotation

ROOT_DIR = Path(__file__).resolve().parents[1]


def _draw_storefront_background(
    size: Tuple[int, int],
    wall_color: Tuple[int, int, int],
    trim_color: Tuple[int, int, int],
    window_color: Tuple[int, int, int],
    awning_color: Tuple[int, int, int],
    awning_y: int,
) -> Image.Image:
    width, height = size
    scene = Image.new("RGBA", size, (*wall_color, 255))
    draw = ImageDraw.Draw(scene)
    draw.rectangle((0, 0, width - 1, 24), fill=(*trim_color, 255))
    draw.rectangle((0, awning_y - 10, width - 1, awning_y), fill=(*trim_color, 255))

    awning_bottom = min(awning_y + 88, height - 220)
    stripe_width = max(width // 10, 1)
    for left in range(0, width, stripe_width):
        right = min(left + stripe_width - 1, width - 1)
        color = awning_color if (left // stripe_width) % 2 == 0 else wall_color
        draw.polygon(
            [
                (left, awning_y + 1),
                (right, awning_y + 1),
                (right - 10, awning_bottom),
                (left + 10, awning_bottom),
            ],
            fill=(*color, 255),
        )
    draw.rectangle(
        (0, awning_bottom, width - 1, awning_bottom + 12),
        fill=(*trim_color, 255),
    )

    storefront_top = awning_bottom + 30
    frame_width = max(width // 45, 12)
    window_left = width // 8
    window_right = width - window_left
    window_bottom = height - 42
    draw.rectangle(
        (window_left, storefront_top, window_right, window_bottom),
        fill=(*trim_color, 255),
    )
    pane_inset = frame_width
    draw.rectangle(
        (
            window_left + pane_inset,
            storefront_top + pane_inset,
            window_right - pane_inset,
            window_bottom - pane_inset,
        ),
        fill=(*window_color, 255),
    )
    center_x = width // 2
    draw.rectangle(
        (center_x - frame_width // 2, storefront_top, center_x + frame_width // 2, window_bottom),
        fill=(*trim_color, 255),
    )
    draw.rectangle(
        (window_left, storefront_top + (window_bottom - storefront_top) // 2,
         window_right, storefront_top + (window_bottom - storefront_top) // 2 + frame_width),
        fill=(*trim_color, 255),
    )
    return scene


def compose_storefront_scene(
    board: Image.Image,
    board_annotation: dict,
    scene_seed: int,
    scene_id: str,
    config: Optional[dict] = None,
) -> Tuple[Image.Image, dict, Image.Image]:
    if config is None:
        config = load_config()
    generation = config["shop_board_generation"]
    scene_config = generation["storefront_scene"]
    rng = random.Random(scene_seed)
    scene_width, scene_height = rng.choice(scene_config["scene_sizes"])
    board_top = rng.choice(scene_config["board_top_margin_px"])
    board_x = (scene_width - board.width) // 2
    if board_x < 0 or board_top < 0 or board_top + board.height > scene_height:
        raise ValueError("Configured signboard does not fit inside the storefront scene.")

    awning_y = board_top + board.height + 42
    if awning_y + 88 > scene_height - 220:
        raise ValueError("Signboard leaves insufficient room for storefront details.")
    wall_color = tuple(rng.choice(scene_config["wall_colors"]))
    trim_color = tuple(rng.choice(scene_config["trim_colors"]))
    window_color = tuple(rng.choice(scene_config["window_colors"]))
    awning_color = tuple(rng.choice(scene_config["awning_colors"]))
    scene = _draw_storefront_background(
        (scene_width, scene_height),
        wall_color,
        trim_color,
        window_color,
        awning_color,
        awning_y,
    )
    scene.alpha_composite(board, dest=(board_x, board_top))

    annotation = dict(board_annotation)
    annotation["sample_id"] = scene_id
    annotation["image"] = f"{scene_id}.png"
    annotation["image_width"] = scene_width
    annotation["image_height"] = scene_height
    annotation["board_layer"] = f"{scene_id}_board.png"
    annotation["source_board_sample_id"] = board_annotation["sample_id"]
    annotation["scene_transform"] = {
        "type": "translation_only",
        "translate_x": board_x,
        "translate_y": board_top,
        "scale": 1.0,
        "rotation_degrees": 0.0,
        "perspective": False,
    }
    annotation["scene_parameters"] = {
        "template_id": "procedural_storefront_v1",
        "wall_rgb": list(wall_color),
        "trim_rgb": list(trim_color),
        "window_rgb": list(window_color),
        "awning_rgb": list(awning_color),
        "awning_y": awning_y,
        "random_seed": scene_seed,
        "visual_review_required": True,
    }
    annotation["generation_parameters"] = dict(board_annotation["generation_parameters"])
    annotation["generation_parameters"]["board_position_in_scene"] = {
        "x": board_x,
        "y": board_top,
        "width": board.width,
        "height": board.height,
    }

    for region in annotation["text_regions"]:
        region["board_bbox"] = dict(region["bbox"])
        region["board_polygon"] = [list(point) for point in region["polygon"]]
        region["bbox"] = {
            "x_min": region["board_bbox"]["x_min"] + board_x,
            "y_min": region["board_bbox"]["y_min"] + board_top,
            "x_max": region["board_bbox"]["x_max"] + board_x,
            "y_max": region["board_bbox"]["y_max"] + board_top,
        }
        region["polygon"] = [
            [x + board_x, y + board_top]
            for x, y in region["board_polygon"]
        ]

    annotation["validation"] = {
        "status": "pending",
        "warnings": ["visual_review_required"],
        "visual_review_required": True,
    }
    return scene, annotation, board


def generate_storefront_scenes(
    count: int,
    seed: int,
    image_dir: Path,
    annotation_dir: Path,
    contact_sheet_path: Optional[Path] = None,
) -> List[dict]:
    if count < len(SUPPORTED_LANGUAGES) or count % len(SUPPORTED_LANGUAGES) != 0:
        raise ValueError(
            f"count must be a positive multiple of {len(SUPPORTED_LANGUAGES)} "
            "so all four languages are represented equally."
        )
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")

    config = load_config()
    board_layouts: Sequence[str] = config["shop_board_generation"].get(
        "layouts", SHOP_BOARD_LAYOUTS
    )
    if not board_layouts:
        raise ValueError("At least one board layout must be configured.")
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    root_rng = random.Random(seed)
    languages = list(SUPPORTED_LANGUAGES) * (count // len(SUPPORTED_LANGUAGES))
    root_rng.shuffle(languages)
    sample_seeds = [root_rng.getrandbits(64) for _ in range(count)]
    scene_ids = [
        f"storefront-{seed & 0xFFFFFFFF:08x}-{index + 1:04d}-{language}"
        for index, language in enumerate(languages)
    ]
    layout_rng = random.Random(seed ^ 0xC0FFEE)
    scheduled_layouts = (
        list(board_layouts)
        * ((count + len(board_layouts) - 1) // len(board_layouts))
    )[:count]
    layout_rng.shuffle(scheduled_layouts)

    planned_paths = []
    for scene_id in scene_ids:
        planned_paths.extend(
            [
                image_dir / f"{scene_id}.png",
                image_dir / f"{scene_id}_board.png",
                annotation_dir / f"{scene_id}.json",
            ]
        )
    if contact_sheet_path is not None:
        planned_paths.append(contact_sheet_path)
    collisions = [path for path in planned_paths if path.exists()]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing output files: "
            + ", ".join(str(path) for path in collisions)
        )

    records: List[dict] = []
    scene_paths: List[Path] = []
    for index, (language, sample_seed, scene_id) in enumerate(
        zip(languages, sample_seeds, scene_ids)
    ):
        board_seed = root_rng.getrandbits(64)
        board_id = f"{scene_id}-board"
        board, board_annotation = render_shop_board_sample(
            language,
            board_seed,
            board_id,
            scheduled_layouts[index],
        )
        scene, annotation, board_layer = compose_storefront_scene(
            board,
            board_annotation,
            sample_seed,
            scene_id,
            config,
        )
        image_path = image_dir / annotation["image"]
        board_path = image_dir / annotation["board_layer"]
        annotation_path = annotation_dir / f"{scene_id}.json"
        board_layer.save(board_path, format="PNG")
        scene.save(image_path, format="PNG")

        validation_errors = validate_annotation(annotation, image_path)
        validation_errors.extend(validate_rendered_geometry(annotation))
        annotation["validation"]["status"] = "passed" if not validation_errors else "failed"
        annotation["validation"]["errors"] = validation_errors
        write_annotation(annotation, annotation_path)
        if validation_errors:
            raise ValueError(
                f"Generated scene {scene_id} failed validation: "
                + "; ".join(validation_errors)
            )
        records.append(annotation)
        scene_paths.append(image_path)

    if contact_sheet_path is not None:
        _make_contact_sheet(scene_paths, languages, contact_sheet_path)
    return records
