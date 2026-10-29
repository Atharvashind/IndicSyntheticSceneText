from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .annotation_exporter import SCRIPT_BY_LANGUAGE, validate_annotation, write_annotation
from .content_engine import SUPPORTED_LANGUAGES, generate_text, load_config
from .text_renderer import render_multiline_text

ROOT_DIR = Path(__file__).resolve().parents[1]
SHOP_BOARD_LAYOUTS = (
    "centered_business_subtitle",
    "left_name_services",
    "headline_contact_details",
    "multiline_hierarchy",
    "logo_business_sign",
)


def _choose(rng: random.Random, values: Sequence):
    if not values:
        raise ValueError("Generation configuration lists must not be empty.")
    return rng.choice(values)


def _contrast_ratio(
    first: Tuple[int, int, int],
    second: Tuple[int, int, int],
) -> float:
    def relative_luminance(color: Tuple[int, int, int]) -> float:
        channels = []
        for value in color:
            component = value / 255
            channels.append(
                component / 12.92
                if component <= 0.04045
                else ((component + 0.055) / 1.055) ** 2.4
            )
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    lighter, darker = sorted(
        (relative_luminance(first), relative_luminance(second)),
        reverse=True,
    )
    return round((lighter + 0.05) / (darker + 0.05), 3)


def _draw_board_background(
    size: Tuple[int, int],
    background_color: Tuple[int, int, int],
    border_color: Tuple[int, int, int],
    border_width: int,
) -> Image.Image:
    width, height = size
    board = Image.new("RGBA", size, (*background_color, 255))
    draw = ImageDraw.Draw(board)
    inset = max(border_width // 2, 1)
    draw.rectangle(
        (inset, inset, width - inset - 1, height - inset - 1),
        outline=(*border_color, 255),
        width=border_width,
    )
    inner_inset = border_width + max(border_width // 2, 2)
    draw.rounded_rectangle(
        (inner_inset, inner_inset, width - inner_inset - 1, height - inner_inset - 1),
        radius=max(border_width * 2, 8),
        outline=(*border_color, 255),
        width=max(border_width // 3, 1),
    )

    marker_radius = max(border_width, 5)
    for x in (inner_inset + marker_radius, width - inner_inset - marker_radius - 1):
        y = inner_inset + marker_radius
        draw.ellipse(
            (x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius),
            fill=(*border_color, 255),
        )
    return board


def _draw_simple_logo(
    board: Image.Image,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
) -> dict:
    center_x, center_y = center
    draw = ImageDraw.Draw(board)
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill=(*color, 255),
    )
    draw.ellipse(
        (
            center_x - radius + 7,
            center_y - radius + 7,
            center_x + radius - 7,
            center_y + radius - 7,
        ),
        outline=(255, 255, 255, 255),
        width=max(radius // 12, 2),
    )
    roof_y = center_y - radius // 3
    draw.polygon(
        [
            (center_x - radius // 2, roof_y),
            (center_x, center_y - radius * 2 // 3),
            (center_x + radius // 2, roof_y),
        ],
        fill=(255, 255, 255, 255),
    )
    draw.rectangle(
        (
            center_x - radius // 3,
            roof_y,
            center_x + radius // 3,
            center_y + radius // 2,
        ),
        fill=(255, 255, 255, 255),
    )
    draw.rectangle(
        (center_x - radius // 10, center_y + radius // 8, center_x + radius // 10, center_y + radius // 2),
        fill=(*color, 255),
    )
    return {
        "type": "procedural_storefront_emblem",
        "center": [center_x, center_y],
        "radius_px": radius,
        "color_rgb": list(color),
        "contains_text": False,
    }


def _make_region_annotation(
    *,
    region_id: str,
    transcription: str,
    language: str,
    reading_order: int,
    render_result: dict,
    canvas_origin: Tuple[int, int],
    alignment: str,
    foreground: Tuple[int, int, int],
    font_id: Optional[str],
    font_weight: Optional[int],
    outline_width: int,
    shadow_offset: Tuple[int, int],
) -> dict:
    ink_bbox = render_result["ink_bbox"]
    origin_x, origin_y = canvas_origin
    x_min = origin_x + ink_bbox["x_min"]
    y_min = origin_y + ink_bbox["y_min"]
    x_max = origin_x + ink_bbox["x_max"]
    y_max = origin_y + ink_bbox["y_max"]
    font_path = Path(render_result["font_path"]).resolve()
    try:
        font_record = font_path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        font_record = font_path.name
    resolved_font_id = font_id or (
        f"{render_result['font_family'].lower().replace(' ', '-')}-"
        f"{render_result['font_style'].lower().replace(' ', '-')}"
    )

    return {
        "region_id": region_id,
        "transcription": transcription,
        "language": language,
        "script": SCRIPT_BY_LANGUAGE[language],
        "reading_order": reading_order,
        "bbox": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },
        "polygon": [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        "font": {
            "id": resolved_font_id,
            "family": render_result["font_family"],
            "path": font_record,
            "style": render_result["font_style"],
            "weight": font_weight or render_result["font_weight"],
            "size_px": render_result["font_size"],
        },
        "rendering": {
            "shaping_engine": render_result["shaping_engine"],
            "alignment": alignment,
            "foreground_rgb": list(foreground),
            "outline_width_px": outline_width,
            "shadow_offset_px": list(shadow_offset),
        },
    }


def _render_shop_board(
    language: str,
    sample_seed: int,
    sample_id: str,
    config: dict,
    layout_id: str,
    content_category: str = "shop_board",
    font_path: Optional[str] = None,
    font_id: Optional[str] = None,
    font_weight: Optional[int] = None,
) -> Tuple[Image.Image, dict]:
    generation = config["shop_board_generation"]
    rng = random.Random(sample_seed)
    if layout_id not in SHOP_BOARD_LAYOUTS:
        raise ValueError(f"Unknown shop-board layout '{layout_id}'.")
    if content_category not in {"shop_board", "advertising_hoarding"}:
        raise ValueError(f"Unsupported signboard content category '{content_category}'.")
    width, height = _choose(rng, generation["board_sizes"])
    background_color = tuple(_choose(rng, generation["background_colors"]))
    border_color = tuple(_choose(rng, generation["border_colors"]))
    minimum_contrast = float(generation.get("minimum_text_contrast_ratio", 1.0))
    eligible_text_colors = [
        tuple(color)
        for color in generation["text_colors"]
        if _contrast_ratio(tuple(color), background_color) >= minimum_contrast
    ]
    if not eligible_text_colors:
        raise ValueError(
            "No configured text color meets the minimum contrast ratio for the selected board background."
        )
    foreground = tuple(_choose(rng, eligible_text_colors))
    border_width = int(_choose(rng, generation["border_widths"]))
    alignment = _choose(rng, generation["alignments"])
    outline_width = int(generation["outline_width"])
    shadow_offset = tuple(generation.get("shadow_offset", [0, 0]))
    outer_padding = int(generation["outer_padding"])
    logo_record = None

    board = _draw_board_background(
        (width, height),
        background_color,
        border_color,
        border_width,
    )
    full_width = width - 2 * outer_padding
    if full_width <= 0:
        raise ValueError("Configured outer padding leaves no room for text regions.")

    specs_by_layout = {
        "centered_business_subtitle": [
            ("shop_name", "shop_name", round(height * 0.16), round(height * 0.37), generation["font_sizes"], "main"),
            ("storefront_description", "storefront_description", round(height * 0.61), round(height * 0.20), generation["description_font_sizes"], "secondary"),
        ],
        "left_name_services": [
            ("shop_name", "shop_name", round(height * 0.13), round(height * 0.32), generation["font_sizes"], "main"),
            ("service_list", "service_list", round(height * 0.48), round(height * 0.36), generation["description_font_sizes"], "secondary"),
        ],
        "headline_contact_details": [
            ("shop_name", "shop_name", round(height * 0.13), round(height * 0.35), generation["font_sizes"], "main"),
            ("storefront_description", "storefront_description", round(height * 0.52), round(height * 0.17), generation["description_font_sizes"], "secondary"),
            ("contact_details", "contact_details", round(height * 0.72), round(height * 0.13), [22, 24, 26, 28], "tertiary"),
        ],
        "multiline_hierarchy": [
            ("multiline_shop_name", "multiline_shop_name", round(height * 0.12), round(height * 0.49), [42, 48, 54], "main"),
            ("service_list", "service_list", round(height * 0.62), round(height * 0.26), generation["description_font_sizes"], "secondary"),
        ],
        "logo_business_sign": [
            ("shop_name", "shop_name", round(height * 0.24), round(height * 0.30), generation["font_sizes"], "main"),
            ("storefront_description", "storefront_description", round(height * 0.59), round(height * 0.18), generation["description_font_sizes"], "secondary"),
        ],
    }
    if content_category == "advertising_hoarding":
        specs_by_layout = {
            "centered_business_subtitle": [
                ("headline", "advertising_message", round(height * 0.16), round(height * 0.37), generation["font_sizes"], "main"),
                ("offer", "offer", round(height * 0.61), round(height * 0.20), generation["description_font_sizes"], "secondary"),
            ],
            "left_name_services": [
                ("headline", "advertising_message", round(height * 0.13), round(height * 0.32), generation["font_sizes"], "main"),
                ("offer", "offer", round(height * 0.48), round(height * 0.36), generation["description_font_sizes"], "secondary"),
            ],
            "headline_contact_details": [
                ("headline", "advertising_message", round(height * 0.13), round(height * 0.35), generation["font_sizes"], "main"),
                ("offer", "offer", round(height * 0.52), round(height * 0.17), generation["description_font_sizes"], "secondary"),
                ("details", "offer", round(height * 0.72), round(height * 0.13), [22, 24, 26, 28], "tertiary"),
            ],
            "multiline_hierarchy": [
                ("headline", "advertising_message", round(height * 0.12), round(height * 0.49), [42, 48, 54], "main"),
                ("offer", "offer", round(height * 0.62), round(height * 0.26), generation["description_font_sizes"], "secondary"),
            ],
            "logo_business_sign": [
                ("headline", "advertising_message", round(height * 0.24), round(height * 0.30), generation["font_sizes"], "main"),
                ("offer", "offer", round(height * 0.59), round(height * 0.18), generation["description_font_sizes"], "secondary"),
            ],
        }
    specifications = specs_by_layout[layout_id]
    logo_reserve = round(width * 0.21) if layout_id == "logo_business_sign" else 0
    region_x = outer_padding + logo_reserve
    region_width = width - region_x - outer_padding
    if region_width <= 0:
        raise ValueError("Configured logo and padding leave no room for text regions.")
    if layout_id == "logo_business_sign":
        logo_radius = min(round(height * 0.20), round(width * 0.085))
        logo_center = (outer_padding + logo_radius + 12, height // 2)
        logo_color = tuple(_choose(rng, generation["logo_colors"]))
        logo_record = _draw_simple_logo(board, logo_center, logo_radius, logo_color)

    alignment = {
        "centered_business_subtitle": "center",
        "left_name_services": "left",
        "headline_contact_details": "center",
        "multiline_hierarchy": "center",
        "logo_business_sign": "left",
    }[layout_id]
    text_regions = []
    layout_records = []
    for reading_order, specification in enumerate(specifications):
        region_id, content_type, region_y, region_height, size_options, hierarchy = specification
        text_seed = rng.getrandbits(64)
        transcription = generate_text(
            language,
            content_category,
            seed=text_seed,
            content_type=content_type,
        )
        font_size = int(_choose(rng, size_options))
        region_alignment = alignment
        render_result = render_multiline_text(
            transcription,
            language,
            width=region_width,
            height=region_height,
            font_size=font_size,
            foreground=foreground,
            background=None,
            padding=8 if hierarchy == "tertiary" else 10,
            line_spacing=1.12,
            align=region_alignment,
            outline_width=outline_width,
            shadow_offset=shadow_offset,
            font_path=font_path,
        )
        region_origin = (region_x, region_y)
        board.alpha_composite(render_result["image"], dest=region_origin)
        text_regions.append(
            _make_region_annotation(
                region_id=region_id,
                transcription=transcription,
                language=language,
                reading_order=reading_order,
                render_result=render_result,
                canvas_origin=region_origin,
                alignment=region_alignment,
                foreground=foreground,
                font_id=font_id,
                font_weight=font_weight,
                outline_width=outline_width,
                shadow_offset=shadow_offset,
            )
        )
        layout_records.append(
            {
                "region_id": region_id,
                "canvas_x": region_origin[0],
                "canvas_y": region_origin[1],
                "canvas_width": region_width,
                "canvas_height": region_height,
                "font_size_px": font_size,
                "padding_px": 8 if hierarchy == "tertiary" else 10,
                "line_spacing": 1.12,
                "content_type": content_type,
                "hierarchy": hierarchy,
                "alignment": region_alignment,
            }
        )

    annotation = {
        "schema_version": "1.0",
        "sample_id": sample_id,
        "image": f"{sample_id}.png",
        "image_width": width,
        "image_height": height,
        "language": language,
        "script": SCRIPT_BY_LANGUAGE[language],
        "category": content_category,
        "random_seed": sample_seed,
        "coordinate_convention": {
            "origin": "top_left",
            "x_direction": "right",
            "y_direction": "down",
            "coordinates": "pixel_edges; minimum edges inclusive and maximum edges exclusive",
        },
        "text_regions": text_regions,
        "generation_parameters": {
            "template_id": layout_id,
            "content_category": content_category,
            "font_id": text_regions[0]["font"]["id"],
            "font_weight": text_regions[0]["font"]["weight"],
            "font_path": text_regions[0]["font"]["path"],
            "background_rgb": list(background_color),
            "border_rgb": list(border_color),
            "text_rgb": list(foreground),
            "text_background_contrast_ratio": _contrast_ratio(
                foreground,
                background_color,
            ),
            "minimum_text_contrast_ratio": minimum_contrast,
            "border_width_px": border_width,
            "outer_padding_px": outer_padding,
            "alignment": alignment,
            "logo": logo_record,
            "rendering_effects": {
                "outline_width_px": outline_width,
                "shadow_offset_px": list(shadow_offset),
            },
            "region_layout": layout_records,
        },
        "validation": {
            "status": "pending",
            "warnings": ["visual_review_required"],
            "visual_review_required": True,
        },
    }
    return board, annotation


def render_shop_board_sample(
    language: str,
    sample_seed: int,
    sample_id: str,
    layout_id: str,
    content_category: str = "shop_board",
    font_path: Optional[str] = None,
    font_id: Optional[str] = None,
    font_weight: Optional[int] = None,
) -> Tuple[Image.Image, dict]:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language '{language}'.")
    return _render_shop_board(
        language,
        sample_seed,
        sample_id,
        load_config(),
        layout_id,
        content_category,
        font_path,
        font_id,
        font_weight,
    )


def _make_contact_sheet(
    image_paths: Sequence[Path],
    languages: Sequence[str],
    destination: Path,
) -> None:
    columns = 2
    cell_width, cell_height = 520, 330
    rows = (len(image_paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    caption_font = ImageFont.load_default()
    for index, (image_path, language) in enumerate(zip(image_paths, languages)):
        with Image.open(image_path) as source:
            sample = source.convert("RGB")
            sample.thumbnail((cell_width - 24, cell_height - 48), Image.Resampling.LANCZOS)
        x = (index % columns) * cell_width + (cell_width - sample.width) // 2
        y = (index // columns) * cell_height + 8
        sheet.paste(sample, (x, y))
        draw.text(
            ((index % columns) * cell_width + 12, (index // columns) * cell_height + cell_height - 30),
            f"{image_path.stem} | {language}",
            font=caption_font,
            fill=(20, 20, 20),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def generate_shop_boards(
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

    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    root_rng = random.Random(seed)
    languages = list(SUPPORTED_LANGUAGES) * (count // len(SUPPORTED_LANGUAGES))
    root_rng.shuffle(languages)
    sample_seeds = [root_rng.getrandbits(64) for _ in range(count)]
    sample_ids = [
        f"shop-{seed & 0xFFFFFFFF:08x}-{index + 1:04d}-{language}"
        for index, language in enumerate(languages)
    ]

    planned_paths = []
    for sample_id in sample_ids:
        planned_paths.extend(
            [image_dir / f"{sample_id}.png", annotation_dir / f"{sample_id}.json"]
        )
    if contact_sheet_path is not None:
        planned_paths.append(contact_sheet_path)
    collisions = [path for path in planned_paths if path.exists()]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing output files: "
            + ", ".join(str(path) for path in collisions)
        )

    config = load_config()
    records = []
    saved_images: List[Path] = []
    layout_ids = list(config["shop_board_generation"]["layouts"])
    if not layout_ids:
        raise ValueError("At least one shop-board layout must be configured.")
    layout_rng = random.Random(seed ^ 0x5EED)
    scheduled_layouts = (
        layout_ids * ((count + len(layout_ids) - 1) // len(layout_ids))
    )[:count]
    layout_rng.shuffle(scheduled_layouts)
    for language, sample_seed, sample_id in zip(languages, sample_seeds, sample_ids):
        image, annotation = _render_shop_board(
            language,
            sample_seed,
            sample_id,
            config,
            scheduled_layouts[len(records)],
        )
        image_path = image_dir / annotation["image"]
        annotation_path = annotation_dir / f"{sample_id}.json"
        image.save(image_path, format="PNG")

        validation_errors = validate_annotation(annotation, image_path)
        annotation["validation"]["status"] = "passed" if not validation_errors else "failed"
        annotation["validation"]["errors"] = validation_errors
        write_annotation(annotation, annotation_path)
        if validation_errors:
            raise ValueError(
                f"Generated sample {sample_id} failed annotation validation: "
                + "; ".join(validation_errors)
            )
        records.append(annotation)
        saved_images.append(image_path)

    if contact_sheet_path is not None:
        _make_contact_sheet(saved_images, languages, contact_sheet_path)
    return records
