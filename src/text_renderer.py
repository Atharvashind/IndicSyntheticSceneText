from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import freetype
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageFilter

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "configs" / "generation_config.json"
DEFAULT_PREVIEW_DIR = ROOT_DIR / "previews"

LANGUAGE_SHAPING = {
    "marathi": ("Deva", "mr"),
    "hindi": ("Deva", "hi"),
    "gujarati": ("Gujr", "gu"),
    "bengali": ("Beng", "bn"),
}
TEST_STRINGS = {
    "marathi": "नमस्कार, महाराष्ट्र!\nक्षेत्र आणि ज्ञान",
    "hindi": "नमस्ते, भारत!\nक्षेत्र और प्रज्ञा",
    "gujarati": "નમસ્તે, ગુજરાત!\nક્ષેત્ર અને જ્ઞાન",
    "bengali": "নমস্কার, বাংলা!\nক্ষেত্র ও জ্ঞান",
}
SHAPING_CONTROLS = {"\u200c", "\u200d"}


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_font_path(language: str, custom_paths: Optional[Sequence[str]] = None) -> str:
    if language not in LANGUAGE_SHAPING:
        raise ValueError(f"Unsupported language '{language}'.")

    if custom_paths is not None:
        candidates = [Path(path) for path in custom_paths]
    else:
        configured = load_config().get("font_paths", {}).get(language)
        if not configured:
            raise FileNotFoundError(f"No font is configured for '{language}' in {CONFIG_PATH}.")
        if isinstance(configured, str):
            configured = [configured]
        candidates = [ROOT_DIR / path for path in configured]

    for path in candidates:
        if path.is_file() and path.suffix.lower() in {".ttf", ".otf"}:
            return str(path.resolve())

    requested = ", ".join(str(path) for path in candidates) or "(no paths provided)"
    raise FileNotFoundError(
        f"No usable font file was found for '{language}' at {requested}. "
        "Check the configured local Noto Sans font path."
    )


def _validate_font_coverage(text: str, font_path: str) -> None:
    font = TTFont(font_path, lazy=True)
    try:
        cmap = font.getBestCmap()
        missing = sorted(
            {
                char
                for char in text
                if not char.isspace()
                and char not in SHAPING_CONTROLS
                and ord(char) not in cmap
            },
            key=ord,
        )
    finally:
        font.close()
    if missing:
        codepoints = ", ".join(f"U+{ord(char):04X} ({char})" for char in missing)
        raise ValueError(f"Font '{font_path}' does not cover required characters: {codepoints}")


def _font_names(font_path: str) -> Tuple[str, str]:
    font = TTFont(font_path, lazy=True)
    try:
        family = font["name"].getDebugName(1)
        style = font["name"].getDebugName(2)
    finally:
        font.close()
    if not family or not style:
        raise ValueError(f"Font '{font_path}' does not contain family and style metadata.")
    return family, style


def _shape_line(text: str, language: str, font_data: bytes, font_size: int) -> Tuple[List[int], List[object], float]:
    script, language_tag = LANGUAGE_SHAPING[language]
    face = hb.Face(font_data)
    font = hb.Font(face)
    hb.ot_font_set_funcs(font)
    font.scale = (font_size * 64, font_size * 64)

    buffer = hb.Buffer()
    buffer.add_str(text)
    buffer.direction = "ltr"
    buffer.script = script
    buffer.language = language_tag
    hb.shape(font, buffer)
    glyph_ids = list(buffer.glyph_infos)
    positions = list(buffer.glyph_positions)
    if any(info.codepoint == 0 for info in glyph_ids):
        raise ValueError(f"HarfBuzz produced a missing glyph while shaping {language} text: {text!r}")
    advance = sum(position.x_advance for position in positions) / 64
    return [info.codepoint for info in glyph_ids], positions, advance


def _wrap_lines(
    text: str,
    language: str,
    font_data: bytes,
    font_size: int,
    max_width: int,
) -> List[str]:
    wrapped: List[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            wrapped.append("")
            continue
        words = paragraph.split()
        if not words:
            wrapped.append("")
            continue
        current = words[0]
        _, _, word_width = _shape_line(current, language, font_data, font_size)
        if word_width > max_width:
            raise ValueError(
                f"A word is wider than the available text area ({word_width:.1f}px > {max_width}px); "
                "increase the canvas width or reduce font size."
            )
        for word in words[1:]:
            candidate = f"{current} {word}"
            _, _, candidate_width = _shape_line(candidate, language, font_data, font_size)
            if candidate_width <= max_width:
                current = candidate
            else:
                wrapped.append(current)
                current = word
                _, _, word_width = _shape_line(current, language, font_data, font_size)
                if word_width > max_width:
                    raise ValueError(
                        f"A word is wider than the available text area ({word_width:.1f}px > {max_width}px); "
                        "increase the canvas width or reduce font size."
                    )
        wrapped.append(current)
    return wrapped


def _glyph_mask(face: freetype.Face, glyph_id: int) -> Image.Image:
    face.load_glyph(glyph_id, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
    bitmap = face.glyph.bitmap
    width, rows = bitmap.width, bitmap.rows
    if width == 0 or rows == 0:
        return Image.new("L", (0, 0))
    pitch = abs(bitmap.pitch)
    raw = bytes(bitmap.buffer)
    if bitmap.pixel_mode == freetype.FT_PIXEL_MODE_GRAY:
        pixels = b"".join(raw[row * pitch : row * pitch + width] for row in range(rows))
        return Image.frombytes("L", (width, rows), pixels)
    if bitmap.pixel_mode == freetype.FT_PIXEL_MODE_MONO:
        pixels = bytearray(width * rows)
        for row in range(rows):
            for col in range(width):
                pixels[row * width + col] = (
                    255 if raw[row * pitch + col // 8] & (0x80 >> (col % 8)) else 0
                )
        return Image.frombytes("L", (width, rows), bytes(pixels))
    raise ValueError(f"Unsupported FreeType bitmap pixel mode: {bitmap.pixel_mode}")


def _make_text_mask(
    lines: List[str],
    language: str,
    font_data: bytes,
    font_path: str,
    font_size: int,
    canvas_size: Tuple[int, int],
    padding: int,
    line_spacing: float,
    align: str,
) -> Tuple[Image.Image, List[dict], int]:
    width, height = canvas_size
    face = freetype.Face(font_path)
    face.set_pixel_sizes(0, font_size)
    ascender = face.size.ascender / 64
    descender = abs(face.size.descender / 64)
    natural_line_height = max(face.size.height / 64, ascender + descender)
    line_step = max(int(natural_line_height * line_spacing), font_size)
    total_height = ascender + descender + max(len(lines) - 1, 0) * line_step
    if total_height > height - 2 * padding:
        raise ValueError(
            "Rendered text exceeds the provided canvas height; increase height or reduce font size."
        )

    start_baseline = max((height - total_height) / 2 + ascender, padding + ascender)
    mask = Image.new("L", canvas_size, 0)
    text_boxes = []
    glyph_count = 0
    for line_index, line in enumerate(lines):
        glyph_ids, positions, advance = _shape_line(line, language, font_data, font_size)
        if advance > width - 2 * padding:
            raise ValueError(
                f"Rendered line exceeds available width ({advance:.1f}px > {width - 2 * padding}px)."
            )
        if align == "center":
            pen_x = (width - advance) / 2
        elif align == "right":
            pen_x = width - padding - advance
        elif align == "left":
            pen_x = float(padding)
        else:
            raise ValueError("align must be 'left', 'center', or 'right'.")

        baseline = start_baseline + line_index * line_step
        line_start = pen_x
        for glyph_id, position in zip(glyph_ids, positions):
            glyph_mask = _glyph_mask(face, glyph_id)
            bitmap_left = face.glyph.bitmap_left
            bitmap_top = face.glyph.bitmap_top
            glyph_x = round(pen_x + position.x_offset / 64 + bitmap_left)
            glyph_y = round(baseline - position.y_offset / 64 - bitmap_top)
            if glyph_mask.width and glyph_mask.height:
                mask.paste(glyph_mask, (glyph_x, glyph_y), glyph_mask)
            pen_x += position.x_advance / 64
            glyph_count += 1
        text_boxes.append(
            {
                "text": line,
                "x": max(0, round(line_start)),
                "y": max(0, round(baseline - ascender)),
                "width": round(advance),
                "height": round(ascender + descender),
            }
        )
    return mask, text_boxes, glyph_count


def render_multiline_text(
    text: str,
    language: str,
    *,
    width: int = 1200,
    height: int = 220,
    font_size: int = 64,
    foreground: Tuple[int, int, int] = (0, 0, 0),
    background: Optional[Tuple[int, int, int]] = (255, 255, 255),
    padding: int = 24,
    line_spacing: float = 1.15,
    align: str = "center",
    outline_width: int = 0,
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    shadow_offset: Tuple[int, int] = (4, 4),
    shadow_color: Tuple[int, int, int, int] = (0, 0, 0, 120),
) -> dict:
    text = str(text).strip()
    if not text:
        raise ValueError("Text must not be empty when rendering a region.")
    if language not in LANGUAGE_SHAPING:
        raise ValueError(f"Unsupported language '{language}'.")
    if width <= 0 or height <= 0 or font_size <= 0 or padding < 0:
        raise ValueError("width, height, and font_size must be positive; padding cannot be negative.")
    if line_spacing <= 0 or outline_width < 0:
        raise ValueError("line_spacing must be positive and outline_width cannot be negative.")

    font_path = resolve_font_path(language)
    _validate_font_coverage(text, font_path)
    font_family, font_style = _font_names(font_path)
    font_data = Path(font_path).read_bytes()
    max_text_width = max(width - 2 * padding, 1)
    lines = _wrap_lines(text, language, font_data, font_size, max_text_width)
    mask, text_boxes, glyph_count = _make_text_mask(
        lines,
        language,
        font_data,
        font_path,
        font_size,
        (width, height),
        padding,
        line_spacing,
        align,
    )

    canvas_color = (*background, 255) if background is not None else (0, 0, 0, 0)
    image = Image.new("RGBA", (width, height), canvas_color)
    if shadow_offset != (0, 0):
        shadow = Image.new("RGBA", (width, height), shadow_color)
        shifted_mask = ImageChops.offset(mask, shadow_offset[0], shadow_offset[1])
        image.paste(shadow, (0, 0), shifted_mask)
    if outline_width:
        expanded_mask = mask.filter(ImageFilter.MaxFilter(2 * outline_width + 1))
        outline_mask = ImageChops.subtract(expanded_mask, mask)
        outline = Image.new("RGBA", (width, height), (*outline_color, 255))
        image.paste(outline, (0, 0), outline_mask)
    foreground_layer = Image.new("RGBA", (width, height), (*foreground, 255))
    image.paste(foreground_layer, (0, 0), mask)
    ink_bbox = mask.getbbox()
    if ink_bbox is None:
        raise ValueError(f"No visible glyph pixels were rendered for {language} text: {text!r}")

    return {
        "image": image,
        "text_mask": mask,
        "text": text,
        "language": language,
        "font_path": font_path,
        "font_family": font_family,
        "font_style": font_style,
        "font_size": font_size,
        "shaping_engine": f"HarfBuzz {hb.version_string()} + FreeType",
        "glyph_count": glyph_count,
        "width": width,
        "height": height,
        "bbox": {"x": 0, "y": 0, "width": width, "height": height},
        "ink_bbox": {
            "x_min": ink_bbox[0],
            "y_min": ink_bbox[1],
            "x_max": ink_bbox[2],
            "y_max": ink_bbox[3],
        },
        "text_boxes": text_boxes,
    }


def render_language_diagnostics(output_dir: Optional[Path] = None) -> List[Path]:
    if output_dir is None:
        output_dir = DEFAULT_PREVIEW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for language, sample_text in TEST_STRINGS.items():
        result = render_multiline_text(
            sample_text,
            language,
            width=1200,
            height=250,
            font_size=52,
            foreground=(20, 36, 80),
            background=(248, 248, 244),
            outline_width=0,
            shadow_offset=(0, 0),
        )
        output_path = output_dir / f"{language}_diagnostic.png"
        result["image"].save(output_path)
        saved.append(output_path)
    return saved


if __name__ == "__main__":
    for path in render_language_diagnostics():
        print(f"Saved preview: {path}")
