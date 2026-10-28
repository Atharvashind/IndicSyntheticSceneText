from pathlib import Path

import pytest
from PIL import Image, ImageChops

from src.content_engine import generate_text, get_category_names, get_supported_languages
from src.text_renderer import (
    TEST_STRINGS,
    render_multiline_text,
    resolve_font_path,
    _shape_line,
    _validate_font_coverage,
)


def test_language_and_category_configuration():
    assert set(get_supported_languages()) == {"marathi", "hindi", "gujarati", "bengali"}
    assert set(get_category_names()) == {
        "shop_board",
        "advertising_hoarding",
        "election_banner",
        "birthday_event_banner",
    }


def test_unicode_preservation():
    sample = generate_text("hindi", "shop_board", seed=42)
    assert isinstance(sample, str)
    assert sample.strip() != ""
    assert any(ord(ch) > 127 for ch in sample)

    rendered = render_multiline_text(sample, "hindi", width=800, height=180, font_size=46)
    assert rendered["text"] == sample
    assert rendered["image"].size == (800, 180)


def test_missing_font_raises():
    with pytest.raises(FileNotFoundError):
        resolve_font_path("marathi", custom_paths=[r"C:\Windows\Fonts\definitely_missing_font_1234.ttf"])


def test_image_creation():
    result = render_multiline_text("नमस्कार", "marathi", width=600, height=200, font_size=44)
    assert result["image"].mode in {"RGBA", "RGB"}
    assert result["image"].size == (600, 200)


def test_reproducibility():
    first = generate_text("bengali", "birthday_event_banner", seed=7)
    second = generate_text("bengali", "birthday_event_banner", seed=7)
    assert first == second


@pytest.mark.parametrize("language", ["marathi", "hindi", "gujarati", "bengali"])
def test_local_font_coverage_and_shaping(language):
    font_path = resolve_font_path(language)
    sample = TEST_STRINGS[language]
    _validate_font_coverage(sample, font_path)
    glyph_ids, positions, advance = _shape_line(
        sample.splitlines()[-1], language, Path(font_path).read_bytes(), 52
    )
    assert glyph_ids
    assert all(glyph_id != 0 for glyph_id in glyph_ids)
    assert len(glyph_ids) < len(sample.splitlines()[-1])
    assert len(positions) == len(glyph_ids)
    assert advance > 0


@pytest.mark.parametrize("language", ["marathi", "hindi", "gujarati", "bengali"])
def test_diagnostic_text_renders_visible_pixels(language):
    result = render_multiline_text(
        TEST_STRINGS[language],
        language,
        width=1200,
        height=250,
        font_size=52,
        shadow_offset=(0, 0),
    )
    assert result["image"].size == (1200, 250)
    assert result["shaping_engine"].startswith("HarfBuzz")
    assert result["glyph_count"] > 0
    background = Image.new("RGB", result["image"].size, (255, 255, 255))
    assert ImageChops.difference(result["image"].convert("RGB"), background).getbbox() is not None
