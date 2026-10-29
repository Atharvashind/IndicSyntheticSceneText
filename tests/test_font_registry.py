import json

import pytest

from src.font_registry import validate_font_registry
from src.reference_inspired_pilot import _assign_fonts, _make_schedule
from src.scene_transformer import PROFILE_NAMES
from src.content_engine import SUPPORTED_LANGUAGES
from src.text_renderer import render_multiline_text


def test_registry_validates_supported_fonts_and_excludes_unverified_candidates(tmp_path):
    output_dir = tmp_path / "font-check"
    report = validate_font_registry(output_dir)
    assert report["font_candidate_count"] == 9
    assert report["passed_font_count"] == 6
    assert set(report["eligible_font_ids"]) == {
        "noto-devanagari-regular",
        "noto-devanagari-bold",
        "noto-gujarati-regular",
        "noto-gujarati-bold",
        "noto-bengali-regular",
        "noto-bengali-bold",
    }
    assert set(report["failed_or_unlicensed_font_ids"]) == {
        "local-annapurna-bold-unverified",
        "local-chandra-head-unverified",
        "local-kirati-unverified",
    }
    for candidate in report["fonts"]:
        if candidate["eligible_for_generation"]:
            assert candidate["license_status"] == "verified"
            assert candidate["family"]
            assert candidate["weight"] in {400, 700}
            assert candidate["style"] in {"Regular", "Bold"}
            assert candidate["supported_scripts"]
            assert all(language["passed"] for language in candidate["language_validation"])
            assert all(
                language["exact_transcription_preserved"]
                and language["clipping_check_passed"]
                and not language["missing_glyph_box_detected"]
                and all(language["feature_probes"].values())
                for language in candidate["language_validation"]
            )
    stored = json.loads((output_dir / "font_validation_report.json").read_text(encoding="utf-8"))
    assert stored["eligible_font_ids"] == report["eligible_font_ids"]
    assert (output_dir / report["diagnostic_contact_sheet"]).is_file()
    with pytest.raises(FileExistsError):
        validate_font_registry(output_dir)


def test_explicit_font_render_preserves_exact_unicode_and_reports_weight():
    text = "  नमस्ते, भारत!\nक्षेत्र और प्रज्ञा  "
    result = render_multiline_text(
        text,
        "hindi",
        width=1000,
        height=300,
        font_size=48,
        font_path="assets/fonts/devanagari/NotoSansDevanagari-Bold.ttf",
        shadow_offset=(0, 0),
    )
    assert result["text"] == text
    assert result["font_family"] == "Noto Sans Devanagari"
    assert result["font_style"] == "Bold"
    assert result["font_weight"] == 700
    assert 0 <= result["ink_bbox"]["x_min"] < result["ink_bbox"]["x_max"] <= 1000
    assert 0 <= result["ink_bbox"]["y_min"] < result["ink_bbox"]["y_max"] <= 300


def test_large_schedule_and_font_assignment_are_balanced_and_repeatable():
    schedule = _make_schedule(1000, 731)
    assert schedule == _make_schedule(1000, 731)
    assert len(schedule) == 1000

    fonts_by_language = {
        "marathi": [
            {"id": "deva-regular", "path": "regular.ttf", "weight": 400, "style": "Regular"},
            {"id": "deva-bold", "path": "bold.ttf", "weight": 700, "style": "Bold"},
        ],
        "hindi": [
            {"id": "deva-regular", "path": "regular.ttf", "weight": 400, "style": "Regular"},
            {"id": "deva-bold", "path": "bold.ttf", "weight": 700, "style": "Bold"},
        ],
        "gujarati": [
            {"id": "guj-regular", "path": "regular.ttf", "weight": 400, "style": "Regular"},
            {"id": "guj-bold", "path": "bold.ttf", "weight": 700, "style": "Bold"},
        ],
        "bengali": [
            {"id": "ben-regular", "path": "regular.ttf", "weight": 400, "style": "Regular"},
            {"id": "ben-bold", "path": "bold.ttf", "weight": 700, "style": "Bold"},
        ],
    }
    assignments = _assign_fonts(schedule, fonts_by_language, 731)
    assert assignments == _assign_fonts(schedule, fonts_by_language, 731)
    assert {language: sum(item[0] == language for item in schedule) for language in SUPPORTED_LANGUAGES} == {
        language: 250 for language in SUPPORTED_LANGUAGES
    }
    assert {category: sum(item[1] == category for item in schedule) for category in ("shopfront", "outdoor_hoarding")} == {
        "shopfront": 500,
        "outdoor_hoarding": 500,
    }
    profile_counts = {profile: sum(item[2] == profile for item in schedule) for profile in PROFILE_NAMES}
    assert sorted(profile_counts.values()) == [166, 166, 167, 167, 167, 167]
    for language in SUPPORTED_LANGUAGES:
        counts = {}
        for item, font in zip(schedule, assignments):
            if item[0] == language:
                counts[font["font_id"]] = counts.get(font["font_id"], 0) + 1
        assert max(counts.values()) - min(counts.values()) <= 1
