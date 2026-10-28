import json
from collections import Counter

import pytest
from PIL import Image

from src.annotation_exporter import validate_annotation
from src.content_engine import SUPPORTED_LANGUAGES
from src.annotation_exporter import validate_rendered_geometry
from src.signboard_generator import generate_shop_boards
from src.text_renderer import render_multiline_text


def _generate_pair(root, run_name="run"):
    return generate_shop_boards(
        count=8,
        seed=4711,
        image_dir=root / run_name / "images",
        annotation_dir=root / run_name / "annotations",
        contact_sheet_path=root / run_name / "contact_sheet.png",
    )


def test_tiny_dataset_has_balanced_languages_and_valid_annotations(tmp_path):
    records = _generate_pair(tmp_path)
    assert len(records) == 8
    assert Counter(record["language"] for record in records) == {
        language: 2 for language in SUPPORTED_LANGUAGES
    }

    for record in records:
        image_path = tmp_path / "run" / "images" / record["image"]
        annotation_path = tmp_path / "run" / "annotations" / f"{record['sample_id']}.json"
        with Image.open(image_path) as image:
            assert image.format == "PNG"
            assert image.size == (record["image_width"], record["image_height"])
        stored = json.loads(annotation_path.read_text(encoding="utf-8"))
        assert stored["validation"]["status"] == "passed"
        assert stored["validation"]["visual_review_required"] is True
        assert validate_annotation(stored, image_path) == []

        assert len(stored["text_regions"]) >= 2
        layout_by_id = {
            entry["region_id"]: entry
            for entry in stored["generation_parameters"]["region_layout"]
        }
        for region in stored["text_regions"]:
            assert region["transcription"]
            box = region["bbox"]
            assert box["x_max"] > box["x_min"]
            assert box["y_max"] > box["y_min"]
            assert 0 <= box["x_min"] < box["x_max"] <= image.width
            assert 0 <= box["y_min"] < box["y_max"] <= image.height
            assert region["polygon"][0] == [box["x_min"], box["y_min"]]
            layout = layout_by_id[region["region_id"]]
            rendered = render_multiline_text(
                region["transcription"],
                record["language"],
                width=layout["canvas_width"],
                height=layout["canvas_height"],
                font_size=layout["font_size_px"],
                foreground=tuple(
                    stored["generation_parameters"]["text_rgb"]
                ),
                background=None,
                padding=layout["padding_px"],
                line_spacing=layout["line_spacing"],
                align=layout["alignment"],
                shadow_offset=(0, 0),
            )
            ink = rendered["ink_bbox"]
            canvas_x, canvas_y = layout["canvas_x"], layout["canvas_y"]
            assert box == {
                "x_min": canvas_x + ink["x_min"],
                "y_min": canvas_y + ink["y_min"],
                "x_max": canvas_x + ink["x_max"],
                "y_max": canvas_y + ink["y_max"],
            }
        assert validate_rendered_geometry(stored) == []
    assert {
        record["generation_parameters"]["template_id"] for record in records
    } == {
        "centered_business_subtitle",
        "left_name_services",
        "headline_contact_details",
        "multiline_hierarchy",
        "logo_business_sign",
    }


def test_same_seed_reproduces_images_and_annotations(tmp_path):
    first = _generate_pair(tmp_path, "first")
    second = _generate_pair(tmp_path, "second")
    assert first == second
    for record in first:
        first_image = tmp_path / "first" / "images" / record["image"]
        second_image = tmp_path / "second" / "images" / record["image"]
        assert first_image.read_bytes() == second_image.read_bytes()
        first_annotation = tmp_path / "first" / "annotations" / f"{record['sample_id']}.json"
        second_annotation = tmp_path / "second" / "annotations" / f"{record['sample_id']}.json"
        assert first_annotation.read_bytes() == second_annotation.read_bytes()
    assert (tmp_path / "first" / "contact_sheet.png").is_file()


def test_existing_output_files_are_not_overwritten(tmp_path):
    _generate_pair(tmp_path)
    with pytest.raises(FileExistsError):
        _generate_pair(tmp_path)


def test_invalid_annotation_bbox_is_rejected(tmp_path):
    records = _generate_pair(tmp_path)
    record = records[0]
    image_path = tmp_path / "run" / "images" / record["image"]
    record["text_regions"][0]["bbox"]["x_min"] = -1
    errors = validate_annotation(record, image_path)
    assert any("outside image bounds" in error for error in errors)
