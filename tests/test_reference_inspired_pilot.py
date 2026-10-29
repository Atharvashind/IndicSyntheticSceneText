import json
from collections import Counter

from PIL import Image

from src.content_engine import SUPPORTED_LANGUAGES
from src.font_registry import validate_font_registry
from src.quality_checker import validate_dataset
from src.reference_inspired_pilot import generate_reference_inspired_pilot
from src.scene_transformer import PROFILE_NAMES
from src.signboard_generator import render_shop_board_sample


def _run(root, output_name):
    output_root = root / output_name / "data"
    preview_root = root / output_name / "previews"
    report = generate_reference_inspired_pilot(
        output_root,
        preview_root,
        seed=20261029,
        sample_count=12,
    )
    return report, output_root, preview_root


def test_reference_inspired_pilot_is_balanced_and_validated(tmp_path):
    report, output_root, preview_root = _run(tmp_path, "pilot")
    assert report["photorealistic"] is False
    assert report["complete"] is True
    assert report["generated_scene_count"] == 12
    assert report["coverage"]["language_counts"] == {
        language: 3 for language in SUPPORTED_LANGUAGES
    }
    assert report["coverage"]["scene_category_counts"] == {
        "outdoor_hoarding": 6,
        "shopfront": 6,
    }
    assert report["coverage"]["difficulty_profile_counts"] == {
        profile: 2 for profile in PROFILE_NAMES
    }
    assert all(value["passed"] for value in report["profile_validation"].values())
    manifest = json.loads((preview_root / "pilot_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["scenes"]) == 12
    assert manifest["reference_usage"]["reference_pixels_used"] is False

    sample_ids = set()
    for profile in PROFILE_NAMES:
        image_dir = output_root / profile / "images"
        annotation_dir = output_root / profile / "annotations"
        validation = validate_dataset(image_dir, annotation_dir)
        assert validation["passed"] is True, validation["errors"]
        assert validation["image_count"] == validation["annotation_count"] == 2
        assert (preview_root / f"{profile}_contact_sheet.png").is_file()
        for path in annotation_dir.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            assert record["sample_id"] not in sample_ids
            sample_ids.add(record["sample_id"])
            assert record["reference_usage"]["mode"] == "visual_reference_only"
            assert record["reference_usage"]["reference_pixels_used"] is False
            assert record["validation"]["status"] == "passed"
            assert record["scene_difficulty"]["profile"] == profile
            assert record["text_regions"]
            assert all(region["transcription"] for region in record["text_regions"])
            with Image.open(image_dir / record["image"]) as image:
                assert image.size == (record["image_width"], record["image_height"])
        assert len(list((preview_root / "overlays" / profile).glob("*_overlay.png"))) == 2


def test_reference_inspired_pilot_reproduces_images_and_annotations(tmp_path):
    first_report, first_root, _ = _run(tmp_path, "first")
    second_report, second_root, _ = _run(tmp_path, "second")
    assert first_report["coverage"] == second_report["coverage"]
    first_manifest = json.loads(
        (tmp_path / "first" / "previews" / "pilot_manifest.json").read_text(encoding="utf-8")
    )
    second_manifest = json.loads(
        (tmp_path / "second" / "previews" / "pilot_manifest.json").read_text(encoding="utf-8")
    )
    assert first_manifest == second_manifest
    for profile in PROFILE_NAMES:
        first_images = first_root / profile / "images"
        second_images = second_root / profile / "images"
        first_annotations = first_root / profile / "annotations"
        second_annotations = second_root / profile / "annotations"
        for image_path in first_images.glob("*.png"):
            assert image_path.read_bytes() == (second_images / image_path.name).read_bytes()
        for annotation_path in first_annotations.glob("*.json"):
            assert annotation_path.read_bytes() == (
                second_annotations / annotation_path.name
            ).read_bytes()


def test_pilot_schedule_keeps_each_language_and_category_balanced():
    from src.reference_inspired_pilot import _make_schedule

    schedule = _make_schedule(24, 9)
    assert Counter(language for language, _, _ in schedule) == {
        language: 6 for language in SUPPORTED_LANGUAGES
    }
    assert Counter(category for _, category, _ in schedule) == {
        "shopfront": 12,
        "outdoor_hoarding": 12,
    }
    assert Counter(profile for _, _, profile in schedule) == {
        profile: 4 for profile in PROFILE_NAMES
    }
    for profile in PROFILE_NAMES:
        profile_schedule = [item for item in schedule if item[2] == profile]
        assert Counter(language for language, _, _ in profile_schedule) == {
            language: 1 for language in SUPPORTED_LANGUAGES
        }
        assert Counter(category for _, category, _ in profile_schedule) == {
            category: 2 for category in ("shopfront", "outdoor_hoarding")
        }


def test_multiline_hierarchy_fits_two_line_service_text():
    board, annotation = render_shop_board_sample(
        "hindi",
        13917917427397914965,
        "multiline-regression",
        "multiline_hierarchy",
    )
    assert board.size == (1100, 560)
    assert len(annotation["text_regions"]) == 2
    for region in annotation["text_regions"]:
        box = region["bbox"]
        assert 0 <= box["x_min"] < box["x_max"] <= board.width
        assert 0 <= box["y_min"] < box["y_max"] <= board.height


def test_validated_font_choices_are_recorded_in_generated_annotations(tmp_path):
    font_report = validate_font_registry(tmp_path / "font-validation")
    output_root = tmp_path / "font-variety" / "data"
    preview_root = tmp_path / "font-variety" / "previews"
    report = generate_reference_inspired_pilot(
        output_root,
        preview_root,
        seed=1919,
        sample_count=12,
        font_validation_report_path=tmp_path / "font-validation" / "font_validation_report.json",
        sample_id_prefix="font-test",
    )
    assert report["complete"] is True
    assert report["coverage"]["background_source_counts"] == {
        "procedural_illustration": 12
    }
    assert set(report["coverage"]["font_counts"]) <= set(font_report["eligible_font_ids"])
    for profile in PROFILE_NAMES:
        image_dir = output_root / profile / "images"
        annotation_dir = output_root / profile / "annotations"
        validation = validate_dataset(image_dir, annotation_dir)
        assert validation["passed"] is True, validation["errors"]
        for path in annotation_dir.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            assert record["background_source_type"] == "procedural_illustration"
            for region in record["text_regions"]:
                assert region["font"]["id"] == record["generation_parameters"]["font_id"]
                assert region["font"]["weight"] in {400, 700}
                assert region["font"]["style"] in {"Regular", "Bold"}
    for language in SUPPORTED_LANGUAGES:
        assert (preview_root / "by_language" / f"{language}.png").is_file()
    for font_id in report["coverage"]["font_counts"]:
        assert (preview_root / "by_font" / f"{font_id}.png").is_file()
