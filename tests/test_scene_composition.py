import json
from collections import Counter

from PIL import Image, ImageChops

from src.annotation_exporter import validate_annotation, validate_rendered_geometry
from src.content_engine import SUPPORTED_LANGUAGES
from src.quality_checker import validate_dataset
from src.scene_composer import generate_storefront_scenes


def _generate_scenes(root, name="run", count=8):
    return generate_storefront_scenes(
        count=count,
        seed=83017,
        image_dir=root / name / "images",
        annotation_dir=root / name / "annotations",
        contact_sheet_path=root / name / "contact_sheet.png",
    )


def test_storefront_scenes_keep_board_layer_and_translate_text_geometry(tmp_path):
    records = _generate_scenes(tmp_path)
    assert len(records) == 8
    assert Counter(record["language"] for record in records) == {
        language: 2 for language in SUPPORTED_LANGUAGES
    }
    assert {
        record["generation_parameters"]["template_id"] for record in records
    } == {
        "centered_business_subtitle",
        "left_name_services",
        "headline_contact_details",
        "multiline_hierarchy",
        "logo_business_sign",
    }

    image_dir = tmp_path / "run" / "images"
    annotation_dir = tmp_path / "run" / "annotations"
    for record in records:
        scene_path = image_dir / record["image"]
        board_path = image_dir / record["board_layer"]
        annotation_path = annotation_dir / f"{record['sample_id']}.json"
        stored = json.loads(annotation_path.read_text(encoding="utf-8"))
        assert validate_annotation(stored, scene_path) == []
        assert validate_rendered_geometry(stored) == []
        assert stored["scene_transform"]["scale"] == 1
        assert stored["scene_transform"]["rotation_degrees"] == 0
        dx = stored["scene_transform"]["translate_x"]
        dy = stored["scene_transform"]["translate_y"]
        with Image.open(scene_path) as scene, Image.open(board_path) as board:
            assert scene.size == (stored["image_width"], stored["image_height"])
            assert dx >= 0 and dy >= 0
            assert dx + board.width <= scene.width
            assert dy + board.height <= scene.height
            crop = scene.crop((dx, dy, dx + board.width, dy + board.height))
            assert ImageChops.difference(crop.convert("RGBA"), board.convert("RGBA")).getbbox() is None

        for region in stored["text_regions"]:
            source = region["board_bbox"]
            box = region["bbox"]
            assert box == {
                "x_min": source["x_min"] + dx,
                "y_min": source["y_min"] + dy,
                "x_max": source["x_max"] + dx,
                "y_max": source["y_max"] + dy,
            }
    assert (tmp_path / "run" / "contact_sheet.png").is_file()


def test_storefront_generation_is_reproducible(tmp_path):
    first = _generate_scenes(tmp_path, "first", count=4)
    second = _generate_scenes(tmp_path, "second", count=4)
    assert first == second
    for record in first:
        for filename in (record["image"], record["board_layer"]):
            assert (
                (tmp_path / "first" / "images" / filename).read_bytes()
                == (tmp_path / "second" / "images" / filename).read_bytes()
            )
        name = f"{record['sample_id']}.json"
        assert (
            (tmp_path / "first" / "annotations" / name).read_bytes()
            == (tmp_path / "second" / "annotations" / name).read_bytes()
        )


def test_dataset_validator_checks_scene_samples(tmp_path):
    _generate_scenes(tmp_path)
    report = validate_dataset(
        tmp_path / "run" / "images",
        tmp_path / "run" / "annotations",
    )
    assert report["passed"] is True, report["errors"]
    assert report["image_count"] == 8
    assert report["annotation_count"] == 8
