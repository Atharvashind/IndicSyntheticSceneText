import json

import numpy as np
import pytest
from PIL import Image, ImageChops

from src.annotation_exporter import validate_visibility_metadata
from src.difficulty_dataset import generate_difficulty_variants
from src.quality_checker import validate_dataset
from src.scene_composer import generate_storefront_scenes
from src.scene_transformer import (
    apply_blur_degradation,
    apply_distance_scale,
    apply_lighting,
    apply_occlusion,
    apply_perspective,
    apply_profile,
    make_seeded_perspective_homography,
)


def _sample():
    image = Image.new("RGB", (200, 160), (220, 210, 190))
    annotation = {
        "sample_id": "sample-001",
        "image": "sample-001.png",
        "image_width": 200,
        "image_height": 160,
        "random_seed": 1,
        "text_regions": [
            {
                "region_id": "name",
                "transcription": "text",
                "language": "hindi",
                "script": "Devanagari",
                "reading_order": 0,
                "bbox": {"x_min": 70, "y_min": 60, "x_max": 130, "y_max": 90},
                "polygon": [[70, 60], [130, 60], [130, 90], [70, 90]],
                "board_bbox": {"x_min": 20, "y_min": 20, "x_max": 80, "y_max": 50},
                "board_polygon": [[20, 20], [80, 20], [80, 50], [20, 50]],
                "font": {"family": "test"},
            }
        ],
    }
    return image, annotation


def test_distance_scale_updates_dimensions_and_all_coordinates():
    image, annotation = _sample()
    scaled, result = apply_distance_scale(image, annotation, 0.5, seed=2)
    assert scaled.size == (100, 80)
    assert (result["image_width"], result["image_height"]) == scaled.size
    assert result["text_regions"][0]["polygon"] == [
        [35.0, 30.0],
        [65.0, 30.0],
        [65.0, 45.0],
        [35.0, 45.0],
    ]
    assert result["text_regions"][0]["bbox"] == {
        "x_min": 35.0,
        "y_min": 30.0,
        "x_max": 65.0,
        "y_max": 45.0,
    }
    assert annotation["image_width"] == 200


def test_distance_scale_rejects_degenerate_output_geometry():
    image, annotation = _sample()
    with pytest.raises(ValueError, match="degenerate"):
        apply_distance_scale(image, annotation, 0.001)


def test_perspective_homography_maps_polygon_and_is_deterministic():
    image, annotation = _sample()
    image.putpixel((99, 79), (255, 0, 0))
    homography = np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]], dtype=float)
    warped, result = apply_perspective(image, annotation, homography, seed=3)
    assert warped.size == image.size
    assert warped.getpixel((109, 84)) == (255, 0, 0)
    assert result["text_regions"][0]["polygon"] == [
        [80.0, 65.0],
        [140.0, 65.0],
        [140.0, 95.0],
        [80.0, 95.0],
    ]
    assert result["text_regions"][0]["bbox"] == {
        "x_min": 80.0,
        "y_min": 65.0,
        "x_max": 140.0,
        "y_max": 95.0,
    }
    matrix_a, source_a, destination_a = make_seeded_perspective_homography(200, 160, 0.04, 81)
    matrix_b, source_b, destination_b = make_seeded_perspective_homography(200, 160, 0.04, 81)
    assert np.array_equal(matrix_a, matrix_b)
    assert source_a == source_b
    assert destination_a == destination_b


def test_perspective_rejects_singular_and_out_of_frame_geometry():
    image, annotation = _sample()
    with pytest.raises(ValueError, match="singular"):
        apply_perspective(image, annotation, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="outside"):
        apply_perspective(
            image,
            annotation,
            np.array([[1, 0, 100], [0, 1, 0], [0, 0, 1]], dtype=float),
        )


def test_lighting_and_blur_keep_geometry_unchanged():
    image, annotation = _sample()
    lit, lit_annotation = apply_lighting(
        image, annotation, 0.8, 1.1, shadow_opacity=0.25, shadow_sigma=4, seed=4
    )
    blurred, blur_annotation = apply_blur_degradation(
        image, annotation, 1.2, downsample_scale=0.6, seed=5
    )
    assert lit.size == blurred.size == image.size
    assert lit_annotation["text_regions"][0]["polygon"] == annotation["text_regions"][0]["polygon"]
    assert blur_annotation["text_regions"][0]["bbox"] == annotation["text_regions"][0]["bbox"]
    assert lit_annotation["scene_difficulty"]["transformations"][0]["geometry_changed"] is False
    assert blur_annotation["scene_difficulty"]["transformations"][0]["geometry_changed"] is False


def test_occlusion_keeps_text_annotation_and_records_visibility_estimate():
    image, annotation = _sample()
    first_image, first = apply_occlusion(
        image, annotation, seed=72, width_fraction=0.15, height_fraction=0.12
    )
    second_image, _ = apply_occlusion(
        image, annotation, seed=72, width_fraction=0.15, height_fraction=0.12
    )
    assert ImageChops.difference(first_image, second_image).getbbox() is None
    assert len(first["text_regions"]) == 1
    visibility = first["text_regions"][0]["visibility"]
    assert 0.0 <= visibility["estimated_fraction"] <= 1.0
    assert visibility["status"] in {"visible", "partially_occluded", "fully_occluded"}
    assert visibility["policy"] == "bbox_unoccluded_area_fraction"
    assert validate_visibility_metadata(first) == []
    assert first["scene_difficulty"]["transformations"][0]["occlusion_region"]


def test_occlusion_policy_classifies_full_partial_and_clear_boxes():
    image, annotation = _sample()
    _, full = apply_occlusion(
        image, annotation, seed=1, width_fraction=0.8, height_fraction=0.8
    )
    assert len(full["text_regions"]) == 1
    assert full["text_regions"][0]["visibility"]["status"] == "fully_occluded"

    _, clear = apply_occlusion(
        image, annotation, seed=1, width_fraction=0.01, height_fraction=0.01
    )
    assert clear["text_regions"][0]["visibility"]["status"] == "visible"


def test_clean_profile_is_pixel_identity_and_keeps_source_geometry():
    image, annotation = _sample()
    clean, clean_annotation = apply_profile(image, annotation, "D0_clean", 2)
    assert ImageChops.difference(image, clean).getbbox() is None
    assert clean_annotation["text_regions"][0]["bbox"] == annotation["text_regions"][0]["bbox"]
    assert clean_annotation["scene_difficulty"]["transformations"] == [
        {"type": "identity", "seed": 2, "geometry_changed": False}
    ]


@pytest.mark.parametrize(
    "profile",
    ["D2_distance", "D2_perspective", "D2_lighting", "D2_blur", "D2_occlusion"],
)
def test_profile_outputs_are_reproducible(profile):
    image, annotation = _sample()
    first_image, first_annotation = apply_profile(image, annotation, profile, 940)
    second_image, second_annotation = apply_profile(image, annotation, profile, 940)
    assert ImageChops.difference(first_image, second_image).getbbox() is None
    assert first_annotation == second_annotation


def test_transformed_dataset_passes_image_annotation_and_parent_validation(tmp_path):
    source_images = tmp_path / "source" / "images"
    source_annotations = tmp_path / "source" / "annotations"
    generate_storefront_scenes(
        count=4,
        seed=8311,
        image_dir=source_images,
        annotation_dir=source_annotations,
    )
    output_root = tmp_path / "variants"
    generate_difficulty_variants(
        source_images,
        source_annotations,
        output_root,
        tmp_path / "previews",
        seed=1209,
        profiles=("D0_clean", "D2_distance", "D2_perspective", "D2_lighting", "D2_blur", "D2_occlusion"),
    )
    for profile in (
        "D0_clean",
        "D2_distance",
        "D2_perspective",
        "D2_lighting",
        "D2_blur",
        "D2_occlusion",
    ):
        report = validate_dataset(
            output_root / profile / "images",
            output_root / profile / "annotations",
        )
        assert report["passed"] is True, report["errors"]
        assert report["image_count"] == report["annotation_count"] == 4
    perspective_annotations = list(
        (output_root / "D2_perspective" / "annotations").glob("*.json")
    )
    tampered_path = perspective_annotations[0]
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
    tampered["text_regions"][0]["bbox"]["x_min"] += 1
    tampered["text_regions"][0]["polygon"][0][0] += 1
    tampered["text_regions"][0]["polygon"][3][0] += 1
    tampered_path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
    invalid_report = validate_dataset(
        output_root / "D2_perspective" / "images",
        output_root / "D2_perspective" / "annotations",
    )
    assert invalid_report["passed"] is False
    assert any("replayed transformation" in entry["error"] for entry in invalid_report["errors"])

    clean_annotations = list((output_root / "D0_clean" / "annotations").glob("*.json"))
    for annotation_path in clean_annotations:
        record = json.loads(annotation_path.read_text(encoding="utf-8"))
        output_path = output_root / "D0_clean" / "images" / record["image"]
        parent_path = output_path.parent / record["parent"]["image"]
        assert output_path.read_bytes() == parent_path.read_bytes()
