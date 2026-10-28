import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.difficulty_dataset import generate_difficulty_variants
from src.quality_checker import validate_dataset
from src.scene_transformer import PROFILE_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate separate, individually controlled scene difficulty variants."
    )
    parser.add_argument(
        "--source-images",
        type=Path,
        default=ROOT_DIR / "data" / "generated" / "storefront_scenes_seed_20261029",
    )
    parser.add_argument(
        "--source-annotations",
        type=Path,
        default=ROOT_DIR / "data" / "annotations" / "storefront_scenes_seed_20261029",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT_DIR / "data" / "generated" / "scene_difficulty_seed_20261029",
    )
    parser.add_argument(
        "--preview-root",
        type=Path,
        default=ROOT_DIR / "previews" / "scene_difficulty_seed_20261029",
    )
    parser.add_argument("--seed", type=int, default=20261029)
    arguments = parser.parse_args()

    report = generate_difficulty_variants(
        arguments.source_images,
        arguments.source_annotations,
        arguments.output_root,
        arguments.preview_root,
        arguments.seed,
        PROFILE_NAMES,
    )
    validation_reports = {}
    for profile in PROFILE_NAMES:
        profile_report_path = arguments.preview_root / f"{profile}_validation.json"
        result = validate_dataset(
            arguments.output_root / profile / "images",
            arguments.output_root / profile / "annotations",
        )
        profile_report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        validation_reports[profile] = result
        print(
            f"{profile}: images={result['image_count']} annotations={result['annotation_count']} "
            f"passed={result['passed']}"
        )
        if not result["passed"]:
            for entry in result["errors"]:
                print(f"- {entry['file']}: {entry['error']}")

    report["validation_reports"] = {
        profile: {
            "passed": value["passed"],
            "image_count": value["image_count"],
            "annotation_count": value["annotation_count"],
            "errors": value["errors"],
        }
        for profile, value in validation_reports.items()
    }
    report["passed"] = all(value["passed"] for value in validation_reports.values())
    report_path = arguments.preview_root / "scene_difficulty_generation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Contact sheets, overlays, and reports: {arguments.preview_root}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
