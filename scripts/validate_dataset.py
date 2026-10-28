import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.quality_checker import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate image/annotation files and rendered text geometry.")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing PNG images.")
    parser.add_argument("--annotations", type=Path, required=True, help="Directory containing per-image JSON.")
    parser.add_argument("--report", type=Path, help="Optional machine-readable JSON report output.")
    arguments = parser.parse_args()

    report = validate_dataset(arguments.images, arguments.annotations)
    print(
        f"Images: {report['image_count']}; annotations: {report['annotation_count']}; "
        f"unique IDs: {report['sample_id_count']}"
    )
    print(f"Languages: {report['language_counts']}")
    print(f"Categories: {report['category_counts']}")
    if report["passed"]:
        print("Validation passed.")
    else:
        print(f"Validation failed with {len(report['errors'])} issue(s):")
        for entry in report["errors"]:
            print(f"- {entry['file']}: {entry['error']}")
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Machine-readable report: {arguments.report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
