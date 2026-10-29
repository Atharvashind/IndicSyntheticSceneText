import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.font_registry import validate_font_registry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate configured local font files and produce a rendering contact sheet."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "previews" / "font_validation_seed_20261029",
    )
    arguments = parser.parse_args()
    report = validate_font_registry(arguments.output_dir)
    print(f"Candidates: {report['font_candidate_count']}")
    print(f"Generation-eligible fonts: {report['eligible_font_ids']}")
    print(f"Excluded fonts: {report['failed_or_unlicensed_font_ids']}")
    print(f"Contact sheet: {arguments.output_dir / report['diagnostic_contact_sheet']}")
    print(f"Report: {arguments.output_dir / 'font_validation_report.json'}")
    return 0 if report["eligible_font_ids"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
