import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.reference_inspired_pilot import generate_reference_inspired_pilot


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic procedural pilot using supplied photos as visual references only."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT_DIR / "data" / "generated" / "reference_inspired_seed_20261029",
    )
    parser.add_argument(
        "--preview-root",
        type=Path,
        default=ROOT_DIR / "previews" / "reference_inspired_seed_20261029",
    )
    parser.add_argument("--seed", type=int, default=20261029)
    arguments = parser.parse_args()

    report = generate_reference_inspired_pilot(
        arguments.output_root,
        arguments.preview_root,
        seed=arguments.seed,
        sample_count=24,
    )
    print(f"Generated scenes: {report['generated_scene_count']}/{report['requested_scene_count']}")
    print(f"Coverage: {json.dumps(report['coverage'], ensure_ascii=False, sort_keys=True)}")
    print(f"Automated validation passed: {report['validation_passed']}")
    print(f"Complete: {report['complete']}")
    print(f"Contact sheets, overlays, and reports: {arguments.preview_root}")
    if not report["validation_passed"] or not report["coverage_balanced"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
