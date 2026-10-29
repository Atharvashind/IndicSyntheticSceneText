import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.background_pilot import generate_background_pilot
from src.content_engine import load_config


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Generate a small background-composition pilot from verified local assets."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT_DIR / config["background_pipeline"]["manifest_path"],
        help="Background asset manifest; paths in it are project-relative.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT_DIR / "data" / "generated" / "photorealistic_pilot_seed_20261029",
    )
    parser.add_argument(
        "--preview-root",
        type=Path,
        default=ROOT_DIR / "previews" / "photorealistic_pilot_seed_20261029",
    )
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=config["background_pipeline"]["pilot_scene_count"],
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless the full 24-scene coverage target is met.",
    )
    arguments = parser.parse_args()

    report = generate_background_pilot(
        arguments.manifest,
        arguments.output_root,
        arguments.preview_root,
        arguments.seed,
        maximum_scenes=arguments.max_scenes,
    )
    print(
        f"Eligible backgrounds: {report['eligible_background_count']}; "
        f"generated scenes: {report['generated_scene_count']}/{report['target_scene_count']}"
    )
    print(f"Coverage: {json.dumps(report['coverage'], ensure_ascii=False, sort_keys=True)}")
    if report["validation_applicable"]:
        print(f"Automated validation passed: {report['validation_passed']}")
    else:
        print("Automated validation: not applicable (no eligible samples were generated).")
    print(f"24-scene coverage complete: {report['complete']}")
    print(f"Reports and visual previews: {arguments.preview_root}")
    if report["validation_applicable"] and not report["validation_passed"]:
        return 1
    if arguments.require_complete and not report["complete"]:
        print("The 24-scene target is incomplete; see the report for excluded assets and next steps.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
