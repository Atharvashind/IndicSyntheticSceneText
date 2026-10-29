import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.content_engine import load_config
from src.reference_inspired_pilot import generate_reference_inspired_pilot


def main() -> int:
    defaults = load_config()["batch_generation"]
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic multilingual scene-text batch using validated local fonts."
        )
    )
    parser.add_argument("--sample-count", type=int, default=defaults["default_sample_count"])
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT_DIR / "data" / "generated" / "multilingual_variety_seed_20261030",
    )
    parser.add_argument(
        "--preview-root",
        type=Path,
        default=ROOT_DIR / "previews" / "multilingual_variety_seed_20261030",
    )
    parser.add_argument(
        "--font-report",
        type=Path,
        default=ROOT_DIR / "previews" / "font_validation_seed_20261029" / "font_validation_report.json",
    )
    arguments = parser.parse_args()

    report = generate_reference_inspired_pilot(
        arguments.output_root,
        arguments.preview_root,
        seed=arguments.seed,
        sample_count=arguments.sample_count,
        font_validation_report_path=arguments.font_report,
        sample_id_prefix="indic-multilingual",
    )
    print(
        f"Generated: {report['generated_scene_count']}/{report['requested_scene_count']}; "
        f"complete={report['complete']}"
    )
    print(f"Coverage: {json.dumps(report['coverage'], ensure_ascii=False, sort_keys=True)}")
    print(f"Background sources: {json.dumps(report['background_assets']['selected_source_counts'], sort_keys=True)}")
    print(f"Reports and grouped contact sheets: {arguments.preview_root}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
