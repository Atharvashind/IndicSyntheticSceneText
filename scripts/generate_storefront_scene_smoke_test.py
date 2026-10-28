import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.scene_composer import generate_storefront_scenes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a small, balanced multilingual storefront scene sample."
    )
    parser.add_argument("--count", type=int, default=12, help="Image count; must be divisible by four.")
    parser.add_argument("--seed", type=int, default=20261029, help="Reproducible generation seed.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT_DIR,
        help="Project root or alternate output root for generated artifacts.",
    )
    arguments = parser.parse_args()

    output_root = arguments.output_root.resolve()
    run_name = f"storefront_scenes_seed_{arguments.seed}"
    image_dir = output_root / "data" / "generated" / run_name
    annotation_dir = output_root / "data" / "annotations" / run_name
    contact_sheet = output_root / "previews" / f"{run_name}_contact_sheet.png"

    records = generate_storefront_scenes(
        arguments.count,
        arguments.seed,
        image_dir,
        annotation_dir,
        contact_sheet,
    )
    print(f"Generated {len(records)} storefront scenes.")
    for language in ("marathi", "hindi", "gujarati", "bengali"):
        print(f"{language}: {sum(record['language'] == language for record in records)}")
    print(f"Images and retained board layers: {image_dir}")
    print(f"Annotations: {annotation_dir}")
    print(f"Contact sheet: {contact_sheet}")
    print("All generated scenes passed image, annotation, and text-geometry validation.")


if __name__ == "__main__":
    main()
