import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.signboard_generator import generate_shop_boards


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small, balanced shop-board sample.")
    parser.add_argument("--count", type=int, default=8, help="Image count; must be divisible by four.")
    parser.add_argument("--seed", type=int, default=20261028, help="Reproducible generation seed.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT_DIR,
        help="Project root or alternate output root for generated artifacts.",
    )
    arguments = parser.parse_args()

    run_name = f"shop_boards_seed_{arguments.seed}"
    output_root = arguments.output_root.resolve()
    image_dir = output_root / "data" / "generated" / run_name
    annotation_dir = output_root / "data" / "annotations" / run_name
    contact_sheet_path = output_root / "previews" / f"{run_name}_contact_sheet.png"

    records = generate_shop_boards(
        count=arguments.count,
        seed=arguments.seed,
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        contact_sheet_path=contact_sheet_path,
    )
    print(f"Generated {len(records)} shop-board images.")
    for language in ("marathi", "hindi", "gujarati", "bengali"):
        language_count = sum(record["language"] == language for record in records)
        print(f"{language}: {language_count}")
    print(f"Images: {image_dir}")
    print(f"Annotations: {annotation_dir}")
    print(f"Contact sheet: {contact_sheet_path}")
    print("All generated records passed image and annotation validation.")


if __name__ == "__main__":
    main()
