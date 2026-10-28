import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.text_renderer import render_language_diagnostics


if __name__ == "__main__":
    output_dir = ROOT_DIR / "previews"
    saved = render_language_diagnostics(output_dir)
    print(f"Saved {len(saved)} diagnostic previews to {output_dir}")
    for path in saved:
        print(path)
