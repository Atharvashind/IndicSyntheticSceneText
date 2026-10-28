# IndicSyntheticSceneText

IndicSyntheticSceneText is a Python research project for generating synthetic multilingual scene-text samples for four Indian languages: Marathi, Hindi, Gujarati and Bengali. The initial focus is on storefront signage, hoardings, election banners and event banners with realistic typography and scene degradation effects.

## Research objective

The project creates controlled synthetic scene-text images that support OCR research on decorative Indian-language signage. It is intentionally designed to remain local, reproducible and CPU-friendly, without relying on cloud APIs or paid services.

This is a synthetic-data generation framework, not an OCR training pipeline. Synthetic data is useful for controlled investigations, but it is not automatically equivalent to real-world scene text.

## Architecture

- `src/content_engine.py`: multilingual content generation and template selection
- `src/text_renderer.py`: script-aware rendering for Indic text and test previews
- `src/signboard_generator.py`: board layouts and region composition
- `src/annotation_exporter.py`: per-image JSON annotations and geometry validation
- `src/scene_composer.py`: procedural storefront scenes with retained board layers
- `src/quality_checker.py`: dataset pairing, annotation, and rendered-geometry validation
- `src/scene_transformer.py`: planned perspective and scene degradation pipeline
- `src/dataset_builder.py`: planned larger dataset generation coordination
- `tests/`: lightweight validation of render and dataset logic
- `scripts/`: local utilities for running tests and generating data

## Supported languages

- Marathi (Devanagari)
- Hindi (Devanagari)
- Gujarati
- Bengali

## Supported image categories

- `shop_board`
- `advertising_hoarding`
- `election_banner`
- `birthday_event_banner`

## Installation

1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
3. Add local fonts under the project font folders:
   - `assets/fonts/devanagari/`
   - `assets/fonts/gujarati/`
   - `assets/fonts/bengali/`
4. Update `configs/generation_config.json` if you want to point at different local font files.

## Font installation and configuration

Do not automatically download fonts. This project expects local font files from your own machine or licensed sources. The project checks these directories and raises a helpful error when no compatible font is found.

Recommended minimal setup:

- Marathi/Hindi: Devanagari fonts such as Nirmala, Mangal, or any locally installed Marathi/Hindi-compatible font
- Gujarati: Gujarati font such as Shruti, Nirmala UI, or local Gujarati script font
- Bengali: Bengali font such as Nirmala UI, Kalpurush, or a local Bengali script font

Google Fonts Noto Sans Regular font files used by the current renderer are stored locally:

- `assets/fonts/devanagari/NotoSansDevanagari-Regular.ttf` (Marathi and Hindi)
- `assets/fonts/gujarati/NotoSansGujarati-Regular.ttf`
- `assets/fonts/bengali/NotoSansBengali-Regular.ttf`

The renderer shapes text with HarfBuzz and rasterizes shaped glyph IDs with FreeType. This is deliberate: this Windows Pillow build does not provide RAQM, so `ImageDraw.text` alone cannot be relied on for Indic shaping. Font coverage is checked against the selected font cmap before rendering; missing characters fail explicitly.

## Running rendering tests

```powershell
.\.venv\Scripts\python -m pytest tests/test_script_rendering.py -q
```

## Smoke-test generation

A smoke test renders the four scripts and saves diagnostic images.

```powershell
.\.venv\Scripts\python scripts/run_render_tests.py
```

The four diagnostic PNGs are kept in `previews/`. Run the rendering tests before generating sample boards:

```powershell
.\.venv\Scripts\python -m pytest tests/test_script_rendering.py -q
```

Generate eight shop boards (two per language) using the fixed default seed:

```powershell
.\.venv\Scripts\python scripts/generate_shop_board_smoke_test.py
```

The script refuses to overwrite an existing output file. Use another seed or output root for another run. Images go to `data/generated/`, one JSON annotation per image to `data/annotations/`, and a contact sheet to `previews/`. Content lists and board options are configurable in `configs/generation_config.json`. Supplied custom text is retained; the content engine does not translate or transliterate it.

The shop-board generator cycles through centered/subtitle, left-aligned/services, headline/contact, multiline hierarchy, and simple logo layouts. A small scene sample can place those boards into a procedural storefront with translation only:

```powershell
.\.venv\Scripts\python scripts/generate_storefront_scene_smoke_test.py
```

Each base scene keeps its signboard as a separate `_board.png` layer. Scene annotations record the integer translation and retain both board-local and final-image geometry. Base scenes remain untransformed; controlled difficulty variants are generated separately below.

## Controlled scene-difficulty pilot

The pilot starts with the existing 12 storefront scenes and writes variants separately; source scenes remain unchanged:

```powershell
.\.venv\Scripts\python scripts/generate_scene_difficulty_pilot.py
```

The configured profiles are `D0_clean`, `D2_distance`, `D2_perspective`, `D2_lighting`, `D2_blur`, and `D2_occlusion`. Each non-clean profile changes only its named factor. Distance resizes the full image and updates dimensions and geometry; perspective applies an OpenCV homography to the scene and text polygons; lighting and blur leave geometry unchanged; blur may include downsampling and upsampling; occlusion retains every text region and records estimated visibility. No profile combines these effects.

Occlusion visibility uses a documented coarse **axis-aligned bounding-box unoccluded-area fraction**, not glyph-mask pixel visibility. `visible` means at least 0.99 estimated visible, `partially_occluded` means 0.01 to under 0.99, and `fully_occluded` means under 0.01. Fully or partially covered text annotations are retained.

The script writes one 12-sample contact sheet per profile, per-sample annotation overlays, per-profile validation reports, and a consolidated generation report under `previews/scene_difficulty_seed_20261029/`. Profile images and annotations are saved separately under `data/generated/scene_difficulty_seed_20261029/`. Run the dataset validator for each profile directory to inspect its structural checks.

## Validation

```powershell
.\.venv\Scripts\python -m pytest -q
```

Validate a generated sample and optionally save a machine-readable report:

```powershell
.\.venv\Scripts\python scripts/validate_dataset.py `
  --images data/generated/storefront_scenes_seed_20261029 `
  --annotations data/annotations/storefront_scenes_seed_20261029 `
  --report previews/storefront_scenes_seed_20261029_validation.json
```

The validator checks image/annotation pairing, labels, unique IDs, image dimensions, annotation schemas, rendered text geometry, transformed-parent links, visibility metadata, and exact board-layer placement for untransformed scenes.

## Coordinate convention

Annotations use a top-left origin with x increasing right and y increasing down. Bounding boxes and polygons are derived from actual rendered glyph masks. Coordinates refer to pixel edges: minimum edges are inclusive and maximum edges are exclusive. Samples are marked for visual review even after structural checks pass.

## Limitations and known risks

- Unicode cmap checks and HarfBuzz shaping tests do not prove every rendered glyph is visually correct; inspect generated previews for each script.
- The current scene-difficulty pilot is controlled and small. Geometry for transformed samples is validated against recorded transformations; occlusion visibility is an estimate based on text bounding boxes.
- No OCR model is trained or evaluated by these scripts. No OCR performance claim is made.
- Synthetic data is a controlled approximation of real scene text.
- Decorative or highly stylized typography can still require manual review.

## Reproducibility

Use fixed seeds for generation, record the seed in outputs, and keep the generation configuration under version control. For large collections, store generated datasets outside Git and keep only scripts, configs and metadata in the repository.

## Future work

The project is designed to support later evaluation of OCR performance under:

- language
- image category
- difficulty profile
- individual scene degradation factors

Those experiments should remain separate from the synthetic data generation itself.
