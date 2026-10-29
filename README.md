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
- `src/scene_transformer.py`: seeded, independent distance, perspective, lighting, blur, and occlusion profiles
- `src/background_assets.py`: background manifest, image-integrity, provenance, and geometry gates
- `src/background_composer.py`: board-to-surface perspective composition and geometry replay checks
- `src/background_pilot.py`: bounded background pilot generation and evidence export
- `src/reference_inspired_pilot.py`: balanced procedural scenes inspired by reference composition only
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

### Font registry and style validation

`configs/generation_config.json` contains the explicit font registry. The six generation-approved candidates are Noto Sans Regular and Bold files for Devanagari (shared by Marathi and Hindi), Gujarati, and Bengali. Each has a local SIL Open Font License 1.1 file and a provenance note. Bold files were extracted from the corresponding Noto archives already present under `assets/fonts/`; no font downloader or unverified URL is used.

The local `Annapurna Bold.ttf`, `Chandra Head Regular.ttf`, and `Kirati Regular.ttf` files are recorded as unverified candidates, not approved for generation. Their current font cmap checks do not cover the target Unicode scripts, and licensing/provenance has not been established. A font's filename alone is not treated as proof of script support.

Run the diagnostic before generating a batch:

```powershell
.\.venv\Scripts\python scripts/validate_font_registry.py
```

The command tests vowel/combining marks, conjuncts, punctuation, numerals, multiline rendering, missing glyphs, exact Unicode preservation, and ink bounds for every configured candidate and target language. It saves diagnostic renderings, a contact sheet, and `font_validation_report.json` under `previews/font_validation_seed_20261029/`. Only fonts that pass the rendering checks and have a verified local license are eligible.

To add another font safely, obtain it manually from its official maintainer or the official Google Fonts repository at a pinned revision; inspect its embedded family/style/weight and actual Unicode cmap; verify HarfBuzz shaping with this diagnostic; retain the corresponding license text and record the precise source/revision and checksum in the registry. Do not fetch a font from an unverified mirror, infer script support from its filename, or mark unknown license terms as verified. Add actual files for each desired style; the pipeline never synthesizes bold or decorative variants.

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

## Reference-inspired procedural pilot

The supplied reference photos are used for broad composition inspiration only. This pipeline does not read or composite reference pixels and does not use reference annotations as ground truth. It creates new 2D procedural shopfront and outdoor-hoarding scenes with locally rendered text; the results are **not photorealistic**.

Generate the reproducible 24-scene pilot:

```powershell
.\.venv\Scripts\python scripts/generate_reference_inspired_pilot.py
```

The pilot balances Marathi, Hindi, Gujarati, and Bengali; shopfront and outdoor-hoarding scenes; and all six isolated difficulty profiles. It writes profile-separated images and annotations under `data/generated/reference_inspired_seed_20261029/`, plus contact sheets, annotation overlays, and validation reports under `previews/reference_inspired_seed_20261029/`. The command refuses to overwrite existing outputs. Review the contact sheets and overlays in addition to the automated geometry and dataset checks. Occlusion visibility is still estimated from axis-aligned text boxes, not glyph pixels. This is a data-generation pilot only; it does not train or evaluate OCR.

## Multilingual dataset batch

After font validation, generate the configured 1,000-image batch (or use `--sample-count` for a smaller reproducibility run):

```powershell
.\.venv\Scripts\python scripts/generate_multilingual_dataset.py
```

The command requires the font-validation report, samples validated licensed fonts evenly within each language, and balances the four languages, the shopfront/outdoor-hoarding scene categories, and all six difficulty profiles. The seed, font/style/weight, text sizes and hierarchy, palette and contrast ratio, layout, background provenance, parent links, and transformation parameters are retained in annotations/manifests. It refuses to reuse an existing output or preview directory. Use `--output-root`, `--preview-root`, `--font-report`, `--seed`, and `--sample-count` to select a new reproducible run.

Contact sheets are grouped by language, font, and difficulty profile. The generator uses eligible manifest backgrounds when available and otherwise labels 2D procedural illustrations explicitly; these are not photorealistic. The report lists real-photo/AI source counts and any nonprocedural-background shortfall. In the current manifest there are no eligible background assets, so the 1,000-image batch will use procedural illustrations only. It does not train or evaluate OCR and is not a basis for a 10,000-image scale-up without reviewing its contact sheets and reports.

## Verified background asset pipeline

The optional background composer accepts local shopfront/outdoor-hoarding images and images generated manually by a user-selected image-generation tool. The project does not scrape the web, call an image service, or require a GPU. Existing backgrounds and difficulty outputs are not overwritten.

Put real photographs in `assets/backgrounds/real_photos/` and user-created AI backgrounds in `assets/backgrounds/ai_generated/`. Common JPEG, PNG, WebP, TIFF, and BMP files are supported. Keep originals unchanged. Asset image folders are ignored by Git by default so private or unlicensed source images are not accidentally committed; add an asset only if you have the right to redistribute it.

For each image, add an entry to `assets/backgrounds/manifest.json`. Record only facts you know. A record uses these fields:

```json
{
  "background_id": "unique-local-id",
  "asset_id": "unique-local-id",
  "relative_path": "assets/backgrounds/real_photos/shopfront.jpg",
  "original_filename": "shopfront.jpg",
  "source_type": "real_photo",
  "scene_category": "shopfront",
  "image_width": 1600,
  "image_height": 1200,
  "provenance": "Describe how this exact file was obtained; do not guess a URL or source.",
  "licensing": {
    "terms": "Record the actual license or state that it is unknown.",
    "verification_status": "unverified",
    "research_use_allowed": false,
    "redistribution_allowed": false,
    "evidence": ""
  },
  "generation_prompt": null,
  "created_date": null,
  "signboard_surface_polygon": [[280, 150], [1320, 155], [1300, 440], [300, 435]],
  "surface_verification_status": "pending",
  "manual_review_status": "pending",
  "validation_warnings": ["Replace example geometry and metadata after reviewing the actual image."]
}
```

The JSON above is a field example, not a usable asset record. Its polygon, dimensions, provenance, rights, and review state must be replaced with information from the actual image. By default an asset is eligible only when its file and dimensions validate, it has an explicitly verified research-use license, a manually checked convex blank-surface polygon in frame, `surface_verification_status` is `verified`, `manual_review_status` is `approved`, and no unresolved validation warnings remain. Public accessibility alone does not establish a license. Do not mark rights verified without evidence. If the blank surface cannot be identified confidently, leave it pending and the generator will exclude it rather than guess.

Polygon coordinates use pixel edges and the order top-left, top-right, bottom-right, bottom-left. Mark the actual blank usable surface, not the surrounding sign frame. Minimum image and surface dimensions, provenance/license requirements, and pilot targets are in `configs/generation_config.json`.

For a manually generated background, use a prompt for an Indian shopfront or outdoor hoarding with a clear, mostly blank sign surface; avoid readable text, logos, watermarks, and recognizable real people. Import the saved image yourself, then record its actual asset ID, prompt, creation date if known, tool or provenance details if known, and applicable terms. The project does not call or assume a generation service. Do not identify an image as AI-generated unless that is known.

Run a one-scene geometry smoke test using a procedural test fixture (not a photo, not an AI output, and not a pilot sample):

```powershell
.\.venv\Scripts\python scripts/run_background_composition_smoke_test.py
```

It saves the untouched test background, composed scene, board layer, perspective variant, annotations, and an overlay under `previews/background_composition_smoke_test_seed_20261029/`. This verifies composition mechanics only; inspect the overlay and do not treat it as photorealism evidence.

Run the bounded pilot with the fixed seed:

```powershell
.\.venv\Scripts\python scripts\generate_photorealistic_pilot.py
```

The default run reads only eligible assets in the manifest and writes to the new `data/generated/photorealistic_pilot_seed_20261029/` and `previews/photorealistic_pilot_seed_20261029/` directories. It refuses to overwrite listed outputs. The pilot uses each selected background once, schedules the four languages and the six existing difficulty profiles as evenly as the available asset count permits, and targets 12 shopfronts and 12 outdoor hoardings, 12 real photographs and 12 AI-generated images, and six samples per language. If those verified assets are unavailable, it generates fewer (or zero) samples, records excluded assets and the shortfall, and does not mark the pilot complete. Add verified assets and use a new output root for another run. `--require-complete` makes a shortfall return a nonzero exit code.

Each final annotation preserves its exact rendered transcription and records the background ID, provenance/license fields, source and final signboard polygons, board-to-background homography, dimensions, font/rendering values, random seeds, difficulty factors, and parent image/annotation. Source images stay in place; the original composed background, transparent board layer, and final transformed image are separate. Annotation polygons are transformed mathematically; the axis-aligned boxes are derived from those polygons. Invalid or out-of-frame geometry is rejected.

Inspect the per-profile contact sheets in the preview directory and representative `overlays/` before treating outputs as visually reviewed. `pilot_manifest.json`, `validation_report.json`, `generation_report.json`, and per-profile validation JSON files provide machine-readable counts and checks. To run structural validation independently:

```powershell
.\.venv\Scripts\python scripts/validate_dataset.py `
  --images data/generated/photorealistic_pilot_seed_20261029/D0_clean/images `
  --annotations data/generated/photorealistic_pilot_seed_20261029/D0_clean/annotations
```

Repeat with each profile directory. Automated checks cover image/annotation pairing, exact Unicode fields, polygon/bbox consistency and frame bounds, source provenance, surface geometry, transformed-parent replay, and coverage counts. They do not prove photographic realism or licensing, and contact sheets still require human visual inspection. No OCR model is trained or evaluated.

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
