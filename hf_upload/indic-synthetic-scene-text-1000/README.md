---
pretty_name: Indic Synthetic Scene Text 1000
language:
  - mr
  - hi
  - gu
  - bn
task_categories:
  - image-to-text
tags:
  - synthetic-data
  - scene-text
  - ocr
  - computer-vision
  - multilingual
---

# Indic Synthetic Scene Text 1000

This dataset contains 1,000 reproducible synthetic scene-text images for Marathi, Hindi, Gujarati, and Bengali. It is intended for controlled research and prototyping involving multilingual scene-text detection and recognition.

The scenes are **procedurally drawn 2D illustrations**, not photographs or photorealistic images. Text is rendered locally from exact Unicode strings using the project's Indic shaping pipeline. This release does not evaluate an OCR model and provides no evidence of improved real-world OCR accuracy.

## Dataset composition

| Property | Verified count |
|---|---:|
| Primary samples | 1,000 |
| Marathi | 250 |
| Hindi | 250 |
| Gujarati | 250 |
| Bengali | 250 |
| Shopfront scenes | 500 |
| Outdoor hoarding scenes | 500 |
| Procedural illustration backgrounds | 1,000 |

The six isolated difficulty profiles are `D0_clean`, `D2_blur`, `D2_distance`, `D2_lighting`, `D2_occlusion`, and `D2_perspective`. Their counts are 167, 166, 167, 167, 166, and 167, respectively. The exact counts and per-sample assignments are in [`metadata/manifest.jsonl`](metadata/manifest.jsonl).

`images/` and `annotations/` are grouped by difficulty profile. Each manifest record points to one primary image and its annotation. The profile folders also retain the parent images and annotations referenced by transformed samples, plus the signboard layer files referenced by annotations. Those nested support files preserve provenance and are **not additional primary samples**.

## Annotation format

Each primary image has one UTF-8 JSON annotation. The schema preserves the generated transcription and final-image geometry; no transcription, coordinates, or polygon values were normalized for staging.

Common top-level fields include:

- `sample_id`, `image`, `image_width`, `image_height`
- `language`, `script`, `category`, `scene_difficulty`
- `random_seed`, `generation_parameters`, `scene_parameters`
- `parent`, `board_layer`, and background provenance fields
- `coordinate_convention` and validation status
- `text_regions`, an ordered array of text-region records

Each text-region record includes `region_id`, exact Unicode `transcription`, `language`, `script`, `reading_order`, `bbox`, `polygon`, `font`, `rendering`, and visibility metadata. Boxes and polygons use pixel-edge coordinates: origin at the top-left, x increases rightward, y increases downward, minimum edges are inclusive, and maximum edges are exclusive. Polygon points are stored in the source JSON's original order.

The JSONL manifest has one row per primary sample and includes its ID, language, category, difficulty profile, font ID, seed, background source, and staged relative paths. It also retains the original batch-relative paths as `source_image` and `source_annotation`. The source manifest is preserved separately at `metadata/source_generation_manifest.json`.

## Loading a sample

The files are raw PNG and JSON artifacts. Uploading them does not by itself make the repository a ready-to-use `datasets.load_dataset()` image dataset. The JSONL index can be read directly, and images and annotations loaded by their staged paths:

```python
import json
from pathlib import Path

from PIL import Image

dataset_root = Path(".")
with (dataset_root / "metadata" / "manifest.jsonl").open(encoding="utf-8") as index:
    sample = json.loads(next(index))

image = Image.open(dataset_root / sample["image"]).convert("RGB")
annotation = json.loads(
    (dataset_root / sample["annotation"]).read_text(encoding="utf-8")
)
print(sample["sample_id"], annotation["text_regions"][0]["transcription"])
```

For a `datasets` workflow, load the JSONL records as JSON and explicitly resolve/cast their `image` paths; the index's path strings are not automatically decoded into image objects.

## Generation and validation

The source project generated the batch with seed `20261030`, balancing the four languages and two scene categories and distributing samples as evenly as possible across the six profiles. It used the project's validated local Noto Sans font registry and recorded font, text, layout, palette, background, and transformation choices in the annotations and manifest.

The included source reports are under [`validation_reports/`](validation_reports/). All six profile reports and the consolidated generation report record successful structural validation, with no failed samples. The staging audit additionally checks primary image/annotation pairing, unique IDs and filenames, valid JSON and PNG files, annotation dimensions and labels, transcription field types, manifest consistency, and linked parent/board-layer files. Its result is [`validation_reports/hf_upload_validation_report.json`](validation_reports/hf_upload_validation_report.json).

Automated structural validation does not establish visual realism, perfect human readability across every image, or downstream OCR performance. Occlusion visibility is an estimated bounding-box-area measure rather than pixel-level glyph visibility. Contact-sheet review is useful for quality control but does not replace evaluation on real-world data.

## Intended use and limitations

Appropriate uses include controlled experimentation, annotation-pipeline testing, and exploratory multilingual scene-text research. Do not interpret these images as photographs, as a representative sample of real signage, or as proof of OCR accuracy or robustness in deployment.

## Source and licensing

The generation source repository is [Atharvashind/IndicSyntheticSceneText](https://github.com/Atharvashind/IndicSyntheticSceneText), as configured in the project Git remote.

**No dataset license is asserted.** Keep the Hub repository private until the dataset owner has selected a license and reviewed rights to redistribute the rendered assets, font-derived imagery, and any source assets. This staging export contains no font binaries. Font paths and family/style metadata in annotations describe the generating project and may not resolve in this repository. Although the project's configured Noto font files have matching SIL Open Font License records, the right to redistribute this dataset and all related assets still requires review. Do not treat the absence of a license field as permission to reuse.
