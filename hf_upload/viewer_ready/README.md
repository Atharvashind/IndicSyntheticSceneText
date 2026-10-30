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
configs:
  - config_name: default
    data_files:
      - split: train
        path: train-00000-of-00001.parquet
      - split: validation
        path: validation-00000-of-00001.parquet
      - split: test
        path: test-00000-of-00001.parquet
---

# Indic Synthetic Scene Text 1000

This release packages 1,000 synthetic scene-text images for Marathi, Hindi,
Gujarati, and Bengali as three image-aware Parquet splits. The images are
procedurally drawn illustrations, not photographs or photorealistic scenes.

## Dataset metadata

| Property | Value |
|---|---:|
| Total records | 1,000 |
| Languages | Marathi, Hindi, Gujarati, Bengali |
| Shopfront scenes | 500 |
| Outdoor hoarding scenes | 500 |
| Difficulty profiles | `D0_clean`, `D2_blur`, `D2_distance`, `D2_lighting`, `D2_occlusion`, `D2_perspective` |
| Image format | PNG, encoded in Parquet as `datasets.Image()` |
| Annotation format | Original UTF-8 JSON, preserved per record |
| Split seed | 42 |

Language counts are balanced at 250 each. Original difficulty-profile counts
are 166 or 167 per profile. Exact split-wise counts and joint
language/scene/profile distributions are recorded in
[`metadata/split_distribution.json`](metadata/split_distribution.json).

## Splits

| Split | Records | Proportion |
|---|---:|---:|
| Train | 700 | 70% |
| Validation | 150 | 15% |
| Test | 150 | 15% |

Records were stratified jointly by language, scene category, and difficulty
profile. The split assignment uses seed `42`; all records sharing a source
scene identity or matching the near-duplicate screen are kept in one split.
No exact duplicate image hashes were found. Near-duplicate screening first
compares 64-bit perceptual hashes (Hamming distance at most 2), then groups
candidate pairs whose global grayscale SSIM at 128 x 128 is at least 0.995.
This conservative screen avoids treating shared procedural layout alone as
proof that two images are the same scene.

The complete sample-to-split mapping is in
[`metadata/split_manifest.jsonl`](metadata/split_manifest.jsonl). The
standalone preparation script, [`prepare_viewer_ready.py`](prepare_viewer_ready.py),
recreates the Parquet files and local validation report from the original
staging files with the same seed. It requires the `datasets`, `numpy`,
`opencv-python-headless`, and `Pillow` packages.

## Schema

| Column | Type | Description |
|---|---|---|
| `image` | `datasets.Image()` | Original PNG image embedded in the Parquet record. |
| `annotation` | string | Original per-image annotation JSON text, unchanged; parse with `json.loads`. |
| `sample_id` | string | Original manifest sample identifier. |
| `language` | string | Original language label. |
| `scene_category` | string | Original scene type, such as `shopfront` or `outdoor_hoarding`. |
| `category` | string | Original category label, such as `shop_board` or `advertising_hoarding`. |
| `difficulty_profile` | string | Original manifest profile. |
| `manifest_json` | string | Original JSONL manifest record for this sample. |

The annotation JSON is preserved as a whole; no keys or text regions are
dropped or renamed. It includes the source top-level metadata, generation and
scene parameters, provenance, validation information, coordinate convention,
and the ordered `text_regions` array. The regions retain their source
transcriptions, `bbox`, `polygon`, board geometry, reading order, font,
rendering, and visibility data. The source schema includes two or three text
regions per record. Coordinates and polygon point ordering are not normalized.

## Loading

```python
import json
from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files={
        "train": "train-00000-of-00001.parquet",
        "validation": "validation-00000-of-00001.parquet",
        "test": "test-00000-of-00001.parquet",
    },
)

sample = dataset["train"][0]
image = sample["image"]  # decoded PIL image
annotation = json.loads(sample["annotation"])
print(sample["sample_id"], annotation["text_regions"][0]["transcription"])
```

After the files are uploaded to the dataset repository, they can also be loaded
directly from the Hub:

```python
dataset = load_dataset(
    "parquet",
    data_files={
        split: (
            "https://huggingface.co/datasets/Aus-23/"
            "indic-synthetic-scene-text-1000/resolve/main/"
            f"{split}-00000-of-00001.parquet"
        )
        for split in ("train", "validation", "test")
    },
)
```

## Validation and provenance

[`validation_reports/viewer_ready_validation.json`](validation_reports/viewer_ready_validation.json)
records local image/annotation checks, exact record counts, split
distributions, leakage checks, and a `load_dataset("parquet", ...)` round-trip
test that decodes the images and compares every annotation with its source.
The source manifests and validation reports are copied unchanged under
[`provenance/`](provenance/); the original PNGs and JSON annotations remain in
the separate source staging directory and are not rewritten by this export.

## Limitations

These are synthetic 2D illustrations and do not represent the visual diversity
of real-world signage. They do not establish OCR accuracy, generalization, or
deployment robustness. Similar procedural scene layouts may occur in multiple
splits even though source-linked and high-confidence near-duplicate scenes are
kept together. OCR or geometry quality should not be inferred solely from the
automated structural validation.

No dataset license is asserted. Redistribution rights for rendered assets,
font-derived imagery, and source assets have not been independently established.
The current Hub repository was observed as public and ungated during
preparation; this workflow does not change repository visibility or gating.

## Upload command (not executed)

Run this only after explicitly approving the upload. It uploads this directory
to the existing dataset repository and does not change its visibility or
gating:

```powershell
$RepoId = "Aus-23/indic-synthetic-scene-text-1000"
$Upload = "D:\IndicSyntheticSceneText\hf_upload\viewer_ready"

if (!(Test-Path "$Upload\README.md")) {
    throw "Dataset README is missing."
}

hf upload $RepoId $Upload . --repo-type dataset
```
