from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from datasets import Dataset, Features, Image as DatasetImage, Value, load_dataset
from PIL import Image


SPLIT_TARGETS = {"train": 700, "validation": 150, "test": 150}
SPLIT_FILES = {
    split: f"{split}-00000-of-00001.parquet" for split in SPLIT_TARGETS
}
PHASH_MAX_DISTANCE = 2
SSIM_MINIMUM = 0.995
IMAGE_FEATURES = Features(
    {
        "image": DatasetImage(),
        "annotation": Value("string"),
        "sample_id": Value("string"),
        "language": Value("string"),
        "scene_category": Value("string"),
        "category": Value("string"),
        "difficulty_profile": Value("string"),
        "manifest_json": Value("string"),
    }
)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, item: int) -> int:
        while self.parents[item] != item:
            self.parents[item] = self.parents[self.parents[item]]
            item = self.parents[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parents[second_root] = first_root


def _read_source(source_root: Path) -> list[dict[str, Any]]:
    manifest_path = source_root / "metadata" / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest not found: {manifest_path}")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    for line_number, manifest_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not manifest_line.strip():
            continue
        manifest = json.loads(manifest_line)
        sample_id = manifest["sample_id"]
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
        seen_ids.add(sample_id)

        image_relative = manifest["image"]
        annotation_relative = manifest["annotation"]
        if image_relative in seen_images:
            raise ValueError(f"Duplicate image path in manifest: {image_relative}")
        seen_images.add(image_relative)

        image_path = source_root / image_relative
        annotation_path = source_root / annotation_relative
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image from manifest line {line_number} is missing: {image_path}"
            )
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Annotation from manifest line {line_number} is missing: "
                f"{annotation_path}"
            )

        annotation_text = annotation_path.read_bytes().decode("utf-8")
        annotation = json.loads(annotation_text)
        if annotation.get("sample_id") != sample_id:
            raise ValueError(f"Annotation sample_id mismatch for {sample_id}")
        regions = annotation.get("text_regions")
        if not isinstance(regions, list) or not regions:
            raise ValueError(f"Missing text_regions for {sample_id}")
        for region in regions:
            if not isinstance(region, dict):
                raise ValueError(f"Invalid text region in {sample_id}")
            if "bbox" not in region or "polygon" not in region:
                raise ValueError(
                    f"Text region missing bbox or polygon for {sample_id}"
                )

        with Image.open(image_path) as image:
            if image.format != "PNG":
                raise ValueError(f"Expected PNG image: {image_path}")
            image.verify()
        with Image.open(image_path) as image:
            if (image.width, image.height) != (
                annotation["image_width"],
                annotation["image_height"],
            ):
                raise ValueError(f"Image dimensions disagree for {sample_id}")

        records.append(
            {
                "sample_id": sample_id,
                "image_path": image_path,
                "annotation_path": annotation_path,
                "annotation_text": annotation_text,
                "annotation_object": annotation,
                "manifest": manifest,
                "manifest_json": manifest_line,
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "stratum": (
                    manifest["language"],
                    manifest["scene_category"],
                    manifest["profile"],
                ),
            }
        )

    if len(records) != 1000:
        raise ValueError(f"Expected 1,000 source records, found {len(records)}")
    return records


def _global_ssim(first: np.ndarray, second: np.ndarray) -> float:
    mean_first = float(first.mean())
    mean_second = float(second.mean())
    variance_first = float(first.var())
    variance_second = float(second.var())
    covariance = float(
        np.mean((first - mean_first) * (second - mean_second))
    )
    c1 = 6.5025
    c2 = 58.5225
    return (
        (2 * mean_first * mean_second + c1) * (2 * covariance + c2)
    ) / (
        (mean_first**2 + mean_second**2 + c1)
        * (variance_first + variance_second + c2)
    )


def _perceptual_groups(
    records: list[dict[str, Any]], union_find: UnionFind
) -> list[dict[str, Any]]:
    images: list[np.ndarray] = []
    hashes: list[int] = []
    for record in records:
        gray = cv2.imread(str(record["image_path"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"OpenCV could not decode {record['image_path']}")
        resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        images.append(resized.astype(np.float32))
        low_resolution = cv2.resize(
            resized, (32, 32), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        coefficients = cv2.dct(low_resolution)[:8, :8].reshape(-1)[1:]
        median = float(np.median(coefficients))
        hashes.append(
            sum(
                int(value > median) << bit
                for bit, value in enumerate(coefficients)
            )
        )

    candidate_pairs = 0
    near_duplicate_pairs = 0
    for first_index in range(len(records)):
        for second_index in range(first_index):
            distance = (hashes[first_index] ^ hashes[second_index]).bit_count()
            if distance > PHASH_MAX_DISTANCE:
                continue
            candidate_pairs += 1
            similarity = _global_ssim(
                images[first_index], images[second_index]
            )
            if similarity >= SSIM_MINIMUM:
                union_find.union(first_index, second_index)
                near_duplicate_pairs += 1

    return [
        {
            "method": "64-bit pHash candidate followed by 128x128 grayscale "
            "global SSIM",
            "phash_max_hamming_distance": PHASH_MAX_DISTANCE,
            "ssim_minimum": SSIM_MINIMUM,
            "phash_candidate_pair_count": candidate_pairs,
            "near_duplicate_pair_count": near_duplicate_pairs,
        }
    ]


def _build_scene_groups(
    records: list[dict[str, Any]],
) -> tuple[list[list[int]], dict[str, Any]]:
    union_find = UnionFind(len(records))
    identity_owner: dict[tuple[str, str], int] = {}
    exact_image_hashes: dict[str, int] = {}

    for index, record in enumerate(records):
        annotation = record["annotation_object"]
        identities = [
            ("parent_sample_id", annotation["parent"]["sample_id"]),
            ("source_board_sample_id", annotation["source_board_sample_id"]),
            ("image_sha256", record["image_sha256"]),
        ]
        for namespace, value in identities:
            key = (namespace, str(value))
            if key in identity_owner:
                union_find.union(index, identity_owner[key])
            else:
                identity_owner[key] = index
        if record["image_sha256"] in exact_image_hashes:
            union_find.union(index, exact_image_hashes[record["image_sha256"]])
        else:
            exact_image_hashes[record["image_sha256"]] = index

    perceptual_report = _perceptual_groups(records, union_find)
    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped_indices[union_find.find(index)].append(index)
    groups = sorted(
        grouped_indices.values(),
        key=lambda indices: min(records[index]["sample_id"] for index in indices),
    )
    duplicate_groups = [group for group in groups if len(group) > 1]
    return groups, {
        "exact_duplicate_image_count": len(records) - len(exact_image_hashes),
        "identity_group_count": len(duplicate_groups),
        "identity_group_size_distribution": dict(
            sorted(Counter(len(group) for group in duplicate_groups).items())
        ),
        **perceptual_report[0],
    }


def _stratum_counts(
    records: list[dict[str, Any]],
) -> Counter[tuple[str, str, str]]:
    return Counter(record["stratum"] for record in records)


def _assign_groups(
    records: list[dict[str, Any]],
    groups: list[list[int]],
    seed: int,
) -> dict[str, str]:
    rng = random.Random(seed)
    stratum_totals = _stratum_counts(records)
    assigned: dict[str, str] = {}
    assigned_by_split: Counter[str] = Counter()
    assigned_by_stratum: dict[tuple[str, tuple[str, str, str]], int] = Counter()
    multi_record_groups = [group for group in groups if len(group) > 1]
    rng.shuffle(multi_record_groups)
    multi_record_groups.sort(key=len, reverse=True)

    for group in multi_record_groups:
        contributions = Counter(records[index]["stratum"] for index in group)
        feasible = [
            split
            for split, target in SPLIT_TARGETS.items()
            if assigned_by_split[split] + len(group) <= target
        ]
        if not feasible:
            raise ValueError("Duplicate groups cannot fit the requested split sizes")

        scored: list[tuple[float, float, str]] = []
        for split in feasible:
            local_cost = 0.0
            for stratum, amount in contributions.items():
                expected = (
                    stratum_totals[stratum]
                    * SPLIT_TARGETS[split]
                    / len(records)
                )
                current = assigned_by_stratum[(split, stratum)]
                local_cost += (
                    (current + amount - expected) ** 2
                    - (current - expected) ** 2
                ) / max(1.0, expected)
            current_total = assigned_by_split[split]
            target_total = SPLIT_TARGETS[split]
            global_cost = (
                (current_total + len(group) - target_total) ** 2
                - (current_total - target_total) ** 2
            ) / target_total
            scored.append((local_cost + global_cost, rng.random(), split))
        split = min(scored)[2]
        assigned_by_split[split] += len(group)
        for stratum, amount in contributions.items():
            assigned_by_stratum[(split, stratum)] += amount
        for index in group:
            assigned[records[index]["sample_id"]] = split

    singletons_by_stratum: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for group in groups:
        if len(group) == 1:
            index = group[0]
            singletons_by_stratum[records[index]["stratum"]].append(index)

    remaining_targets = {
        split: SPLIT_TARGETS[split] - assigned_by_split[split]
        for split in SPLIT_TARGETS
    }
    singleton_count = sum(len(indices) for indices in singletons_by_stratum.values())
    if singleton_count != sum(remaining_targets.values()):
        raise ValueError("Scene-group allocation did not preserve the total size")
    if singleton_count == 0:
        return assigned

    total_remaining = singleton_count
    desired_by_stratum: dict[
        tuple[str, str, str], dict[str, float]
    ] = {}
    allocations: dict[
        tuple[str, str, str], dict[str, int]
    ] = {}
    split_order = list(SPLIT_TARGETS)
    for stratum in sorted(singletons_by_stratum):
        count = len(singletons_by_stratum[stratum])
        desired = {
            split: count * remaining_targets[split] / total_remaining
            for split in split_order
        }
        desired_by_stratum[stratum] = desired
        allocation = {
            split: math.floor(desired[split]) for split in split_order
        }
        remainder = count - sum(allocation.values())
        tie_order = {split: rng.random() for split in split_order}
        candidates = sorted(
            (
                split
                for split in split_order
                if remaining_targets[split] > 0
            ),
            key=lambda split: (
                -(desired[split] - allocation[split]),
                tie_order[split],
            ),
        )
        if remainder > len(candidates):
            raise ValueError("Could not round stratified singleton allocations")
        for split in candidates[:remainder]:
            allocation[split] += 1
        allocations[stratum] = allocation

    allocated_totals = {
        split: sum(row[split] for row in allocations.values())
        for split in split_order
    }
    while allocated_totals != remaining_targets:
        sources = [
            split
            for split in split_order
            if allocated_totals[split] > remaining_targets[split]
        ]
        destinations = [
            split
            for split in split_order
            if allocated_totals[split] < remaining_targets[split]
        ]
        if not sources or not destinations:
            raise ValueError("Could not balance stratified split totals")

        moves: list[
            tuple[float, float, tuple[str, str, str], str, str]
        ] = []
        for source in sources:
            for destination in destinations:
                for stratum, allocation in allocations.items():
                    if allocation[source] == 0:
                        continue
                    desired = desired_by_stratum[stratum]
                    before = (
                        (allocation[source] - desired[source]) ** 2
                        + (allocation[destination] - desired[destination]) ** 2
                    )
                    after = (
                        (allocation[source] - 1 - desired[source]) ** 2
                        + (allocation[destination] + 1 - desired[destination])
                        ** 2
                    )
                    moves.append(
                        (
                            after - before,
                            rng.random(),
                            stratum,
                            source,
                            destination,
                        )
                    )
        if not moves:
            raise ValueError("No valid move can satisfy the exact split sizes")
        _, _, stratum, source, destination = min(moves)
        allocations[stratum][source] -= 1
        allocations[stratum][destination] += 1
        allocated_totals[source] -= 1
        allocated_totals[destination] += 1

    for stratum, indices in singletons_by_stratum.items():
        rng.shuffle(indices)
        offset = 0
        for split in split_order:
            end = offset + allocations[stratum][split]
            for index in indices[offset:end]:
                assigned[records[index]["sample_id"]] = split
            offset = end

    return assigned


def _distribution(
    records: list[dict[str, Any]], assignments: dict[str, str]
) -> dict[str, Any]:
    dimensions = {
        "language": lambda record: record["manifest"]["language"],
        "scene_category": lambda record: record["manifest"]["scene_category"],
        "difficulty_profile": lambda record: record["manifest"]["profile"],
        "category": lambda record: record["manifest"]["category"],
    }
    result: dict[str, Any] = {}
    for split in SPLIT_TARGETS:
        split_records = [
            record
            for record in records
            if assignments[record["sample_id"]] == split
        ]
        result[split] = {"records": len(split_records)}
        for dimension, value_for in dimensions.items():
            result[split][dimension] = dict(
                sorted(Counter(value_for(record) for record in split_records).items())
            )

    joint_counts: dict[str, Counter[str]] = {
        split: Counter() for split in SPLIT_TARGETS
    }
    for record in records:
        stratum = "|".join(record["stratum"])
        joint_counts[assignments[record["sample_id"]]][stratum] += 1
    result["joint_stratification"] = {
        split: dict(sorted(counts.items()))
        for split, counts in joint_counts.items()
    }
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _copy_provenance(source_root: Path, output_root: Path) -> None:
    metadata_source = source_root / "metadata"
    metadata_output = output_root / "provenance" / "metadata"
    metadata_output.mkdir(parents=True, exist_ok=True)
    for name in (
        "dataset_metadata.json",
        "manifest.jsonl",
        "source_generation_manifest.json",
    ):
        source = metadata_source / name
        if source.is_file():
            shutil.copy2(source, metadata_output / name)

    reports_source = source_root / "validation_reports"
    reports_output = output_root / "provenance" / "validation_reports"
    for source in reports_source.rglob("*"):
        if source.is_file():
            destination = reports_output / source.relative_to(reports_source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def _validate_loaded_parquet(
    output_root: Path, records: list[dict[str, Any]], assignments: dict[str, str]
) -> dict[str, Any]:
    data_files = {
        split: str(output_root / filename)
        for split, filename in SPLIT_FILES.items()
    }
    loaded = load_dataset("parquet", data_files=data_files)
    source_by_id = {record["sample_id"]: record for record in records}
    loaded_ids: set[str] = set()
    checked_images = 0
    checked_annotations = 0

    for split, dataset in loaded.items():
        expected = SPLIT_TARGETS[split]
        if len(dataset) != expected:
            raise ValueError(
                f"Loaded {split} split has {len(dataset)} records; expected {expected}"
            )
        if not isinstance(dataset.features["image"], DatasetImage):
            raise TypeError(f"Loaded {split} image column is not datasets.Image")
        for row in dataset:
            sample_id = row["sample_id"]
            if sample_id in loaded_ids:
                raise ValueError(f"Duplicate sample across splits: {sample_id}")
            loaded_ids.add(sample_id)
            if assignments[sample_id] != split:
                raise ValueError(f"Record loaded in the wrong split: {sample_id}")
            source = source_by_id[sample_id]
            if row["annotation"] != source["annotation_text"]:
                raise ValueError(f"Annotation changed in Parquet for {sample_id}")
            if json.loads(row["annotation"]) != source["annotation_object"]:
                raise ValueError(f"Annotation JSON did not round-trip for {sample_id}")
            if row["manifest_json"] != source["manifest_json"]:
                raise ValueError(f"Manifest record changed in Parquet for {sample_id}")
            image = row["image"]
            if not isinstance(image, Image.Image):
                raise TypeError(f"Decoded image is not a PIL image for {sample_id}")
            if image.size != (
                source["annotation_object"]["image_width"],
                source["annotation_object"]["image_height"],
            ):
                raise ValueError(f"Decoded image dimensions changed for {sample_id}")
            checked_images += 1
            checked_annotations += 1

    if loaded_ids != set(source_by_id):
        raise ValueError("The Parquet splits do not preserve all source records")
    return {
        "loaded_record_count": len(loaded_ids),
        "decoded_image_count": checked_images,
        "round_tripped_annotation_count": checked_annotations,
        "image_feature_type": str(loaded["train"].features["image"]),
    }


def prepare(source_root: Path, output_root: Path) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("Output must be separate from the original dataset")

    records = _read_source(source_root)
    groups, leakage_report = _build_scene_groups(records)
    assignments = _assign_groups(records, groups, seed=42)
    if set(assignments) != {record["sample_id"] for record in records}:
        raise ValueError("Split assignment omitted or added source records")
    for split, target in SPLIT_TARGETS.items():
        actual = sum(assignment == split for assignment in assignments.values())
        if actual != target:
            raise ValueError(f"{split} has {actual} records, expected {target}")

    group_for_id = {
        records[index]["sample_id"]: group_number
        for group_number, group in enumerate(groups, start=1)
        for index in group
    }
    group_splits: dict[int, set[str]] = defaultdict(set)
    for sample_id, split in assignments.items():
        group_splits[group_for_id[sample_id]].add(split)
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("A duplicate or near-duplicate scene crosses splits")

    output_root.mkdir(parents=True, exist_ok=True)
    records_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLIT_TARGETS
    }
    for record in records:
        records_by_split[assignments[record["sample_id"]]].append(record)

    for split, split_records in records_by_split.items():
        rows = [
            {
                "image": {
                    "path": None,
                    "bytes": record["image_path"].read_bytes(),
                },
                "annotation": record["annotation_text"],
                "sample_id": record["sample_id"],
                "language": record["manifest"]["language"],
                "scene_category": record["manifest"]["scene_category"],
                "category": record["manifest"]["category"],
                "difficulty_profile": record["manifest"]["profile"],
                "manifest_json": record["manifest_json"],
            }
            for record in split_records
        ]
        dataset = Dataset.from_list(rows, features=IMAGE_FEATURES)
        parquet_path = output_root / SPLIT_FILES[split]
        temporary_path = parquet_path.with_name(
            f".{parquet_path.name}.tmp.parquet"
        )
        dataset.to_parquet(str(temporary_path))
        temporary_path.replace(parquet_path)

    distribution = _distribution(records, assignments)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    split_manifest_path = output_root / "metadata" / "split_manifest.jsonl"
    with split_manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in sorted(records, key=lambda item: item["sample_id"]):
            line = {
                "sample_id": record["sample_id"],
                "split": assignments[record["sample_id"]],
                "scene_group": f"group-{group_for_id[record['sample_id']]:04d}",
            }
            stream.write(json.dumps(line, ensure_ascii=False) + "\n")
    _write_json(
        output_root / "metadata" / "split_distribution.json",
        {
            "seed": 42,
            "split_sizes": SPLIT_TARGETS,
            "distributions": distribution,
            "leakage_check": leakage_report,
        },
    )

    loaded_report = _validate_loaded_parquet(output_root, records, assignments)
    _copy_provenance(source_root, output_root)

    validation_report = {
        "passed": True,
        "source_record_count": len(records),
        "split_record_counts": {
            split: len(records_by_split[split]) for split in SPLIT_TARGETS
        },
        "unique_sample_id_count": len({record["sample_id"] for record in records}),
        "valid_png_count": len(records),
        "valid_annotation_count": len(records),
        "annotation_top_level_fields": sorted(
            set().union(
                *(record["annotation_object"].keys() for record in records)
            )
        ),
        "text_region_field_union": sorted(
            set().union(
                *(
                    region.keys()
                    for record in records
                    for region in record["annotation_object"]["text_regions"]
                )
            )
        ),
        "text_region_count_distribution": dict(
            sorted(
                Counter(
                    len(record["annotation_object"]["text_regions"])
                    for record in records
                ).items()
            )
        ),
        "all_regions_have_bbox_and_polygon": True,
        "exact_image_duplicate_count": leakage_report[
            "exact_duplicate_image_count"
        ],
        "near_duplicate_pair_count": leakage_report[
            "near_duplicate_pair_count"
        ],
        "duplicate_groups_crossing_splits": 0,
        "split_distributions": distribution,
        "local_load_dataset_parquet_test": loaded_report,
        "source_data_modified": False,
    }
    _write_json(
        output_root / "validation_reports" / "viewer_ready_validation.json",
        validation_report,
    )
    print(
        json.dumps(
            {
                "output": str(output_root),
                "parquet_files": SPLIT_FILES,
                "validation": validation_report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Build and validate Viewer-ready Parquet splits."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project_root / "hf_upload" / "indic-synthetic-scene-text-1000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    arguments = parser.parse_args()
    prepare(arguments.source_root, arguments.output_dir)


if __name__ == "__main__":
    main()
