from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from .content_engine import SUPPORTED_LANGUAGES, load_config
from .text_renderer import LANGUAGE_SHAPING, _shape_line, _validate_font_coverage, render_multiline_text

ROOT_DIR = Path(__file__).resolve().parents[1]
FONT_TEST_STRINGS = {
    "marathi": (
        "नमस्कार, मराठी!",
        "कि, की, क्रांती",
        "क्षेत्र, त्रिकोण, ज्ञान",
        "०१२३४५६७८९ ₹",
    ),
    "hindi": (
        "नमस्ते, भारत!",
        "कि, की, कृपया",
        "क्षेत्र, त्रिकोण, प्रज्ञा",
        "०१२३४५६७८९ ₹",
    ),
    "gujarati": (
        "નમસ્તે, ગુજરાતી!",
        "કિંમત, કીર્તિ",
        "ક્ષેત્ર, જ્ઞાન, શ્રદ્ધા",
        "૦૧૨૩૪૫૬૭૮૯ ₹",
    ),
    "bengali": (
        "নমস্কার, বাংলা!",
        "কি, কী, কৃষি",
        "ক্ষেত্র, জ্ঞান, শ্রদ্ধা",
        "০১২৩৪৫৬৭৮৯ ₹",
    ),
}
CONJUNCT_PROBES = {
    "marathi": "क्षेत्र",
    "hindi": "क्षेत्र",
    "gujarati": "ક્ષેત્ર",
    "bengali": "ক্ষেত্র",
}


def _read_font_metadata(path: Path) -> dict:
    font = TTFont(path, lazy=True)
    try:
        names = font["name"]
        family = names.getDebugName(1)
        style = names.getDebugName(2)
        full_name = names.getDebugName(4)
        cmap = font.getBestCmap()
        weight = font["OS/2"].usWeightClass
        glyph_count = len(font.getGlyphOrder())
        if not family or not style:
            raise ValueError(f"Font '{path}' is missing family or style metadata.")
        return {
            "family": family,
            "style": style,
            "full_name": full_name,
            "weight": weight,
            "glyph_count": glyph_count,
            "cmap": cmap,
        }
    finally:
        font.close()


def _font_candidate(configured: dict) -> dict:
    relative_path = Path(configured["path"])
    if relative_path.is_absolute():
        raise ValueError(f"Font registry paths must be project-relative: {relative_path}")
    path = (ROOT_DIR / relative_path).resolve()
    try:
        path.relative_to(ROOT_DIR.resolve())
    except ValueError as error:
        raise ValueError(f"Font registry path escapes the project: {relative_path}") from error
    return {
        **configured,
        "path": path,
        "relative_path": relative_path.as_posix(),
    }


def _feature_results(language: str, lines: Tuple[str, ...]) -> Dict[str, bool]:
    flattened = "".join(lines)
    has_marks = any(unicodedata.category(char) in {"Mn", "Mc", "Me"} for char in flattened)
    has_punctuation = any(unicodedata.category(char).startswith("P") for char in flattened)
    has_numerals = any(unicodedata.category(char) == "Nd" for char in flattened)
    return {
        "vowel_signs": has_marks,
        "combining_marks": has_marks,
        "conjuncts": CONJUNCT_PROBES[language] in flattened,
        "punctuation": has_punctuation,
        "numerals": has_numerals,
        "multiline": True,
    }


def _validate_candidate_language(
    candidate: dict,
    language: str,
    output_dir: Path,
) -> Tuple[dict, Optional[Path]]:
    path = candidate["path"]
    text = "\n".join(FONT_TEST_STRINGS[language])
    script, language_tag = LANGUAGE_SHAPING[language]
    result = {
        "language": language,
        "script": script,
        "shaping_language": language_tag,
        "exact_transcription_preserved": False,
        "missing_codepoints": [],
        "missing_glyph_box_detected": False,
        "clipping_check_passed": False,
        "feature_probes": _feature_results(language, FONT_TEST_STRINGS[language]),
        "passed": False,
        "error": None,
    }
    try:
        _validate_font_coverage(text, str(path))
        font_data = path.read_bytes()
        for line in FONT_TEST_STRINGS[language]:
            glyph_ids, _, _ = _shape_line(line, language, font_data, 52)
            if not glyph_ids or any(glyph_id == 0 for glyph_id in glyph_ids):
                result["missing_glyph_box_detected"] = True
                raise ValueError(f"Missing-glyph box detected while shaping {line!r}.")
        rendered = render_multiline_text(
            text,
            language,
            width=1600,
            height=460,
            font_size=52,
            foreground=(22, 38, 72),
            background=(250, 249, 244),
            padding=20,
            line_spacing=1.15,
            align="left",
            shadow_offset=(0, 0),
            font_path=path,
        )
        result["exact_transcription_preserved"] = rendered["text"] == text
        if not result["exact_transcription_preserved"]:
            raise ValueError("Renderer changed the supplied Unicode transcription.")
        ink = rendered["ink_bbox"]
        result["clipping_check_passed"] = (
            0 <= ink["x_min"] < ink["x_max"] <= rendered["width"]
            and 0 <= ink["y_min"] < ink["y_max"] <= rendered["height"]
        )
        if not result["clipping_check_passed"]:
            raise ValueError("Rendered ink bounds lie outside the diagnostic canvas.")
        image_path = output_dir / "images" / f"{candidate['id']}__{language}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        rendered["image"].save(image_path, format="PNG")
        result["ink_bbox"] = ink
        result["image"] = f"images/{image_path.name}"
        result["passed"] = all(result["feature_probes"].values())
        return result, image_path
    except (OSError, ValueError, KeyError) as error:
        result["error"] = str(error)
        if "missing glyph" in str(error).lower():
            result["missing_glyph_box_detected"] = True
        if "does not cover required characters" in str(error):
            result["missing_codepoints"] = [
                value.strip()
                for value in str(error).partition(":")[2].split(",")
                if value.strip()
            ]
        return result, None


def _write_contact_sheet(samples: List[Tuple[Path, str]], destination: Path) -> None:
    if not samples:
        raise ValueError("No candidate font passed rendering; a diagnostic sheet cannot be created.")
    columns = 2
    cell_width, cell_height = 820, 250
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    caption_font = ImageFont.load_default()
    for index, (image_path, caption) in enumerate(samples):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((cell_width - 20, cell_height - 34), Image.Resampling.LANCZOS)
        x = index % columns * cell_width
        y = index // columns * cell_height
        sheet.paste(image, (x + 8, y + 4))
        draw.text(
            (x + 8, y + cell_height - 23),
            caption,
            font=caption_font,
            fill=(20, 20, 20),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG")


def validate_font_registry(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing font diagnostic directory: {output_dir}")
    configured = load_config().get("font_registry", {}).get("fonts")
    if not isinstance(configured, list) or not configured:
        raise ValueError("No font candidates are configured in font_registry.fonts.")

    candidates = [_font_candidate(entry) for entry in configured]
    if len({candidate["id"] for candidate in candidates}) != len(candidates):
        raise ValueError("Font registry IDs must be unique.")

    output_dir.mkdir(parents=True)
    candidate_results = []
    successful_images = []
    for candidate in candidates:
        path = candidate["path"]
        record = {
            "id": candidate["id"],
            "path": candidate["relative_path"],
            "family": None,
            "weight": None,
            "style": None,
            "license": candidate.get("license", "Unknown"),
            "license_status": candidate.get("license_status", "unknown"),
            "license_file": candidate.get("license_file"),
            "provenance": candidate.get("provenance", "Not recorded."),
            "target_languages": candidate.get("target_languages", []),
            "supported_scripts": [],
            "language_validation": [],
            "eligible_for_generation": False,
            "font_file_exists": path.is_file(),
            "error": None,
        }
        try:
            if not path.is_file():
                raise FileNotFoundError(f"Font file does not exist: {path}")
            metadata = _read_font_metadata(path)
            record.update(
                {
                    "family": metadata["family"],
                    "weight": metadata["weight"],
                    "style": metadata["style"],
                    "glyph_count": metadata["glyph_count"],
                }
            )
            license_file = candidate.get("license_file")
            if candidate.get("license_status") == "verified":
                if not license_file:
                    raise ValueError("Verified license status requires a local license file.")
                license_path = (ROOT_DIR / license_file).resolve()
                if not license_path.is_file():
                    raise FileNotFoundError(f"Recorded font license file is missing: {license_file}")
                license_text = license_path.read_text(encoding="utf-8")
                if "SIL OPEN FONT LICENSE" not in license_text or "Version 1.1" not in license_text:
                    raise ValueError(f"License text at {license_file} is not the expected OFL 1.1.")
            cmap = metadata["cmap"]
            record["unicode_script_coverage"] = {
                script: sum(codepoint in cmap for codepoint in range(start, end + 1))
                for script, (start, end) in {
                    "Devanagari": (0x0900, 0x097F),
                    "Gujarati": (0x0A80, 0x0AFF),
                    "Bengali": (0x0980, 0x09FF),
                }.items()
            }
            for language in candidate["target_languages"]:
                if language not in SUPPORTED_LANGUAGES:
                    raise ValueError(f"Unsupported font target language: {language!r}.")
                language_result, image_path = _validate_candidate_language(
                    candidate,
                    language,
                    output_dir,
                )
                record["language_validation"].append(language_result)
                if language_result["passed"]:
                    script_name = {
                        "Deva": "Devanagari",
                        "Gujr": "Gujarati",
                        "Beng": "Bengali",
                    }[language_result["script"]]
                    if script_name not in record["supported_scripts"]:
                        record["supported_scripts"].append(script_name)
                    if image_path is not None:
                        successful_images.append((image_path, f"{candidate['id']} | {language}"))
            languages_passed = (
                bool(record["target_languages"])
                and len(record["language_validation"]) == len(record["target_languages"])
                and all(item["passed"] for item in record["language_validation"])
            )
            license_verified = candidate.get("license_status") == "verified"
            record["eligible_for_generation"] = languages_passed and license_verified
            if not languages_passed:
                record["error"] = "One or more configured language/script rendering checks failed."
        except (OSError, ValueError, KeyError) as error:
            record["error"] = str(error)
        candidate_results.append(record)

    if not successful_images:
        raise ValueError("No font/script combination passed validation; no contact sheet was written.")
    contact_sheet_path = output_dir / "font_diagnostic_contact_sheet.png"
    _write_contact_sheet(successful_images, contact_sheet_path)
    report = {
        "schema_version": "1.0",
        "rendering_engine": "HarfBuzz + FreeType",
        "font_candidate_count": len(candidate_results),
        "eligible_font_ids": [
            item["id"] for item in candidate_results if item["eligible_for_generation"]
        ],
        "passed_font_count": sum(item["eligible_for_generation"] for item in candidate_results),
        "failed_or_unlicensed_font_ids": [
            item["id"] for item in candidate_results if not item["eligible_for_generation"]
        ],
        "fonts": candidate_results,
        "diagnostic_contact_sheet": contact_sheet_path.name,
    }
    report_path = output_dir / "font_validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
