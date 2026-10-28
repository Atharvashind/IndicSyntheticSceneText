from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "configs" / "generation_config.json"

SUPPORTED_LANGUAGES = ("marathi", "hindi", "gujarati", "bengali")
CATEGORY_NAMES = (
    "shop_board",
    "advertising_hoarding",
    "election_banner",
    "birthday_event_banner",
)


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_supported_languages() -> List[str]:
    return list(SUPPORTED_LANGUAGES)


def get_category_names() -> List[str]:
    return list(CATEGORY_NAMES)


def get_content_types(category: str, config_path: Path = CONFIG_PATH) -> List[str]:
    templates = load_config(config_path).get("content_templates", {})
    category_templates = templates.get(category)
    if not isinstance(category_templates, dict) or not category_templates:
        raise ValueError(f"No content templates are configured for category '{category}'.")
    return list(category_templates)


def ensure_valid_text(value: Optional[str], *, language: str) -> str:
    if value is None:
        raise ValueError(f"Text generation for {language} returned None.")
    text = str(value)
    if not text.strip():
        raise ValueError(f"Text generation for {language} produced an empty value.")
    if any(ord(char) < 32 and char not in {"\n", "\t"} for char in text):
        raise ValueError(f"Text generation for {language} included unsupported control characters.")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        raise ValueError(f"Text generation for {language} included an invalid Unicode surrogate.")
    return text


def generate_text(
    language: str,
    category: str,
    seed: int = 0,
    custom_text: Optional[str] = None,
    content_type: Optional[str] = None,
    config_path: Path = CONFIG_PATH,
) -> str:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language '{language}'. Supported languages: {SUPPORTED_LANGUAGES}")
    if category not in CATEGORY_NAMES:
        raise ValueError(f"Unsupported category '{category}'. Supported categories: {CATEGORY_NAMES}")
    if custom_text is not None:
        return ensure_valid_text(custom_text, language=language)

    config = load_config(config_path)
    category_templates = config.get("content_templates", {}).get(category)
    if not isinstance(category_templates, dict) or not category_templates:
        raise ValueError(f"No content templates are configured for category '{category}'.")
    if content_type is None:
        content_type = next(iter(category_templates))
    if content_type not in category_templates:
        raise ValueError(
            f"Content type '{content_type}' is not configured for category '{category}'. "
            f"Available types: {', '.join(category_templates)}"
        )

    language_templates = category_templates[content_type]
    if not isinstance(language_templates, dict) or language not in language_templates:
        raise ValueError(f"No {language} templates are configured for {category}.{content_type}.")
    choices = language_templates[language]
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"The configured list for {category}.{content_type}.{language} must not be empty.")
    for choice in choices:
        ensure_valid_text(choice, language=language)

    return ensure_valid_text(random.Random(seed).choice(choices), language=language)


def _legacy_content_templates() -> Dict[str, Dict[str, List[str]]]:
    config_templates = load_config().get("content_templates", {})
    legacy: Dict[str, Dict[str, List[str]]] = {}
    for category in CATEGORY_NAMES:
        category_templates = config_templates.get(category, {})
        legacy[category] = {}
        for language in SUPPORTED_LANGUAGES:
            legacy[category][language] = [
                text
                for content_type in category_templates.values()
                for text in content_type.get(language, [])
            ]
    return legacy


CONTENT_TEMPLATES = _legacy_content_templates()
