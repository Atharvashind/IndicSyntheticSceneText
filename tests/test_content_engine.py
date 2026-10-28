import json

import pytest

from src.content_engine import (
    CATEGORY_NAMES,
    SUPPORTED_LANGUAGES,
    generate_text,
    get_content_types,
)


@pytest.mark.parametrize("category", CATEGORY_NAMES)
@pytest.mark.parametrize("language", SUPPORTED_LANGUAGES)
def test_configured_content_is_nonempty_and_reproducible(category, language):
    content_type = get_content_types(category)[0]
    first = generate_text(language, category, seed=27, content_type=content_type)
    second = generate_text(language, category, seed=27, content_type=content_type)
    assert first == second
    assert first.strip()


def test_custom_text_is_not_translated_or_transliterated():
    custom_text = "দোকানের নিজস্ব লেখা"
    assert generate_text(
        "bengali",
        "shop_board",
        custom_text=custom_text,
        content_type="shop_name",
    ) == custom_text


def test_content_templates_can_be_configured(tmp_path):
    config = {
        "content_templates": {
            "shop_board": {
                "shop_name": {
                    "marathi": ["नवीन दुकान"],
                    "hindi": ["नई दुकान"],
                    "gujarati": ["નવી દુકાન"],
                    "bengali": ["নতুন দোকান"],
                }
            }
        }
    }
    config_path = tmp_path / "custom_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    assert generate_text(
        "gujarati",
        "shop_board",
        seed=9,
        content_type="shop_name",
        config_path=config_path,
    ) == "નવી દુકાન"


def test_unsupported_category_language_and_empty_custom_text_fail():
    with pytest.raises(ValueError):
        generate_text("tamil", "shop_board")
    with pytest.raises(ValueError):
        generate_text("hindi", "unknown_category")
    with pytest.raises(ValueError):
        generate_text("hindi", "shop_board", custom_text="  ")
