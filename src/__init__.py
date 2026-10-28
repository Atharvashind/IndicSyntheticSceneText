"""IndicSyntheticSceneText project package."""

from .content_engine import CONTENT_TEMPLATES, generate_text, get_supported_languages
from .text_renderer import render_multiline_text, render_language_diagnostics

__all__ = [
    "CONTENT_TEMPLATES",
    "generate_text",
    "get_supported_languages",
    "render_multiline_text",
    "render_language_diagnostics",
]
