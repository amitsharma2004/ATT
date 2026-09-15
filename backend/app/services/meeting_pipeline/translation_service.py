"""Translation and Summarization service powered by local Llama-3.1-8B-Instruct GGUF."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IndicTranslationService:
    """Pass-through service since Whisper task='translate' produces English text directly."""

    def __init__(self):
        pass

    def translate_line(self, text: str, src_lang: Optional[str] = None) -> str:
        """Returns text directly (Whisper already translates to English)."""
        return (text or "").strip()

    def translate_dialogue_lines(self, dialogue: List[dict]) -> List[dict]:
        """Pass-through dialogue with translated_text set to original text."""
        results = []
        for line in dialogue:
            spk = line.get("speaker", "SPEAKER")
            text = line.get("text", "")
            new_line = dict(line)
            new_line["translated_text"] = text
            new_line["formatted_line"] = f"{spk}: {text}"
            results.append(new_line)
        return results

    def summarize_meeting(self, dialogue_lines: List[str]) -> Dict[str, Any]:
        """Summarization placeholder when LLM is disabled."""
        return {"summary_markdown": ""}


indic_translation_service = IndicTranslationService()
