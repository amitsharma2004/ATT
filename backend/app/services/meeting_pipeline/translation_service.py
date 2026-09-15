"""Indic-to-English translation service powered by facebook/nllb-200-distilled-600M."""
import re
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from backend.app.core.config import get_settings

MODEL_NAME = "facebook/nllb-200-distilled-600M"
TARGET_LANG = "eng_Latn"

# NLLB language codes for common Indian languages
# Reference: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
NLLB_LANG_MAP = {
    "ta": "tam_Taml",  # Tamil
    "hi": "hin_Deva",  # Hindi
    "te": "tel_Telu",  # Telugu
    "kn": "kan_Knda",  # Kannada
    "ml": "mal_Mlym",  # Malayalam
    "mr": "mar_Deva",  # Marathi
    "bn": "ben_Beng",  # Bengali
    "gu": "guj_Gujr",  # Gujarati
    "pa": "pan_Guru",  # Punjabi
    "en": "eng_Latn",  # English
}

def detect_indic_script(text: str) -> str:
    """Detect script and map to Flores-200 / NLLB language tag."""
    # Tamil Unicode block: \u0b80-\u0bff
    if re.search(r'[\u0B80-\u0BFF]', text):
        return "tam_Taml"
    # Devanagari (Hindi/Marathi) Unicode block: \u0900-\u097f
    if re.search(r'[\u0900-\u097F]', text):
        return "hin_Deva"
    # Telugu: \u0c00-\u0c7f
    if re.search(r'[\u0C00-\u0C7F]', text):
        return "tel_Telu"
    # Kannada: \u0c80-\u0cff
    if re.search(r'[\u0C80-\u0CFF]', text):
        return "kan_Knda"
    # Malayalam: \u0d00-\u0d7f
    if re.search(r'[\u0D00-\u0D7F]', text):
        return "mal_Mlym"
    # Bengali: \u0980-\u09ff
    if re.search(r'[\u0980-\u09FF]', text):
        return "ben_Beng"
    # Gujarati: \u0a80-\u0aff
    if re.search(r'[\u0A80-\u0AFF]', text):
        return "guj_Gujr"
    # Default fallback for Indian speech context
    return "tam_Taml"


class IndicTranslationService:
    def __init__(self):
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        token = get_settings().effective_hf_token.get_secret_value() if get_settings().effective_hf_token else None
        print(f"🚀 Loading NLLB Translation Model ({MODEL_NAME})...", flush=True)
        from pathlib import Path
        snapshot_dir = Path.home() / ".cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d"
        if snapshot_dir.exists():
            self._tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir), local_files_only=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(str(snapshot_dir), local_files_only=True)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, token=token)
        print("✅ NLLB-200 Translation Model loaded successfully!", flush=True)

    def translate_line(self, text: str, src_lang: Optional[str] = None) -> str:
        """Translate a single line to English using NLLB-200."""
        clean_txt = text.strip()
        if not clean_txt:
            return ""

        # If text is almost entirely ASCII (English), return as-is
        non_ascii_chars = len([c for c in clean_txt if ord(c) > 127])
        if non_ascii_chars == 0:
            return clean_txt

        self._ensure_loaded()
        
        # Resolve source language tag
        if src_lang and src_lang in NLLB_LANG_MAP:
            lang_tag = NLLB_LANG_MAP[src_lang]
        elif src_lang and "_" in src_lang:
            lang_tag = src_lang
        else:
            lang_tag = detect_indic_script(clean_txt)

        self._tokenizer.src_lang = lang_tag
        inputs = self._tokenizer(clean_txt, return_tensors="pt", truncation=True, max_length=512)
        
        target_lang_id = self._tokenizer.convert_tokens_to_ids(TARGET_LANG)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded[0].strip() if decoded else clean_txt

    def translate_dialogue_lines(self, dialogue: List[dict]) -> List[dict]:
        """
        Takes a list of dialogue dicts:
        [{"speaker": "SPEAKER_00", "start": 0.0, "end": 8.2, "text": "...", "translated_text": ""}, ...]
        Translates each line to English and adds 'translated_text'.
        """
        results = []
        for line in dialogue:
            spk = line.get("speaker", "SPEAKER")
            orig_text = line.get("text", "")
            eng_text = self.translate_line(orig_text)
            new_line = dict(line)
            new_line["translated_text"] = eng_text
            new_line["formatted_line"] = f"{spk}: {eng_text}"
            results.append(new_line)
        return results


indic_translation_service = IndicTranslationService()
