"""Translation and Summarization service powered by local 4-bit Quantized Qwen3.5-2B (GGUF)."""
from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default model repository and GGUF filename (4-bit quantized ~1.3 GB)
DEFAULT_GGUF_REPO = "bartowski/Qwen_Qwen3.5-2B-GGUF"
DEFAULT_GGUF_FILENAME = "Qwen_Qwen3.5-2B-Q4_K_M.gguf"
DEFAULT_LLM_TIMEOUT_S = 120.0


class LLMTimeoutError(RuntimeError):
    """Raised when a local LLM call does not finish within its time budget."""


class IndicTranslationService:
    """Pass-through service for translation (since Whisper task='translate' produces English),
    and fast GPU/CPU-based meeting summarizer + action item extractor powered by Qwen3.5-2B (4-bit Q4_K_M GGUF).
    """

    def __init__(self, repo_id: str = DEFAULT_GGUF_REPO, filename: str = DEFAULT_GGUF_FILENAME):
        self.repo_id = repo_id
        self.filename = filename
        self._llm = None
        self._is_initialized = False
        # Single worker: every call (model load + every generation, whether it
        # comes from the synchronous pipeline or a background translation job
        # thread) is queued onto this one thread. That serializes access to the
        # shared llama.cpp context (not safe for concurrent generation calls)
        # without a separate lock, and gives each call a hard timeout via
        # future.result(timeout=...) instead of letting a caller hang forever.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-llm")

    def _run(self, fn, timeout_s: float):
        future = self._executor.submit(fn)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as exc:
            raise LLMTimeoutError(f"Local LLM call did not finish within {timeout_s:.0f}s") from exc

    def _ensure_model_loaded(self) -> None:
        """Lazy load Qwen3.5-2B 4-bit GGUF via llama-cpp-python. Only ever called
        from within the single executor thread, so no locking is needed."""
        if self._is_initialized and self._llm is not None:
            return

        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        logger.info("⏳ [SUMMARIZER INIT] Locating / downloading '%s' (%s)...", self.repo_id, self.filename)
        t0 = time.time()
        model_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)

        # Offload layers to GPU if CUDA available; falls back to CPU cleanly
        logger.info("⏳ [SUMMARIZER INIT] Loading GGUF into llama-cpp from %s...", model_path)
        self._llm = Llama(
            model_path=model_path,
            n_ctx=32768,  # Full 32K context window
            n_gpu_layers=-1,  # Offload all layers to GPU
            verbose=False,
        )
        self._is_initialized = True
        logger.info("✅ [SUMMARIZER READY] Loaded Qwen3.5-2B 4-bit GGUF (32k context) in %.2fs!", time.time() - t0)

    def generate_raw_completion(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.2,
        timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
    ) -> str:
        """Runs chat inference with local 4-bit quantized model and returns clean decoded text response.

        Raises LLMTimeoutError if the model doesn't respond within timeout_s.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional meeting translation and localization assistant. "
                    "Follow all instructions, output format tags, line numbering, and speaker guidelines strictly. "
                    "Do NOT output thinking steps, reasoning tags, or conversational chatter."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        def _call():
            self._ensure_model_loaded()
            return self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                repeat_penalty=1.05,
            )

        response = self._run(_call, timeout_s)

        decoded = response["choices"][0]["message"]["content"] or ""
        decoded = decoded.strip()

        # Strip any internal reasoning or thinking tags emitted by reasoning models
        if "</think>" in decoded:
            decoded = decoded.split("</think>", 1)[1].strip()
        elif "<think>" in decoded:
            decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()

        return decoded

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

    def summarize_meeting(
        self,
        dialogue_lines: List[str],
        timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
    ) -> Dict[str, Any]:
        """Generate structured meeting summary with key discussion points and action items."""
        if not dialogue_lines:
            return {"summary_markdown": ""}

        try:
            transcript_text = "\n".join(dialogue_lines)
            if len(transcript_text) > 16000:
                transcript_text = transcript_text[:16000] + "\n...[transcript truncated]..."

            system_prompt = (
                "You are an expert meeting assistant. Analyze the provided meeting transcript and create a concise, "
                "well-structured summary in Markdown format with the following sections:\n"
                "### 1. Executive Summary\n"
                "A brief 2-3 sentence overview of what was discussed.\n\n"
                "### 2. Key Discussion Points & Decisions\n"
                "- Bullet points of key decisions made and important updates.\n\n"
                "### 3. Action Items\n"
                "- [ ] **[Owner/Speaker]**: Action task description (mention any deadlines/tools discussed).\n"
                "IMPORTANT: If no specific action items, tasks, or follow-ups were assigned to anyone in the meeting, write exactly:\n"
                "_No action items found in this meeting._\n\n"
                "Keep it clear, professional, and factual based strictly on the transcript."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the meeting transcript:\n\n{transcript_text}\n\nPlease generate the summary:"},
            ]

            def _call():
                self._ensure_model_loaded()
                return self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.3,
                    top_p=0.9,
                    repeat_penalty=1.1,
                )

            response = self._run(_call, timeout_s)

            summary_markdown = response["choices"][0]["message"]["content"] or ""
            summary_markdown = summary_markdown.strip()

            # Clean reasoning tags if any
            if "</think>" in summary_markdown:
                summary_markdown = summary_markdown.split("</think>", 1)[1].strip()

            # Clean markdown formatting if wrapped in code blocks
            if summary_markdown.startswith("```markdown"):
                summary_markdown = summary_markdown[len("```markdown"):].strip()
            if summary_markdown.startswith("```"):
                summary_markdown = summary_markdown[len("```"):].strip()
            if summary_markdown.endswith("```"):
                summary_markdown = summary_markdown[:-3].strip()

            # Post-check: ensure Action Items section doesn't stay empty
            if "Action Items" in summary_markdown:
                action_part = summary_markdown.split("Action Items", 1)[1].strip()
                action_content = action_part.lstrip("# \t\r\n")
                if not action_content or action_content in ("[]", "[ ]", "None", "- None", "- N/A", "N/A"):
                    summary_markdown = summary_markdown.split("Action Items", 1)[0].rstrip() + "\n\n### 3. Action Items\n_No action items found in this meeting._"

            return {"summary_markdown": summary_markdown}
        except Exception as exc:
            logger.error("Failed to generate meeting summary: %s", exc, exc_info=True)
            return {"summary_markdown": "", "error": str(exc)}


indic_translation_service = IndicTranslationService()
