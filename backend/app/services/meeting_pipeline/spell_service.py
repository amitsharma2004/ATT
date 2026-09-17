"""Spell checking and domain-term fuzzy correction service."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple
from spellchecker import SpellChecker
from rapidfuzz import fuzz, process

# Company domain terms & technical entities that may be misheard by Whisper
COMPANY_GLOSSARY = [
    "Jira",
    "Edit Central",
    "Proof Central",
    "Graphic Central",
    "Pulse Matrix",
    "Chaitanya",
    "Subu",
    "Satya",
    "Senthil",
    "Saif",
    "Amit",
    "Page Central",
    "ReportingDB",
    "ApplicationDB",
    "TNQ",
    "Nimble",
    "Docker",
    "FastAPI",
    "Whisper",
    "Pyannote",
    "PostgreSQL",
    "MongoDB",
]


class SpellCheckService:
    """Combines domain glossary fuzzy matching with general English dictionary spell checking."""

    def __init__(self, glossary: Optional[List[str]] = None):
        self.glossary = glossary or COMPANY_GLOSSARY
        self.spell = SpellChecker()

        # Whitelist all glossary words and subwords so they aren't marked as errors
        for term in self.glossary:
            for word in re.findall(r"\w+", term):
                self.spell.word_frequency.add(word.lower())

        # Whitelist tech acronyms & common IT words
        common_tech = [
            "api", "db", "dbms", "rdbms", "sql", "nosql", "crud", "sdk", "ui", "ux",
            "frontend", "backend", "auth", "dev", "prod", "env", "repo", "pr", "ci", "cd",
            "aws", "gcp", "azure", "k8s", "url", "uri", "http", "https", "json", "xml",
            "uuid", "guid", "fifo", "lifo", "poc", "sla", "etl", "dto", "orm", "io",
            "grpc", "oauth", "oauth2", "s3", "ingress", "telemetry", "devops", "synch",
            "say", "says", "said", "saying", "tell", "told", "ask", "asked", "pull",
            "pulse", "matrix", "proof", "central", "page", "report", "sub",
            "aren't", "can't", "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't",
            "haven't", "isn't", "mightn't", "mustn't", "needn't", "shan't", "shouldn't",
            "wasn't", "weren't", "won't", "wouldn't", "it's", "that's", "there's", "what's", "let's"
        ]
        for w in common_tech:
            self.spell.word_frequency.add(w)

    def check_text(self, text: str) -> Dict[str, any]:
        """Analyze text and detect misspelled words along with suggested corrections.

        Returns:
            dict containing:
                - has_errors (bool)
                - errors (list of {original, suggestion, reason, is_glossary_match})
                - corrected_text (str)
        """
        if not text or not text.strip():
            return {"has_errors": False, "errors": [], "corrected_text": text}

        # Match words including English contractions (e.g. aren't, don't, subu's)
        word_tokens = [w for w in re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", text)]

        errors: List[Dict[str, any]] = []
        corrections_map: Dict[str, str] = {}

        # 1. Check Multi-Word Company Glossary Terms first (e.g. 'Edit Centrl' -> 'Edit Central')
        for term in self.glossary:
            term_len = len(term.split())
            if term_len > 1:
                # Sliding window over words
                words = text.split()
                for i in range(len(words) - term_len + 1):
                    window = " ".join(words[i:i + term_len])
                    clean_window = re.sub(r"[^\w\s]", "", window).strip()
                    if clean_window.lower() == term.lower():
                        continue
                    score = fuzz.ratio(clean_window.lower(), term.lower())
                    if score >= 85 and abs(len(clean_window) - len(term)) <= 2:
                        if window not in corrections_map:
                            corrections_map[window] = term
                            errors.append({
                                "original": window,
                                "suggestion": term,
                                "reason": f"Company multi-word match ({score:.0f}%)",
                                "is_glossary": True,
                            })

        # 2. Fuzzy check Single-Word Company Glossary Terms
        # E.g., 'Gira' -> 'Jira', 'Subhu' -> 'Subu'
        for term in self.glossary:
            term_len = len(term.split())
            if term_len == 1:
                term_clean = term.lower()
                for w in word_tokens:
                    w_clean = w.lower()
                    if w_clean == term_clean:
                        continue

                    # PROTECTION 1: If word is already a valid dictionary English word (e.g. 'say', 'sub', 'proof'),
                    # do NOT replace it with a company name unless exact match.
                    if w_clean in self.spell and len(w_clean) <= 4:
                        continue

                    # PROTECTION 2: Skip ALL-CAPS abbreviations/acronyms (e.g. DBMS, SQL, AWS, PR, ID)
                    if w.isupper() and len(w) >= 2:
                        continue

                    # Calculate fuzzy ratio
                    score = fuzz.ratio(w_clean, term_clean)
                    len_diff = abs(len(w) - len(term))
                    is_match = False
                    if len(term) <= 4:
                        # e.g., 'Gira' (4) vs 'Jira' (4) is score 75%, 1 character substitution difference
                        if len_diff == 0 and score >= 75:
                            is_match = True
                        elif score >= 85 and len_diff <= 1:
                            is_match = True
                    else:
                        if score >= 80 and len_diff <= 2:
                            is_match = True

                    if is_match and w not in corrections_map:
                        corrections_map[w] = term
                        errors.append({
                            "original": w,
                            "suggestion": term,
                            "reason": f"Company term match ({score:.0f}%)",
                            "is_glossary": True,
                        })

        # 2. General spellcheck for remaining unknown words
        for w in word_tokens:
            if w in corrections_map or w.isnumeric() or len(w) <= 2:
                continue

            # PROTECTION 2: Never spell-check ALL-CAPS acronyms/jargons (e.g. DBMS, SQL, API)
            if w.isupper() and len(w) >= 2:
                continue

            cleaned_w = w.lower()
            if cleaned_w not in self.spell:
                # Check if it's misspelled
                cand = self.spell.candidates(cleaned_w)
                if cand:
                    best_cand = self.spell.correction(cleaned_w)
                    if best_cand and best_cand != cleaned_w:
                        # Do not change if the suggestion has a totally different length on short words
                        if len(cleaned_w) <= 3 and abs(len(best_cand) - len(cleaned_w)) > 1:
                            continue

                        # Preserve original casing if capitalized
                        if w[0].isupper():
                            best_cand = best_cand.capitalize()

                        corrections_map[w] = best_cand
                        errors.append({
                            "original": w,
                            "suggestion": best_cand,
                            "reason": "Spelling dictionary error",
                            "is_glossary": False,
                        })

        # Generate corrected text
        corrected_text = text
        for orig, sugg in corrections_map.items():
            # Replace whole word only
            corrected_text = re.sub(rf"\b{re.escape(orig)}\b", sugg, corrected_text)

        return {
            "has_errors": len(errors) > 0,
            "errors": errors,
            "corrected_text": corrected_text,
        }

    def check_segments(self, segments: List[Dict[str, any]]) -> Dict[str, any]:
        """Check all segments in a transcript and return highlighted & corrected segments."""
        checked_segments = []
        total_errors = 0

        for seg in segments:
            text = seg.get("text", "")
            res = self.check_text(text)
            seg_copy = dict(seg)
            seg_copy["has_errors"] = res["has_errors"]
            seg_copy["errors"] = res["errors"]
            seg_copy["suggested_text"] = res["corrected_text"]
            if res["has_errors"]:
                total_errors += len(res["errors"])
            checked_segments.append(seg_copy)

        return {
            "total_errors": total_errors,
            "segments": checked_segments,
        }


spell_check_service = SpellCheckService()
