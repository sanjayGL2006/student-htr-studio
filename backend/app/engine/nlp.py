"""
Two-stage text correction, applied to each TrOCR line output before it is
stored as `corrected_text`.

Stage 1 (fast, deterministic): SymSpell dictionary lookup fixes common
OCR/handwriting-recognition glitches at the token level (rn -> m, cl -> d,
l -> 1, etc.) without needing model inference.

Stage 2 (contextual): a T5 grammar-correction model fixes tense, prepositions,
and syntax using the "grammar: {text}" prompt prefix. This catches errors
Stage 1 can't, e.g. a correctly-spelled but grammatically wrong word choice.

Both `raw_text` and `corrected_text` are always kept (see models.Node) so a
teacher can revert an over-correction, e.g. a proper noun the grammar model
"fixed" incorrectly.
"""

from __future__ import annotations

import os

import torch
from symspellpy import SymSpell, Verbosity
from transformers import AutoTokenizer, T5ForConditionalGeneration

from app.config import settings

# Common single/multi-character substitution errors specific to handwritten
# OCR (distinct from typed-text typos, which is what most spellcheck
# dictionaries are tuned for).
_HANDWRITING_GLYPH_FIXES = {
    "rn": "m",
    "cl": "d",
    "vv": "w",
    "ii": "ll",
}


def _build_symspell() -> SymSpell:
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    # symspellpy ships this frequency dictionary as package data, so no
    # separate download/network access is required.
    import symspellpy

    dictionary_path = os.path.join(
        os.path.dirname(symspellpy.__file__), "frequency_dictionary_en_82_765.txt"
    )
    if os.path.exists(dictionary_path):
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    # If the dictionary file isn't present, SymSpell just won't find lookup
    # matches and Stage 1 becomes a no-op — Stage 2 (grammar model) still runs.
    return sym_spell


def lexical_pass(text: str, sym_spell: SymSpell) -> str:
    """Token-level correction: fix known handwriting glyph confusions, then
    run each word through SymSpell for a single best-guess correction."""
    words = text.split()
    fixed_words = []
    for word in words:
        candidate = word
        lower = word.lower()
        for glyph, fix in _HANDWRITING_GLYPH_FIXES.items():
            if glyph in lower:
                candidate = candidate.replace(glyph, fix)

        suggestions = sym_spell.lookup(candidate, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            fixed_words.append(suggestions[0].term)
        else:
            fixed_words.append(candidate)

    return " ".join(fixed_words)


class GrammarEngine:
    """Thin wrapper around the HF T5 grammar-correction model. Loaded once
    and reused across requests (see api/routes.py dependency)."""

    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.device = device or settings.device
        self.model_id = model_id or settings.grammar_model_id
        self.loaded = False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            self.loaded = True
        except Exception as exc:
            print(f"[WARN] Could not load grammar model ({self.model_id}): {exc}. Falling back to lexical correction.")
            self.tokenizer = None
            self.model = None

    @torch.no_grad()
    def correct(self, text: str) -> str:
        if not self.loaded or not text.strip() or self.tokenizer is None or self.model is None:
            return text

        try:
            prompt = f"grammar: {text}"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(
                self.device
            )
            output_ids = self.model.generate(**inputs, max_new_tokens=256, num_beams=4)
            corrected = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return corrected.strip()
        except Exception as exc:
            print(f"[WARN] Grammar correction inference failed: {exc}")
            return text


def two_stage_correct(raw_text: str, sym_spell: SymSpell, grammar_engine: GrammarEngine | None = None) -> str:
    stage1 = lexical_pass(raw_text, sym_spell)
    if grammar_engine is not None:
        try:
            return grammar_engine.correct(stage1)
        except Exception as exc:
            print(f"[WARN] two_stage_correct failed in Stage 2: {exc}")
            return stage1
    return stage1
