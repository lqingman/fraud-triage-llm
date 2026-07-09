"""Phase 3 — speech frontend. Off-the-shelf faster-whisper (CPU, int8); we do
NOT train this.

The interesting engineering note for the writeup: ASR errors propagate
downstream and degrade verdict accuracy — a mis-transcribed word can flip a
fraud verdict even though the LLM stage is unchanged. Measuring end-to-end
(audio->verdict) accuracy vs. gold-transcript accuracy is future work; see
the Priority 5 roadmap in README.md.

Style note (matches src/train/qlora_train.py, src/eval/evaluate.py): the
faster_whisper import lives inside _load_model, so this module imports
cleanly without the (large, CPU-heavy) ctranslate2 dependency installed, and
`transcribe`'s file-handling logic is exercisable in tests via a stubbed
model.
"""

from __future__ import annotations

from pathlib import Path

# Cache loaded models by size so a long-running server process (src.serve.app)
# doesn't reload weights on every request.
_MODEL_CACHE: dict[str, object] = {}


def _load_model(model_size: str):
    from faster_whisper import WhisperModel

    if model_size not in _MODEL_CACHE:
        # CPU + int8: no GPU required, matches this project's "runs without a
        # GPU host" constraint for everything outside QLoRA training itself.
        _MODEL_CACHE[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _MODEL_CACHE[model_size]


def transcribe(audio_path: str | Path, model_size: str = "base") -> str:
    """Transcribe an audio file to text using faster-whisper.

    Returns the concatenated segment text, stripped. Raises whatever
    faster-whisper/ctranslate2 raises on an unreadable or corrupt file — the
    caller (src.serve.app.triage_audio) is responsible for turning that into
    an HTTP error, not this module.
    """
    model = _load_model(model_size)
    segments, _info = model.transcribe(str(audio_path))
    return " ".join(segment.text.strip() for segment in segments).strip()
