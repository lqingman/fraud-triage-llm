"""Phase 3 — speech frontend. Off-the-shelf Whisper; we do NOT train this.

The interesting engineering note for the writeup: ASR errors propagate
downstream and degrade verdict accuracy. Measure end-to-end (audio->verdict)
accuracy vs. gold-transcript accuracy and report the gap.

TODO(Phase 3):
  - load faster_whisper.WhisperModel(config.asr.whisper_model)
  - transcribe(audio_path) -> str
"""

from __future__ import annotations

from pathlib import Path


def transcribe(audio_path: str | Path) -> str:
    # from faster_whisper import WhisperModel
    raise NotImplementedError("Phase 3: implement Whisper transcription")
