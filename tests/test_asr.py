"""Phase 3 ASR tests — no network/model download. faster_whisper.WhisperModel
is monkeypatched so this suite runs instantly and offline; the model was
verified manually against a real synthesized audio clip (see
docs/devlog/phase-3-whisper.md) since that check needs real weights + audio."""

from src.asr import transcribe as asr


class _StubSegment:
    def __init__(self, text):
        self.text = text


class _StubModel:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, path):
        return iter(self._segments), {"language": "en"}


def test_transcribe_concatenates_and_strips_segments(monkeypatch):
    stub = _StubModel([_StubSegment("  Hello  "), _StubSegment(" world. ")])
    monkeypatch.setattr(asr, "_load_model", lambda size: stub)
    assert asr.transcribe("fake.wav") == "Hello world."


def test_transcribe_no_segments_returns_empty_string(monkeypatch):
    monkeypatch.setattr(asr, "_load_model", lambda size: _StubModel([]))
    assert asr.transcribe("fake.wav") == ""


def test_load_model_caches_by_size(monkeypatch):
    import faster_whisper

    calls = []

    class _FakeWhisperModel:
        def __init__(self, size, device, compute_type):
            calls.append(size)

    monkeypatch.setattr(faster_whisper, "WhisperModel", _FakeWhisperModel)
    asr._MODEL_CACHE.clear()

    m1 = asr._load_model("tiny")
    m2 = asr._load_model("tiny")
    m3 = asr._load_model("base")

    assert m1 is m2  # same size -> cached, no second instantiation
    assert m1 is not m3
    assert calls == ["tiny", "base"]

    asr._MODEL_CACHE.clear()
