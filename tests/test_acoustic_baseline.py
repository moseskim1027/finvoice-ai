from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from finvoice_ai.research.acoustic import (
    AcousticIntentBaseline,
    FrozenEmbeddingCache,
    extract_summary_features,
)
from finvoice_ai.research.contracts import INTENT_LABELS, ResearchCase
from finvoice_ai.speech.models import AudioBuffer


def case(case_id: str, intent: str) -> ResearchCase:
    return ResearchCase(
        case_id=case_id,
        audio_path=f"{case_id}.wav",
        audio_sha256="0" * 64,
        reference_transcript="text",
        asr_transcript="text",
        speaker_id="speaker",
        language="en",
        language_mode="monolingual",
        noise_condition="clean",
        device="high-quality",
        intent=INTENT_LABELS[intent],
        role="train",
    )


def test_summary_features_cover_energy_timing_and_zero_crossings() -> None:
    features = extract_summary_features(AudioBuffer(4, (1_000, -1_000, 1_000, -1_000)))

    assert features[0] == 1.0
    assert features[1] == 1_000.0
    assert features[3] == 1_000.0
    assert features[4] == 1.0
    assert extract_summary_features(AudioBuffer(16_000, ())) == (0.0,) * 8


def test_acoustic_baseline_fits_scaler_on_training_features() -> None:
    training_cases = [
        case("a", "pin_reset"),
        case("b", "pin_reset"),
        case("c", "statement_access"),
        case("d", "statement_access"),
    ]
    features = [(0.0,) * 8, (0.1,) * 8, (1.0,) * 8, (1.1,) * 8]

    prediction = (
        AcousticIntentBaseline(seed=7)
        .fit(features, training_cases)
        .predict([(1.05,) * 8], [case("target", "statement_access")])
    )

    assert prediction.predictions == ("statement_access",)


@dataclass
class FakeEmbeddingBackend:
    model_revision: str = "fake/model@abc123"
    calls: int = 0

    def embed(self, audio: AudioBuffer) -> tuple[float, ...]:
        self.calls += 1
        return (audio.duration_seconds, 2.0)


def test_frozen_embedding_cache_keys_audio_and_model_revision(tmp_path: Path) -> None:
    backend = FakeEmbeddingBackend()
    cache = FrozenEmbeddingCache(tmp_path, backend)
    audio = AudioBuffer(2, (1, 2))

    first = cache.get_or_create("0" * 64, audio)
    second = cache.get_or_create("0" * 64, audio)

    assert first == second == (1.0, 2.0)
    assert backend.calls == 1
    payload = next(tmp_path.iterdir()).read_text(encoding="utf-8")
    assert "fake/model@abc123" in payload
