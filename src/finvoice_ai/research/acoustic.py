import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from finvoice_ai.research.contracts import ResearchCase
from finvoice_ai.research.text_baseline import PredictionSet
from finvoice_ai.speech.models import AudioBuffer
from finvoice_ai.speech.wav import PcmWavLoader

SUMMARY_FEATURE_NAMES = (
    "duration_seconds",
    "rms",
    "mean_absolute_amplitude",
    "peak_amplitude",
    "zero_crossing_rate",
    "frame_rms_mean",
    "frame_rms_std",
    "voiced_frame_ratio",
)


def extract_summary_features(audio: AudioBuffer) -> tuple[float, ...]:
    samples = audio.samples
    if not samples:
        return (0.0,) * len(SUMMARY_FEATURE_NAMES)
    rms = _rms(samples)
    absolute = [abs(sample) for sample in samples]
    crossings = sum(
        (left < 0 <= right) or (right < 0 <= left)
        for left, right in zip(samples, samples[1:], strict=False)
    )
    frame_size = max(1, round(audio.sample_rate_hz * 0.02))
    frame_rms = [
        _rms(samples[index : index + frame_size]) for index in range(0, len(samples), frame_size)
    ]
    frame_mean = sum(frame_rms) / len(frame_rms)
    frame_variance = sum((value - frame_mean) ** 2 for value in frame_rms) / len(frame_rms)
    return (
        audio.duration_seconds,
        rms,
        sum(absolute) / len(absolute),
        float(max(absolute)),
        crossings / max(1, len(samples) - 1),
        frame_mean,
        math.sqrt(frame_variance),
        sum(value >= 500.0 for value in frame_rms) / len(frame_rms),
    )


class AcousticIntentBaseline:
    def __init__(self, *, c: float = 1.0, seed: int = 17) -> None:
        self._c = c
        self._seed = seed
        self._pipeline = None

    def fit(self, features: list[tuple[float, ...]], cases: list[ResearchCase]):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self._pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=self._c,
                        max_iter=1_000,
                        random_state=self._seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        self._pipeline.fit(features, [case.intent.intent_id for case in cases])
        return self

    def predict(
        self,
        features: list[tuple[float, ...]],
        cases: list[ResearchCase],
    ) -> PredictionSet:
        if self._pipeline is None:
            raise RuntimeError("baseline must be fitted before prediction")
        probabilities = self._pipeline.predict_proba(features)
        predictions = self._pipeline.predict(features)
        return PredictionSet(
            case_ids=tuple(case.case_id for case in cases),
            labels=tuple(case.intent.intent_id for case in cases),
            predictions=tuple(str(value) for value in predictions),
            probabilities=tuple(tuple(float(value) for value in row) for row in probabilities),
            classes=tuple(str(value) for value in self._pipeline.classes_),
        )


def load_summary_feature_matrix(
    cases: list[ResearchCase], dataset_root: Path
) -> list[tuple[float, ...]]:
    loader = PcmWavLoader()
    return [
        extract_summary_features(loader.load((dataset_root / case.audio_path).read_bytes()))
        for case in cases
    ]


class FrozenEmbeddingBackend(Protocol):
    model_revision: str

    def embed(self, audio: AudioBuffer) -> tuple[float, ...]: ...


@dataclass
class FrozenEmbeddingCache:
    root: Path
    backend: FrozenEmbeddingBackend

    def get_or_create(self, audio_hash: str, audio: AudioBuffer) -> tuple[float, ...]:
        key = f"{audio_hash}-{_safe_model_key(self.backend.model_revision)}.json"
        path = self.root / key
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["audio_sha256"] != audio_hash:
                raise ValueError("cached embedding audio hash mismatch")
            if payload["model_revision"] != self.backend.model_revision:
                raise ValueError("cached embedding model revision mismatch")
            return tuple(float(value) for value in payload["embedding"])
        embedding = self.backend.embed(audio)
        self.root.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "audio_sha256": audio_hash,
                    "model_revision": self.backend.model_revision,
                    "embedding": embedding,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return embedding


class WavLMEmbeddingBackend:
    """Lazy frozen WavLM mean-pooling adapter for optional research runs."""

    def __init__(self, model_revision: str) -> None:
        self.model_revision = model_revision
        self._processor = None
        self._model = None

    def embed(self, audio: AudioBuffer) -> tuple[float, ...]:
        try:
            import torch
            from transformers import AutoFeatureExtractor, WavLMModel
        except ImportError as error:
            raise RuntimeError(
                "WavLM embeddings require the optional torch and transformers dependencies"
            ) from error
        model_id, revision = _split_model_revision(self.model_revision)
        if self._model is None:
            self._processor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
            self._model = WavLMModel.from_pretrained(model_id, revision=revision)
            self._model.eval()
            for parameter in self._model.parameters():
                parameter.requires_grad_(False)
        samples = [sample / 32_768.0 for sample in audio.samples]
        inputs = self._processor(samples, sampling_rate=audio.sample_rate_hz, return_tensors="pt")
        with torch.no_grad():
            hidden = self._model(**inputs).last_hidden_state.mean(dim=1).squeeze(0)
        return tuple(float(value) for value in hidden.tolist())


def _rms(samples: tuple[int, ...]) -> float:
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _safe_model_key(model_revision: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in model_revision)


def _split_model_revision(value: str) -> tuple[str, str | None]:
    model_id, separator, revision = value.partition("@")
    return model_id, revision if separator else None
