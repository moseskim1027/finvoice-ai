import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

from finvoice_ai.config import Settings
from finvoice_ai.evaluation.speech import character_error_rate, word_error_rate
from finvoice_ai.evaluation.speech_manifest import (
    SpeechEvaluationCase,
    SpeechEvaluationManifest,
    load_speech_manifest,
)
from finvoice_ai.evaluation.speech_report import (
    SpeechCaseResult,
    SpeechEvaluationReport,
    build_slice_summaries,
    summarize_results,
)
from finvoice_ai.speech.asr import build_transcription_provider
from finvoice_ai.speech.service import SpeechService
from finvoice_ai.speech.vad import EnergyVoiceActivityDetector
from finvoice_ai.speech.wav import PcmWavLoader


class SpeechEvaluationRunner:
    def __init__(
        self,
        service: SpeechService,
        provider_name: str,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._service = service
        self._provider_name = provider_name
        self._clock = clock

    def run(
        self,
        manifest: SpeechEvaluationManifest,
        dataset_root: Path,
        split: str = "development",
    ) -> SpeechEvaluationReport:
        cases = (
            manifest.cases
            if split == "all"
            else [case for case in manifest.cases if case.split == split]
        )
        results = [self._evaluate_case(case, dataset_root) for case in cases]
        return SpeechEvaluationReport(
            dataset_name=manifest.dataset_name,
            dataset_version=manifest.version,
            evaluation_split=split,
            overall=summarize_results(results),
            slices=build_slice_summaries(results),
            cases=results,
        )

    def _evaluate_case(
        self,
        case: SpeechEvaluationCase,
        dataset_root: Path,
    ) -> SpeechCaseResult:
        try:
            wav_path = _resolve_audio_path(dataset_root, case.audio_path)
            wav_data = wav_path.read_bytes()
        except (OSError, ValueError) as error:
            return self._failed_case(case, type(error).__name__)

        started_at = self._clock()
        try:
            analysis = self._service.analyze(wav_data)
        except Exception as error:
            latency_ms = max(0.0, (self._clock() - started_at) * 1000.0)
            return self._failed_case(case, type(error).__name__, latency_ms)
        latency_ms = max(0.0, (self._clock() - started_at) * 1000.0)

        word_error = word_error_rate(case.reference_transcript, analysis.transcription.text)
        character_error = character_error_rate(
            case.reference_transcript,
            analysis.transcription.text,
        )
        return SpeechCaseResult(
            **_case_dimensions(case),
            status="success",
            hypothesis=analysis.transcription.text,
            detected_language=analysis.transcription.language,
            model=analysis.transcription.model,
            word_errors=word_error.errors,
            reference_words=word_error.reference_units,
            character_errors=character_error.errors,
            reference_characters=character_error.reference_units,
            latency_ms=latency_ms,
            audio_duration_seconds=analysis.duration_seconds,
        )

    def _failed_case(
        self,
        case: SpeechEvaluationCase,
        error_type: str,
        latency_ms: float = 0.0,
    ) -> SpeechCaseResult:
        word_error = word_error_rate(case.reference_transcript, "")
        character_error = character_error_rate(case.reference_transcript, "")
        return SpeechCaseResult(
            **_case_dimensions(case),
            status="failed",
            hypothesis="",
            detected_language="und",
            model=self._provider_name,
            word_errors=word_error.errors,
            reference_words=word_error.reference_units,
            character_errors=character_error.errors,
            reference_characters=character_error.reference_units,
            latency_ms=latency_ms,
            audio_duration_seconds=0.0,
            error_type=error_type,
        )


def _case_dimensions(case: SpeechEvaluationCase) -> dict[str, str]:
    return {
        "case_id": case.case_id,
        "language": case.language,
        "language_mode": case.language_mode,
        "noise_condition": case.noise_condition,
        "device": case.device,
    }


def _resolve_audio_path(dataset_root: Path, audio_path: str) -> Path:
    root = dataset_root.resolve()
    resolved = (root / audio_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("audio path escapes dataset root")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate speech recognition from a manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "test", "all"), default="development")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    provider = build_transcription_provider(settings)
    service = SpeechService(
        loader=PcmWavLoader(),
        vad=EnergyVoiceActivityDetector(),
        transcriber=provider,
    )
    report = SpeechEvaluationRunner(
        service=service,
        provider_name=provider.model_name,
    ).run(
        load_speech_manifest(args.manifest),
        args.dataset_root,
        args.split,
    )
    output = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return int(report.overall.failure_count > 0)


if __name__ == "__main__":
    raise SystemExit(main())
