import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from finvoice_ai.evaluation.audio_transforms import (
    TRANSFORM_VERSION,
    AudioTransformConfig,
    transform_audio,
    write_pcm_wav,
)
from finvoice_ai.speech.models import AudioBuffer
from finvoice_ai.speech.wav import PcmWavLoader

VOICES = (
    ("Samantha", "voice-a", "development"),
    ("Daniel", "voice-b", "development"),
    ("Rishi", "voice-c", "development"),
    ("Karen", "voice-d", "test"),
)
NOISE_CONDITIONS = ("clean", "household", "street", "cafe")
SEED = 20260918


def generate(output_root: Path, taxonomy_path: Path) -> Path:
    _require_command("say")
    _require_command("ffmpeg")
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    source_dir = output_root / "source"
    audio_dir = output_root / "audio"
    source_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    noise = {condition: _synthetic_noise(condition) for condition in NOISE_CONDITIONS[1:]}
    cases = []

    for voice_index, (voice, speaker_id, split) in enumerate(VOICES):
        for template_index, template in enumerate(taxonomy["templates"]):
            case_id = f"{template['template_id']}-{speaker_id}"
            aiff_path = source_dir / f"{case_id}.aiff"
            wav_path = source_dir / f"{case_id}.wav"
            subprocess.run(
                [
                    "say",
                    "-v",
                    voice,
                    "-r",
                    "175",
                    "-o",
                    str(aiff_path),
                    template["text"],
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(aiff_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(wav_path),
                ],
                check=True,
            )
            condition = NOISE_CONDITIONS[(template_index + voice_index) % 4]
            phone_simulated = (template_index + voice_index) % 2 == 1
            config = AudioTransformConfig(
                noise_snr_db=None if condition == "clean" else 12.0,
                phone_simulated=phone_simulated,
                seed=SEED + template_index + voice_index * 100,
            )
            source = PcmWavLoader().load(wav_path.read_bytes())
            transformed = transform_audio(source, config, noise.get(condition))
            relative_path = Path("audio") / f"{case_id}.wav"
            digest = write_pcm_wav(output_root / relative_path, transformed)
            cases.append(
                {
                    "case_id": case_id,
                    "audio_path": relative_path.as_posix(),
                    "reference_transcript": template["text"],
                    "language": template["language"],
                    "language_mode": template["language_mode"],
                    "noise_condition": condition,
                    "device": "phone-simulated" if phone_simulated else "high-quality",
                    "utterance_type": template["utterance_type"],
                    "intent_id": template["intent_id"],
                    "speaker_id": speaker_id,
                    "consent_basis": "synthetic-generated",
                    "license": "Apple system voice output; local evaluation only",
                    "provenance": f"macOS say voice {voice}; synthetic noise {condition}",
                    "collection_method": "scripts/generate_synthetic_benchmark.py",
                    "audio_sha256": digest,
                    "split": split,
                }
            )

    manifest = {
        "dataset_name": "finvoice-local-synthetic-bilingual-v1",
        "version": "1.0.0",
        "taxonomy_version": taxonomy["taxonomy_version"],
        "transform_version": TRANSFORM_VERSION,
        "data_statement": (
            "Synthetic macOS English voices reading English, Filipino, and code-switched "
            "templates. Filipino pronunciation is not representative; local evaluation only."
        ),
        "cases": cases,
    }
    manifest_path = output_root / "speech-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    environment = {
        "seed": SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "taxonomy_sha256": hashlib.sha256(taxonomy_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "voices": [voice for voice, _, _ in VOICES],
        "ffmpeg": subprocess.run(
            ["ffmpeg", "-version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0],
    }
    (output_root / "generation-environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def _synthetic_noise(condition: str, duration_seconds: int = 30) -> AudioBuffer:
    sample_rate = 16_000
    count = duration_seconds * sample_rate
    frequency = {"household": 60.0, "street": 180.0, "cafe": 420.0}[condition]
    samples = tuple(
        round(
            4_000 * math.sin(2 * math.pi * frequency * index / sample_rate)
            + 1_000 * math.sin(2 * math.pi * (frequency * 1.7) * index / sample_rate)
        )
        for index in range(count)
    )
    return AudioBuffer(sample_rate, samples)


def _require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise RuntimeError(f"required command is unavailable: {command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the local synthetic benchmark")
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("src/finvoice_ai/evaluation/data/bilingual_benchmark_taxonomy.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = generate(args.output_root, args.taxonomy)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
