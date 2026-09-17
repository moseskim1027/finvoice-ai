import argparse
import hashlib
import json
import math
import random
import struct
import wave
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from finvoice_ai.speech.models import AudioBuffer
from finvoice_ai.speech.wav import PcmWavLoader

TRANSFORM_VERSION = "1.0.0"


@dataclass(frozen=True)
class AudioTransformConfig:
    sample_rate_hz: int = 16_000
    peak_dbfs: float = -3.0
    noise_snr_db: float | None = None
    seed: int = 17


def transform_audio(
    speech: AudioBuffer,
    config: AudioTransformConfig,
    noise: AudioBuffer | None = None,
) -> AudioBuffer:
    transformed = resample_linear(speech, config.sample_rate_hz)
    transformed = normalize_peak(transformed, config.peak_dbfs)
    if config.noise_snr_db is not None:
        if noise is None:
            raise ValueError("noise audio is required when noise_snr_db is set")
        prepared_noise = resample_linear(noise, config.sample_rate_hz)
        transformed = mix_noise(transformed, prepared_noise, config.noise_snr_db, config.seed)
    return transformed


def resample_linear(audio: AudioBuffer, target_sample_rate_hz: int) -> AudioBuffer:
    if target_sample_rate_hz <= 0:
        raise ValueError("target sample rate must be positive")
    if audio.sample_rate_hz == target_sample_rate_hz or not audio.samples:
        return AudioBuffer(target_sample_rate_hz, audio.samples)
    output_length = round(len(audio.samples) * target_sample_rate_hz / audio.sample_rate_hz)
    if output_length <= 1:
        return AudioBuffer(target_sample_rate_hz, audio.samples[:output_length])
    scale = (len(audio.samples) - 1) / (output_length - 1)
    samples = []
    for output_index in range(output_length):
        position = output_index * scale
        left = math.floor(position)
        right = min(left + 1, len(audio.samples) - 1)
        fraction = position - left
        value = audio.samples[left] * (1.0 - fraction) + audio.samples[right] * fraction
        samples.append(_clip(round(value)))
    return AudioBuffer(target_sample_rate_hz, tuple(samples))


def normalize_peak(audio: AudioBuffer, peak_dbfs: float) -> AudioBuffer:
    if peak_dbfs > 0.0:
        raise ValueError("peak dBFS must not exceed zero")
    peak = max((abs(sample) for sample in audio.samples), default=0)
    if peak == 0:
        return audio
    target = 32_767 * (10 ** (peak_dbfs / 20.0))
    gain = target / peak
    return AudioBuffer(
        audio.sample_rate_hz, tuple(_clip(round(sample * gain)) for sample in audio.samples)
    )


def mix_noise(
    speech: AudioBuffer,
    noise: AudioBuffer,
    snr_db: float,
    seed: int,
) -> AudioBuffer:
    if speech.sample_rate_hz != noise.sample_rate_hz:
        raise ValueError("speech and noise sample rates must match")
    if not noise.samples:
        raise ValueError("noise audio must not be empty")
    speech_rms = _rms(speech.samples)
    noise_rms = _rms(noise.samples)
    if speech_rms == 0.0 or noise_rms == 0.0:
        return speech
    start = random.Random(seed).randrange(len(noise.samples))
    target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    gain = target_noise_rms / noise_rms
    mixed = tuple(
        _clip(round(sample + noise.samples[(start + index) % len(noise.samples)] * gain))
        for index, sample in enumerate(speech.samples)
    )
    return AudioBuffer(speech.sample_rate_hz, mixed)


def write_pcm_wav(path: Path, audio: AudioBuffer) -> str:
    frames = struct.pack(f"<{len(audio.samples)}h", *audio.samples) if audio.samples else b""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio.sample_rate_hz)
        wav_file.writeframes(frames)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rms(samples: tuple[int, ...]) -> float:
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _clip(value: int) -> int:
    return max(-32_768, min(32_767, value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply deterministic speech transforms")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--peak-dbfs", type=float, default=-3.0)
    parser.add_argument("--noise", type=Path)
    parser.add_argument("--noise-snr-db", type=float)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ledger", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loader = PcmWavLoader()
    source = loader.load(args.source.read_bytes())
    noise = loader.load(args.noise.read_bytes()) if args.noise else None
    config = AudioTransformConfig(
        sample_rate_hz=args.sample_rate,
        peak_dbfs=args.peak_dbfs,
        noise_snr_db=args.noise_snr_db,
        seed=args.seed,
    )
    output = transform_audio(source, config, noise)
    digest = write_pcm_wav(args.output, output)
    record = {
        "transform_version": TRANSFORM_VERSION,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "noise_sha256": hashlib.sha256(args.noise.read_bytes()).hexdigest() if args.noise else None,
        "output_sha256": digest,
        "config": asdict(config),
    }
    serialized = json.dumps(record, sort_keys=True)
    if args.ledger:
        with args.ledger.open("a", encoding="utf-8") as ledger:
            ledger.write(serialized + "\n")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
