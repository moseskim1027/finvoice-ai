import hashlib
import json
from pathlib import Path

import pytest

from finvoice_ai.evaluation.audio_transforms import (
    AudioTransformConfig,
    main,
    mix_noise,
    normalize_peak,
    resample_linear,
    simulate_phone_channel,
    transform_audio,
)
from finvoice_ai.speech.models import AudioBuffer
from finvoice_ai.speech.wav import PcmWavLoader
from tests.audio_helpers import create_wav


def test_linear_resampling_is_deterministic() -> None:
    audio = AudioBuffer(4, (0, 100, 200, 300))

    result = resample_linear(audio, 8)

    assert result.sample_rate_hz == 8
    assert result.samples == (0, 43, 86, 129, 171, 214, 257, 300)
    assert resample_linear(result, 8) == result


def test_peak_normalization_scales_and_preserves_silence() -> None:
    audio = AudioBuffer(16_000, (-1_000, 0, 1_000))

    normalized = normalize_peak(audio, -6.0)

    assert max(normalized.samples) == pytest.approx(16_422, abs=1)
    assert normalize_peak(AudioBuffer(16_000, (0, 0)), -3.0).samples == (0, 0)
    with pytest.raises(ValueError, match="must not exceed zero"):
        normalize_peak(audio, 1.0)


def test_seeded_noise_mix_is_repeatable_and_bounded() -> None:
    speech = AudioBuffer(16_000, (20_000, -20_000) * 10)
    noise = AudioBuffer(16_000, (10_000, -5_000, 2_000))

    first = mix_noise(speech, noise, snr_db=0.0, seed=7)
    second = mix_noise(speech, noise, snr_db=0.0, seed=7)

    assert first == second
    assert all(-32_768 <= sample <= 32_767 for sample in first.samples)


def test_phone_simulation_preserves_rate_and_duration() -> None:
    audio = AudioBuffer(16_000, (10_000, -10_000) * 80)

    result = simulate_phone_channel(audio)

    assert result.sample_rate_hz == 16_000
    assert len(result.samples) == len(audio.samples)
    assert result.samples != audio.samples


def test_transform_requires_noise_when_snr_is_configured() -> None:
    with pytest.raises(ValueError, match="noise audio is required"):
        transform_audio(
            AudioBuffer(16_000, (1_000,) * 10),
            AudioTransformConfig(noise_snr_db=10.0),
        )


def test_cli_writes_reproducibility_ledger(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "output.wav"
    ledger = tmp_path / "ledger.jsonl"
    source.write_bytes(create_wav(sample_rate_hz=8_000, duration_seconds=0.1))

    exit_code = main(
        [
            str(source),
            str(output),
            "--sample-rate",
            "16000",
            "--seed",
            "23",
            "--ledger",
            str(ledger),
        ]
    )

    record = json.loads(ledger.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == record
    assert record["config"]["seed"] == 23
    assert record["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert PcmWavLoader().load(output.read_bytes()).sample_rate_hz == 16_000
