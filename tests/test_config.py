from finvoice_ai.config import Settings


def test_asr_settings_default_to_lightweight_stub() -> None:
    settings = Settings(_env_file=None)

    assert settings.transcription_provider == "deterministic"
    assert settings.whisper_model_size == "small"
    assert settings.whisper_device == "cpu"
    assert settings.whisper_compute_type == "int8"
    assert settings.whisper_language is None
    assert settings.whisper_beam_size == 5


def test_asr_settings_can_select_offline_provider(monkeypatch) -> None:
    monkeypatch.setenv("FINVOICE_TRANSCRIPTION_PROVIDER", "faster_whisper")
    monkeypatch.setenv("FINVOICE_WHISPER_LANGUAGE", "en")
    monkeypatch.setenv("FINVOICE_WHISPER_BEAM_SIZE", "3")

    settings = Settings(_env_file=None)

    assert settings.transcription_provider == "faster_whisper"
    assert settings.whisper_language == "en"
    assert settings.whisper_beam_size == 3
