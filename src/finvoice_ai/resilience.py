from dataclasses import dataclass

from finvoice_ai.application.ports import (
    GenerationResult,
    ProviderUnavailableError,
    ResponseGenerator,
    RetrievedDocument,
    Retriever,
)
from finvoice_ai.speech.models import AudioBuffer, SpeechSegment, TranscriptionResult
from finvoice_ai.speech.ports import TranscriptionProvider, TranscriptionUnavailableError


@dataclass
class FaultInjectingRetriever:
    wrapped: Retriever
    unavailable: bool = False

    def retrieve(self, query: str) -> list[RetrievedDocument]:
        if self.unavailable:
            raise ProviderUnavailableError("injected retriever unavailability")
        return self.wrapped.retrieve(query)


@dataclass
class FaultInjectingGenerator:
    wrapped: ResponseGenerator
    unavailable: bool = False

    def generate(
        self,
        message: str,
        context: list[RetrievedDocument],
    ) -> GenerationResult:
        if self.unavailable:
            raise ProviderUnavailableError("injected generator unavailability")
        return self.wrapped.generate(message, context)


@dataclass
class FaultInjectingTranscriber:
    wrapped: TranscriptionProvider
    timeout: bool = False

    def transcribe(
        self,
        audio: AudioBuffer,
        segments: list[SpeechSegment],
    ) -> TranscriptionResult:
        if self.timeout:
            raise TranscriptionUnavailableError("transcription provider timed out")
        return self.wrapped.transcribe(audio, segments)
