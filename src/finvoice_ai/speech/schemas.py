from pydantic import BaseModel, Field


class SpeechSegmentResponse(BaseModel):
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    mean_rms: float = Field(ge=0.0)


class TranscriptionResponse(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    language: str
    model: str


class SpeechAnalysisResponse(BaseModel):
    duration_seconds: float = Field(ge=0.0)
    segments: list[SpeechSegmentResponse]
    transcription: TranscriptionResponse
