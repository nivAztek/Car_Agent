"""API request/response models (Pydantic v2)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from autochat.agent.profile import UserProfile
from autochat.agent.schemas import Source

# CosmosDBSaver rejects its key separator in thread ids; keep ids to a safe, URL-friendly alphabet.
THREAD_ID_PATTERN = r"^[A-Za-z0-9._-]{1,128}$"
TRACE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"


class ChatRequest(BaseModel):
    thread_id: str = Field(pattern=THREAD_ID_PATTERN, description="Client-generated conversation id (uuid4)")
    message: str = Field(min_length=1, max_length=4000)
    user: str | None = Field(None, max_length=64, description="Optional user label for MLflow traces")


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    profile: UserProfile = Field(default_factory=UserProfile)
    intent: str | None = None
    language: str | None = None
    steps: list[str] = Field(default_factory=list)
    trace_id: str | None = None


class FeedbackRequest(BaseModel):
    trace_id: str = Field(pattern=TRACE_ID_PATTERN)
    value: bool = Field(description="True = thumbs up")
    comment: str | None = Field(None, max_length=1000)
    user: str | None = Field(None, max_length=64)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ThreadHistory(BaseModel):
    thread_id: str
    summary: str = ""
    messages: list[HistoryMessage] = Field(default_factory=list)


class CarSummary(BaseModel):
    car_id: str
    make: str
    model: str
    make_he: str | None = None
    model_he: str | None = None
    trim: str | None = None
    model_year: int | None = None
    powertrain: str | None = None
    body_type: str | None = None
    segment: str | None = None
    seats: int | None = None
    price_ils: int | None = None
    range_real_km: int | None = None
    tags: list[str] = Field(default_factory=list)
    url: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, dict[str, Any]]
    checkpointer: str
    cars: int
    tracing: bool
