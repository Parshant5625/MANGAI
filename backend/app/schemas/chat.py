from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    site_id: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    answer: str
    intent: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list)
    data_mode: str
    synthetic_data: bool
    disclaimer: str
