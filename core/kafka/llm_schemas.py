from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from typing import Union

RoleType = Literal["system", "user", "assistant", "tool"]


class LlmMessage(BaseModel):
    role: RoleType
    content: str
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LlmError(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class LlmChatRequest(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)

    chat_session_id: UUID
    user_id: Optional[UUID] = None

    messages: List[LlmMessage]

    model: str = "gemma-2b-it"
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0

    stream: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


FinishReason = Literal["stop", "length", "content_filter", "tool_calls", "error"]


class LlmChatResponse(BaseModel):
    request_id: UUID
    chat_session_id: UUID

    content: Optional[str] = None

    usage: Optional[TokenUsage] = None
    latency_ms: Optional[int] = None
    finish_reason: Optional[FinishReason] = None
    error: Optional[LlmError] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LlmStreamChunk(BaseModel):
    request_id: UUID
    chat_session_id: UUID

    index: int
    delta: str
    is_final: bool = False

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


LlmApiEvent = Union["LlmChatResponse", "LlmStreamChunk"]