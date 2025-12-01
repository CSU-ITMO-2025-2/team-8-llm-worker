import json
from typing import Union

from core.kafka.llm_schemas import LlmChatResponse, LlmStreamChunk

LlmApiEvent = Union[LlmChatResponse, LlmStreamChunk]


def llm_api_event_deserializer(data: str) -> LlmApiEvent:
    """
    Универсальный десериализатор для llm.chat.response и llm.chat.token.
    По полям в payload решаем, что за модель.
    """
    payload = json.loads(data)

    # Стрим-чанк всегда содержит delta + index
    if "delta" in payload and "index" in payload:
        return LlmStreamChunk.model_validate(payload)

    # Всё остальное считаем финальным ответом
    return LlmChatResponse.model_validate(payload)