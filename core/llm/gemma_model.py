import time
from typing import List, AsyncIterator

from gemma import gm

from core.kafka.llm_schemas import LlmMessage, TokenUsage
from core.llm.gemma_async_sampler import AsyncChatSampler

class SingletonMeta(type):
    _instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class GemmaChat(metaclass=SingletonMeta):
    """
    Высокоуровневая обёртка над Gemma3:
    """

    def __init__(self):
        # грузим модель/чекпоинт один раз
        self._model = gm.nn.Gemma3_1B()
        self._params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)

        self._sampler = AsyncChatSampler(
            model=self._model,
            params=self._params,
            multi_turn=False,
        )

    @staticmethod
    def _build_prompt(messages: List[LlmMessage]) -> str:
        """
        Переводим историю сообщений в текст для модели.
        Здесь уже можно наворотить любой формат.
        """
        parts: list[str] = []
        for msg in messages:
            if msg.role == "system":
                prefix = "System"
            elif msg.role == "assistant":
                prefix = "Assistant"
            else:
                prefix = "User"
            parts.append(f"{prefix}: {msg.content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    async def stream_generate(
        self,
        messages: List[LlmMessage],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncIterator[str]:
        """
        Асинхронный стрим текста для воркера.
        """
        prompt = self._build_prompt(messages)

        # temperature/top_p пока не используем → можно протащить в sampling
        async for chunk in self._sampler.stream_chat(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            sampling=None,
        ):
            yield chunk

    async def generate_full(
        self,
        messages: List[LlmMessage],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, TokenUsage, int]:
        """
        Если нужен сразу полный ответ и usage (например для вызова не из воркера).
        """
        start = time.perf_counter()
        parts: list[str] = []
        async for chunk in self.stream_generate(
            messages,
            max_new_tokens,
            temperature,
            top_p,
        ):
            parts.append(chunk)
        latency_ms = int((time.perf_counter() - start) * 1000)

        text = "".join(parts)
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0)

        return text, usage, latency_ms