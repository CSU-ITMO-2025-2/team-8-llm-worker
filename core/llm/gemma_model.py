import asyncio
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, List, Optional

import torch
from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)

from config.settings import Settings
from core.kafka.llm_schemas import LlmMessage, TokenUsage


class SingletonMeta(type):
    _instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


@dataclass(frozen=True)
class GemmaConfig:
    model_id_or_path: str = "google/gemma-3-1b-it"
    # для CPU пусть будет None, для GPU можно "auto"
    device_map: Optional[str] = None
    use_fast_tokenizer: bool = True
    # стример: сколько секунд ждать новый кусок текста, прежде чем проверить ошибки/статус
    streamer_timeout_s: float = 5.0


class GemmaChat(metaclass=SingletonMeta):
    def __init__(self, cfg: Optional[GemmaConfig] = None):
        self.cfg = cfg or GemmaConfig()
        self._hf_token = Settings.HF_TOKEN()

        model_path = self._resolve_model_path(self.cfg.model_id_or_path)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=self._hf_token,
            use_fast=self.cfg.use_fast_tokenizer,
        )

        # pad_token нужен для generate
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._use_gpu = torch.cuda.is_available()

        if self._use_gpu:
            # 8-bit квант имеет смысл ТОЛЬКО на CUDA
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self._model = Gemma3ForCausalLM.from_pretrained(
                model_path,
                token=self._hf_token,
                quantization_config=quantization_config,
                device_map="auto",  # пусть сам раскидает на GPU
            )
            # при device_map="auto" нельзя просто self._model.to(...)
            self._device = None  # inputs трогать не будем
        else:
            # CPU: никакого bitsandbytes. Только float32.
            self._model = Gemma3ForCausalLM.from_pretrained(
                model_path,
                token=self._hf_token,
                torch_dtype=torch.float32,
                device_map=None,
            )
            self._model.to("cpu")
            self._device = torch.device("cpu")

        self._model.eval()

    @staticmethod
    def _resolve_model_path(model_id_or_path: str) -> str:
        p = Path(model_id_or_path)
        if p.exists() and p.is_dir():
            return str(p)
        return model_id_or_path

    def _build_inputs(self, messages: List[LlmMessage]):
        """
        Правильнее использовать chat_template токенизатора, если он есть.
        """
        # HF chat format
        chat = []
        for m in messages:
            role = m.role
            if role not in ("system", "user", "assistant"):
                role = "user"
            chat.append({"role": role, "content": m.content})

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer(prompt, return_tensors="pt")
        else:
            # fallback
            prompt = "\n".join([f"{x['role'].upper()}: {x['content']}" for x in chat] + ["ASSISTANT:"])
            inputs = self._tokenizer(prompt, return_tensors="pt")

        # На CPU переносим inputs на cpu.
        # На GPU (device_map="auto") НЕ переносим — модель сама рулит устройствами.
        if self._device is not None:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        return inputs, prompt

    def _count_tokens(self, text: str) -> int:
        return int(len(self._tokenizer.encode(text)))

    async def stream_generate(
        self,
        messages: List[LlmMessage],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncIterator[str]:
        inputs, prompt = self._build_inputs(messages)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=self.cfg.streamer_timeout_s,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            streamer=streamer,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        exc_holder = {"exc": None}

        def _run():
            try:
                with torch.no_grad():
                    self._model.generate(**gen_kwargs)
            except BaseException as e:
                exc_holder["exc"] = e

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        # streamer — синхронный итератор, но с timeout.
        # если токенов нет — timeout, тогда проверяем, не упала ли генерация.
        while True:
            try:
                piece = next(streamer)
                if piece:
                    yield piece
                await asyncio.sleep(0)
            except StopIteration:
                break
            except Exception:
                # чаще всего сюда попадает queue.Empty из-за timeout
                if exc_holder["exc"] is not None:
                    raise RuntimeError("Generation failed") from exc_holder["exc"]
                if not thread.is_alive():
                    break
                continue

        if exc_holder["exc"] is not None:
            raise RuntimeError("Generation failed") from exc_holder["exc"]

        await asyncio.to_thread(thread.join, 1.0)

    async def generate_full(
        self,
        messages: List[LlmMessage],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, TokenUsage, int]:
        start = time.perf_counter()

        parts: list[str] = []
        async for chunk in self.stream_generate(messages, max_new_tokens, temperature, top_p):
            parts.append(chunk)

        latency_ms = int((time.perf_counter() - start) * 1000)
        text = "".join(parts)

        # usage считаем приближённо
        _, prompt = self._build_inputs(messages)
        usage = TokenUsage(
            prompt_tokens=self._count_tokens(prompt),
            completion_tokens=self._count_tokens(text),
        )
        return text, usage, latency_ms

    def estimate_usage(self, messages: List[LlmMessage], completion_text: str) -> TokenUsage:
        _, prompt = self._build_inputs(messages)
        return TokenUsage(
            prompt_tokens=self._count_tokens(prompt),
            completion_tokens=self._count_tokens(completion_text),
        )


