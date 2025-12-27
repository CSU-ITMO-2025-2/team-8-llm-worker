import os

# CPU-only hard lock
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import asyncio
import gc
import threading
import time
import queue
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from core.kafka.llm_schemas import LlmMessage, TokenUsage


class SingletonMeta(type):
    _instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class DegenerationStop(StoppingCriteria):
    def __init__(self, tokenizer, max_repeat_ngrams: int = 25, n: int = 4, max_newlines: int = 30):
        self.tokenizer = tokenizer
        self.n = n
        self.max_repeat_ngrams = max_repeat_ngrams
        self.max_newlines = max_newlines

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        ids = input_ids[0].tolist()
        if len(ids) < self.n * 2:
            return False

        tail_text = self.tokenizer.decode(ids[-200:], skip_special_tokens=True)
        if tail_text.count("\n") >= self.max_newlines:
            return True

        tail = ids[-400:]
        seen = {}
        for i in range(len(tail) - self.n + 1):
            ng = tuple(tail[i : i + self.n])
            seen[ng] = seen.get(ng, 0) + 1
            if seen[ng] >= self.max_repeat_ngrams:
                return True

        return False


@dataclass(frozen=True)
class QwenMinMemConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    use_fast_tokenizer: bool = True

    # memory critical:
    prefer_bf16: bool = True
    low_cpu_mem_usage: bool = True
    use_cache: bool = False

    streamer_timeout_s: float = 30.0

    stream_flush_interval_s: float = 0.08
    stream_min_chars: int = 40
    stream_drop_whitespace_only: bool = True


class QwenChatMinMem(metaclass=SingletonMeta):
    def __init__(self, cfg: Optional[QwenMinMemConfig] = None):
        self.cfg = cfg or QwenMinMemConfig()
        self._device = torch.device("cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            use_fast=self.cfg.use_fast_tokenizer,
            trust_remote_code=True,
        )

        dtype = torch.bfloat16 if self.cfg.prefer_bf16 else torch.float32

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_id,
                device_map=None,
                torch_dtype=dtype,
                low_cpu_mem_usage=self.cfg.low_cpu_mem_usage,
                trust_remote_code=True,
            )
        except Exception:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_id,
                device_map=None,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=self.cfg.low_cpu_mem_usage,
                trust_remote_code=True,
            )

        self._model.to(self._device)
        self._model.eval()

        self._stopping = StoppingCriteriaList([DegenerationStop(self._tokenizer)])

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        gc.collect()

    def _build_inputs(self, messages: List[LlmMessage]) -> Tuple[dict, str]:
        chat = []
        for m in messages:
            role = m.role if m.role in ("system", "user", "assistant") else "user"
            chat.append({"role": role, "content": m.content})

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join([f"{x['role'].upper()}: {x['content']}" for x in chat] + ["ASSISTANT:"])

        inputs = self._tokenizer(prompt, return_tensors="pt")
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
        inputs, _prompt = self._build_inputs(messages)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=self.cfg.streamer_timeout_s,  # важно: может быть большой
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
            stopping_criteria=self._stopping,
            use_cache=bool(self.cfg.use_cache),
        )

        exc_holder: dict[str, Optional[BaseException]] = {"exc": None}

        def _run():
            try:
                with torch.no_grad():
                    self._model.generate(**gen_kwargs)
            except BaseException as e:
                exc_holder["exc"] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        buf: list[str] = []
        buf_len = 0
        last_flush = time.monotonic()

        def should_drop(piece: str) -> bool:
            if not piece:
                return True
            if self.cfg.stream_drop_whitespace_only and piece.strip() == "":
                return True
            return False

        def flush(force: bool = False) -> Optional[str]:
            nonlocal buf, buf_len, last_flush
            if not buf:
                return None
            now = time.monotonic()
            if not force:
                if (now - last_flush) < self.cfg.stream_flush_interval_s and buf_len < self.cfg.stream_min_chars:
                    return None
            out = "".join(buf)
            buf = []
            buf_len = 0
            last_flush = now
            return out

        while True:
            try:
                # TextIteratorStreamer кидает queue.Empty при timeout
                piece = next(streamer)

                if piece and not should_drop(piece):
                    buf.append(piece)
                    buf_len += len(piece)

                out = flush(force=False)
                if out:
                    yield out

                await asyncio.sleep(0)

            except StopIteration:
                break

            except queue.Empty:
                # Ничего не пришло за timeout — это НЕ ошибка.
                # Если поток генерации упал — пробросим.
                if exc_holder["exc"] is not None:
                    raise RuntimeError("Generation failed") from exc_holder["exc"]

                # Если генерация ещё идёт — попробуем отдать накопленное
                out = flush(force=False)
                if out:
                    yield out

                # Если поток уже закончился, а очередь пуста — выходим
                if not t.is_alive():
                    break

                await asyncio.sleep(0)
                continue

        out = flush(force=True)
        if out:
            yield out

        if exc_holder["exc"] is not None:
            raise RuntimeError("Generation failed") from exc_holder["exc"]

        await asyncio.to_thread(t.join, 1.0)

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

        _inputs, prompt = self._build_inputs(messages)
        usage = TokenUsage(
            prompt_tokens=self._count_tokens(prompt),
            completion_tokens=self._count_tokens(text),
        )
        return text, usage, latency_ms

    def estimate_usage(self, messages: List[LlmMessage], completion_text: str) -> TokenUsage:
        _inputs, prompt = self._build_inputs(messages)
        return TokenUsage(
            prompt_tokens=self._count_tokens(prompt),
            completion_tokens=self._count_tokens(completion_text),
        )
