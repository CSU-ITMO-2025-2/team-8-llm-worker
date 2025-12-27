import asyncio
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, List, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from core.kafka.llm_schemas import LlmMessage, TokenUsage

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
# -------------------- utils --------------------

class SingletonMeta(type):
    _instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class DegenerationStop(StoppingCriteria):
    def __init__(self, tokenizer, max_repeat_ngrams=25, n=4, max_newlines=30):
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


# -------------------- config --------------------

@dataclass(frozen=True)
class BitNetConfig:
    model_id: str = "microsoft/bitnet-b1.58-2B-4T"
    use_fast_tokenizer: bool = True
    streamer_timeout_s: float = 5.0

    stream_flush_interval_s: float = 0.08
    stream_min_chars: int = 40
    stream_drop_whitespace_only: bool = True


# -------------------- chat --------------------

class BitNetChat(metaclass=SingletonMeta):
    def __init__(self, cfg: Optional[BitNetConfig] = None):
        self.cfg = cfg or BitNetConfig()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            use_fast=self.cfg.use_fast_tokenizer,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self._model.eval()

        self._stopping = StoppingCriteriaList(
            [DegenerationStop(self._tokenizer)]
        )

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    # -------- prompt --------

    def _build_inputs(self, messages: List[LlmMessage]):
        chat = []
        for m in messages:
            role = m.role if m.role in ("system", "user", "assistant") else "user"
            chat.append({"role": role, "content": m.content})

        prompt = self._tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        return inputs, prompt

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    # -------- streaming --------

    async def stream_generate(
        self,
        messages: List[LlmMessage],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncIterator[str]:

        inputs, _ = self._build_inputs(messages)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=self.cfg.streamer_timeout_s,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            streamer=streamer,
            stopping_criteria=self._stopping,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        exc: Optional[BaseException] = None

        def _run():
            nonlocal exc
            try:
                with torch.no_grad():
                    self._model.generate(**gen_kwargs)
            except BaseException as e:
                exc = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        buf, buf_len = [], 0
        last_flush = time.monotonic()

        def flush(force=False):
            nonlocal buf, buf_len, last_flush
            if not buf:
                return None
            if not force:
                if (time.monotonic() - last_flush) < self.cfg.stream_flush_interval_s \
                   and buf_len < self.cfg.stream_min_chars:
                    return None
            out = "".join(buf)
            buf, buf_len = [], 0
            last_flush = time.monotonic()
            return out

        while True:
            try:
                piece = next(streamer)
                if piece and not (self.cfg.stream_drop_whitespace_only and piece.strip() == ""):
                    buf.append(piece)
                    buf_len += len(piece)

                out = flush()
                if out:
                    yield out

                await asyncio.sleep(0)

            except StopIteration:
                break

        out = flush(force=True)
        if out:
            yield out

        if exc:
            raise RuntimeError("Generation failed") from exc

    # -------- full --------

    async def generate_full(
        self,
        messages: List[LlmMessage],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        start = time.perf_counter()
        parts = []

        async for chunk in self.stream_generate(messages, max_new_tokens, temperature, top_p):
            parts.append(chunk)

        latency_ms = int((time.perf_counter() - start) * 1000)
        text = "".join(parts)

        _, prompt = self._build_inputs(messages)
        usage = TokenUsage(
            prompt_tokens=self._count_tokens(prompt),
            completion_tokens=self._count_tokens(text),
        )

        return text, usage, latency_ms
