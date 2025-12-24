import asyncio
import os
import time

from core.kafka.llm_schemas import LlmMessage
from core.llm.gemma_model import GemmaChat, GemmaConfig


async def stream_with_timeout(chat: GemmaChat, messages, *, first_token_timeout=60):
    """
    Получаем стрим, но если первый токен не пришёл за first_token_timeout секунд — падаем.
    """
    gen = chat.stream_generate(
        messages=messages,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.95,
    )

    # ждём первый chunk отдельно, чтобы не зависнуть
    try:
        first = await asyncio.wait_for(gen.__anext__(), timeout=first_token_timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Timeout: no first token in {first_token_timeout}s. "
            "Likely generation is stuck/too slow on your setup."
        )

    yield first

    async for x in gen:
        yield x


async def main():
    model_path = os.getenv("GEMMA_MODEL_PATH", "google/gemma-3-1b-it")
    chat = GemmaChat(GemmaConfig(model_id_or_path=model_path))

    messages = [
        LlmMessage(role="system", content="Ты полезный ассистент. Отвечай кратко."),
        LlmMessage(role="user", content="Скажи 'Привет' и одно предложение про Kafka."),
    ]

    print("=== STREAM TEST (timeout) ===")
    print("Model:", model_path)
    print("\n--- streaming output ---\n")

    t0 = time.perf_counter()
    parts = []
    chunks = 0

    try:
        async for delta in stream_with_timeout(chat, messages, first_token_timeout=90):
            print(delta, end="", flush=True)
            parts.append(delta)
            chunks += 1
            if chunks > 5000:
                break
    except Exception as e:
        print("\n\n❌ STREAM FAILED:", repr(e))
        # Печать подсказок
        print("\nHints:")
        print("- If you are on CPU, first token can take long; try a smaller model or GPU.")
        print("- Ensure torch is installed correctly and model is fully downloaded.")
        print("- Try forcing CPU explicitly (see below).")
        return

    dt = time.perf_counter() - t0
    text = "".join(parts)

    print("\n\n--- stats ---")
    print("chunks:", chunks)
    print("chars:", len(text))
    print(f"wall time: {dt:.2f}s")
    print("\n✅ OK")


if __name__ == "__main__":
    asyncio.run(main())
