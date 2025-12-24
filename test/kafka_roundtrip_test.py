import asyncio
import sys
import time
from typing import Optional
from uuid import uuid4

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from config.settings import Settings
from core.kafka.llm_topics import LlmKafkaTopic
from core.kafka.llm_schemas import (
    LlmChatRequest,
    LlmChatResponse,
    LlmStreamChunk,
    LlmMessage,
)


BOOTSTRAP = Settings.KAFKA_SERVERS()


def _b(s: str) -> bytes:
    return s.encode("utf-8")


async def produce_request(producer: AIOKafkaProducer, req: LlmChatRequest):
    await producer.send_and_wait(
        topic=LlmKafkaTopic.CHAT_REQUEST.value,
        key=_b(str(req.request_id)),
        value=_b(req.model_dump_json()),
    )


async def consume_tokens(
    request_id: str,
    session_id: str,
    stop_event: asyncio.Event,
    timeout_s: int = 180,
) -> str:
    """
    Читает llm.chat.token и печатает поток.
    Возвращает собранный текст (по чанкам).
    """
    consumer = AIOKafkaConsumer(
        LlmKafkaTopic.CHAT_TOKEN.value,
        bootstrap_servers=BOOTSTRAP,
        group_id=f"test-token-{uuid4()}",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: v.decode("utf-8"),
        key_deserializer=lambda v: v.decode("utf-8") if v else None,
    )
    await consumer.start()

    collected = []
    started_at = time.perf_counter()

    try:
        while not stop_event.is_set():
            if time.perf_counter() - started_at > timeout_s:
                raise TimeoutError(f"Timeout waiting tokens ({timeout_s}s)")

            msg_pack = await consumer.getmany(timeout_ms=500, max_records=50)
            for tp, msgs in msg_pack.items():
                for msg in msgs:
                    # фильтруем по request_id (key или payload)
                    key = msg.key
                    if key != request_id:
                        # ключ может быть пустым/другим, тогда смотрим payload
                        pass

                    chunk = LlmStreamChunk.model_validate_json(msg.value)

                    if str(chunk.request_id) != request_id:
                        continue
                    if str(chunk.chat_session_id) != session_id:
                        continue

                    if chunk.delta:
                        print(chunk.delta, end="", flush=True)
                        collected.append(chunk.delta)

                    if chunk.is_final:
                        # сигнал: стрим закончен
                        stop_event.set()
                        break

            await asyncio.sleep(0)

    finally:
        await consumer.stop()

    return "".join(collected)


async def consume_final_response(
    request_id: str,
    session_id: str,
    stop_event: asyncio.Event,
    timeout_s: int = 180,
) -> Optional[LlmChatResponse]:
    """
    Читает llm.chat.response и возвращает финальный ответ.
    """
    consumer = AIOKafkaConsumer(
        LlmKafkaTopic.CHAT_RESPONSE.value,
        bootstrap_servers=BOOTSTRAP,
        group_id=f"test-resp-{uuid4()}",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: v.decode("utf-8"),
        key_deserializer=lambda v: v.decode("utf-8") if v else None,
    )
    await consumer.start()

    started_at = time.perf_counter()

    try:
        while not stop_event.is_set():
            if time.perf_counter() - started_at > timeout_s:
                raise TimeoutError(f"Timeout waiting response ({timeout_s}s)")

            msg_pack = await consumer.getmany(timeout_ms=500, max_records=20)
            for tp, msgs in msg_pack.items():
                for msg in msgs:
                    resp = LlmChatResponse.model_validate_json(msg.value)

                    if str(resp.request_id) != request_id:
                        continue
                    if str(resp.chat_session_id) != session_id:
                        continue

                    # получили финальный ответ
                    return resp

            await asyncio.sleep(0)

    finally:
        await consumer.stop()

    return None


async def main():
    print("KAFKA_SERVERS =", BOOTSTRAP)

    # Создаём тестовый request
    req = LlmChatRequest(
        request_id=uuid4(),
        chat_session_id=uuid4(),
        user_id=None,
        messages=[
            LlmMessage(role="system", content="Ты полезный ассистент. Отвечай кратко."),
            LlmMessage(role="user", content="Напиши коротенький стих про разработчика и дедлайны (4 строки)"),
        ],
        max_tokens=128,
        temperature=0.3,
        top_p=0.9,
        stream=True,
        metadata={"test": True},
    )

    request_id = str(req.request_id)
    session_id = str(req.chat_session_id)

    producer = AIOKafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        key_serializer=lambda v: v if isinstance(v, (bytes, bytearray)) else _b(str(v)),
        value_serializer=lambda v: v if isinstance(v, (bytes, bytearray)) else _b(str(v)),
    )
    await producer.start()

    stop_event = asyncio.Event()

    try:
        # Запускаем консюмеры заранее (чтобы не пропустить первые чанки)
        token_task = asyncio.create_task(consume_tokens(request_id, session_id, stop_event))
        resp_task = asyncio.create_task(consume_final_response(request_id, session_id, stop_event))

        # Публикуем запрос
        print("\n=== Sending request ===")
        print("request_id:", request_id)
        print("chat_session_id:", session_id)
        await produce_request(producer, req)

        print("\n\n=== Streaming output ===\n")
        streamed_text = await token_task
        print("\n\n=== Stream finished ===")

        resp = await resp_task
        if resp is None:
            print("❌ No final response received")
            sys.exit(1)

        print("\n=== Final response ===")
        if resp.error:
            print("❌ ERROR:", resp.error.code, resp.error.message)
        else:
            print(resp.content)

        print("\n=== Stats ===")
        print("streamed chars:", len(streamed_text))
        print("final chars   :", len(resp.content or ""))
        if resp.usage:
            print("prompt_tokens     :", resp.usage.prompt_tokens)
            print("completion_tokens :", resp.usage.completion_tokens)
        print("latency_ms:", resp.latency_ms)
        print("finish_reason:", resp.finish_reason)

    finally:
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(main())
