import asyncio
import time
from functools import partial

from config.settings import Settings
from core.llm.qwen_model import QwenChatMinMem
from core.logger import setup_logger
from core.kafka.consumer import ConsumerBase
from core.kafka.producer import ProducerBase
from core.kafka.llm_topics import LlmKafkaTopic
from core.kafka.llm_schemas import (
    LlmChatRequest,
    LlmChatResponse,
    LlmStreamChunk,
    TokenUsage,
    LlmError,
)
from core.llm.gemma_model import GemmaChat  # <-- Transformers GemmaChat (singleton)

logger = setup_logger("GemmaWorker")


async def _send_final_chunk(producer: ProducerBase, req: LlmChatRequest, index: int):
    final_chunk = LlmStreamChunk(
        request_id=req.request_id,
        chat_session_id=req.chat_session_id,
        index=index,
        delta="",
        is_final=True,
    )
    await producer.send_task_message(
        topic=LlmKafkaTopic.CHAT_TOKEN.value,
        key=str(req.request_id),
        message=final_chunk,
    )


async def process_request(req: LlmChatRequest, producer: ProducerBase):
    logger.info("Processing Gemma request %s", req.request_id)

    start = time.perf_counter()
    index = 0
    parts: list[str] = []

    try:
        chat = QwenChatMinMem()  # singleton: модель уже прогрета после первого запроса

        async for delta in chat.stream_generate(
            messages=req.messages,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        ):
            parts.append(delta)

            stream_msg = LlmStreamChunk(
                request_id=req.request_id,
                chat_session_id=req.chat_session_id,
                index=index,
                delta=delta,
                is_final=False,
            )
            index += 1

            await producer.send_task_message(
                topic=LlmKafkaTopic.CHAT_TOKEN.value,
                key=str(req.request_id),
                message=stream_msg,
            )

        await _send_final_chunk(producer, req, index)

        final_text = "".join(parts)
        latency_ms = int((time.perf_counter() - start) * 1000)

        usage = chat.estimate_usage(req.messages, final_text)

        resp = LlmChatResponse(
            request_id=req.request_id,
            chat_session_id=req.chat_session_id,
            content=final_text,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason="stop",
        )

        await producer.send_task_message(
            topic=LlmKafkaTopic.CHAT_RESPONSE.value,
            key=str(req.request_id),
            message=resp,
        )

        logger.info(
            "Done %s: %d chars, tokens=%d, latency=%dms",
            req.request_id,
            len(final_text),
            usage.prompt_tokens + usage.completion_tokens,
            latency_ms,
        )

    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)

        # даже при ошибке — закрываем стрим, чтобы SSE не висел
        try:
            await _send_final_chunk(producer, req, index)
        except Exception:
            pass

        err = LlmError(code="gemma_error", message=str(e))
        resp = LlmChatResponse(
            request_id=req.request_id,
            chat_session_id=req.chat_session_id,
            content="",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
            latency_ms=latency_ms,
            finish_reason="error",
            error=err,
        )

        await producer.send_task_message(
            topic=LlmKafkaTopic.CHAT_RESPONSE.value,
            key=str(req.request_id),
            message=resp,
        )

        logger.exception("Gemma request %s failed: %s", req.request_id, e)


async def worker_loop(stop_event: asyncio.Event, consumer: ConsumerBase, producer: ProducerBase):
    QwenChatMinMem()
    await producer.start()

    logger.info(
        "Starting listener (topic=%s, group=%s)",
        Settings.KAFKA_TOPIC(),
        Settings.KAFKA_CONSUMER_GROUP(),
    )

    async for msg in consumer:
        if stop_event.is_set():
            break
        req: LlmChatRequest = msg.value
        await process_request(req, producer)

    await consumer.stop()
    await producer.stop()


async def _health_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, state: dict):
    try:
        request_line = await reader.readline()
        if not request_line:
            return
        try:
            _, path, *_ = request_line.decode().strip().split()
        except ValueError:
            path = "/"

        # consume headers
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b""):
                break

        status = 200
        body = "ok"

        consumer: ConsumerBase | None = state.get("consumer")
        producer: ProducerBase | None = state.get("producer")
        worker_task: asyncio.Task | None = state.get("worker_task")
        topic: str = state.get("topic", "")
        stop_event: asyncio.Event = state["stop_event"]

        if path == "/health/live":
            alive = worker_task is not None and not worker_task.done()
            status = 200 if alive else 503
            body = "live" if alive else "stopped"
        elif path == "/health/ready":
            ready = (
                consumer is not None
                and getattr(consumer, "_is_running", False)
                and producer is not None
                and getattr(producer, "_is_running", False)
            )
            try:
                subs = consumer.subscription() if consumer else set()
                ready = ready and subs and topic in subs
            except Exception:
                ready = False
            status = 200 if ready else 503
            body = "ready" if ready else "not-ready"
        elif path == "/shutdown":
            stop_event.set()
            status = 200
            body = "shutting-down"
        else:
            status = 404
            body = "not-found"

        resp = (
            f"HTTP/1.1 {status} {'OK' if status == 200 else 'FAIL'}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Content-Type: text/plain\r\n"
            "Connection: close\r\n"
            "\r\n"
            f"{body}"
        )
        writer.write(resp.encode())
        await writer.drain()
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def run_health_server(state: dict, host: str = "0.0.0.0", port: int = 8080):
    server = await asyncio.start_server(partial(_health_handler, state=state), host=host, port=port)
    state["server"] = server
    serve_task = asyncio.create_task(server.serve_forever())
    try:
        await state["stop_event"].wait()
    finally:
        server.close()
        await server.wait_closed()
        serve_task.cancel()
        try:
            await serve_task
        except Exception:
            pass


async def main():
    stop_event = asyncio.Event()

    consumer = ConsumerBase(
        Settings.KAFKA_SERVERS(),
        Settings.KAFKA_TOPIC(),
        Settings.KAFKA_CONSUMER_GROUP(),
        logger,
        LlmChatRequest.model_validate_json,
    )
    producer = ProducerBase(Settings.KAFKA_SERVERS())
    state = {
        "stop_event": stop_event,
        "consumer": consumer,
        "producer": producer,
        "topic": Settings.KAFKA_TOPIC(),
        "worker_task": None,
        "server": None,
    }

    logger.info("Starting Gemma Transformers worker...")
    worker_task = asyncio.create_task(worker_loop(stop_event, consumer, producer))
    state["worker_task"] = worker_task
    health_task = asyncio.create_task(run_health_server(state, port=Settings.HEALTH_PORT()))

    try:
        await worker_task
    finally:
        stop_event.set()
        try:
            await consumer.stop()
        except Exception:
            pass
        try:
            await producer.stop()
        except Exception:
            pass

        try:
            await health_task
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
