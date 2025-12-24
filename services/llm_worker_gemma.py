import asyncio
import time

from config.settings import Settings
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
        chat = GemmaChat()  # singleton: модель уже прогрета после первого запроса

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

        # usage считаем приближённо (как в GemmaChat.generate_full)
        prompt = chat._build_inputs(req.messages)[1]  # prompt string
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


async def worker_loop():
    GemmaChat()

    consumer = ConsumerBase(
        Settings.KAFKA_SERVERS(),
        LlmKafkaTopic.CHAT_REQUEST.value,
        "llm_worker",
        logger,
        LlmChatRequest.model_validate_json,
    )

    producer = ProducerBase(Settings.KAFKA_SERVERS())
    await producer.start()

    logger.info("Starting listener")

    async for msg in consumer:
        req: LlmChatRequest = msg.value
        await process_request(req, producer)


async def main():
    logger.info("Starting Gemma Transformers worker...")
    await worker_loop()


if __name__ == "__main__":
    asyncio.run(main())
