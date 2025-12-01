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
)
from core.llm.gemma_model import get_chat, convert_messages

logger = setup_logger("GemmaWorker")


async def process_request(req: LlmChatRequest, producer: ProducerBase):
    logger.info(f"Processing Gemma request {req.request_id}")

    chat = get_chat()
    history = convert_messages(req.messages)

    start = time.perf_counter()
    final_text = ""
    prompt_tokens = 0
    completion_tokens = 0

    index = 0

    async for chunk in chat.stream(
        history,
        max_output_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    ):
        # chunk.text — новый фрагмент текста
        if chunk.text:
            delta = chunk.text
            final_text += delta

            stream_msg = LlmStreamChunk(
                request_id=req.request_id,
                chat_session_id=req.chat_session_id,
                index=index,
                delta=delta,
                is_final=False
            )
            index += 1

            await producer.send_task_message(
                topic=LlmKafkaTopic.CHAT_TOKEN.value,
                key=str(req.request_id),
                message=stream_msg,
            )

        # chunk.usage — метаданные
        if chunk.usage:
            prompt_tokens = chunk.usage.input_tokens
            completion_tokens = chunk.usage.output_tokens

    latency_ms = int((time.perf_counter() - start) * 1000)

    # финальный чанк
    final_chunk = LlmStreamChunk(
        request_id=req.request_id,
        chat_session_id=req.chat_session_id,
        index=index,
        delta="",
        is_final=True
    )
    await producer.send_task_message(
        topic=LlmKafkaTopic.CHAT_TOKEN.value,
        key=str(req.request_id),
        message=final_chunk,
    )

    # финальный ответ
    resp = LlmChatResponse(
        request_id=req.request_id,
        chat_session_id=req.chat_session_id,
        content=final_text,
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        latency_ms=latency_ms,
        finish_reason="stop",
    )

    await producer.send_task_message(
        topic=LlmKafkaTopic.CHAT_RESPONSE.value,
        key=str(req.request_id),
        message=resp,
    )

    logger.info(
        f"Done {req.request_id}: {len(final_text)} chars, total tokens = {prompt_tokens + completion_tokens}"
    )


async def worker_loop():
    consumer = ConsumerBase(
        Settings.KAFKA_SERVERS(),
        LlmKafkaTopic.CHAT_REQUEST.value,
        "llm_worker",
        logger,
        LlmChatRequest.model_validate_json
    )

    producer = ProducerBase(Settings.KAFKA_SERVERS())
    await producer.start()

    async for msg in consumer:
        req: LlmChatRequest = msg.value
        asyncio.create_task(process_request(req, producer))


async def main():
    logger.info("Starting Gemma worker...")
    await worker_loop()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())