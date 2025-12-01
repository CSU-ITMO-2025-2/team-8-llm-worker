from typing import Optional, List, Tuple, Union

from aiokafka import AIOKafkaProducer
from pydantic import BaseModel

from core.kafka.llm_schemas import LlmChatRequest, LlmChatResponse, LlmStreamChunk
from core.kafka.llm_topics import LlmKafkaTopic


class ProducerBase(AIOKafkaProducer):
    """
    Базовый класс для отправки сообщений в Kafka, наследуется от AIOKafkaProducer.
    """
    def __init__(self, bootstrap_servers: str,  **kwargs):
        super().__init__(bootstrap_servers=bootstrap_servers,
                         value_serializer=lambda x: x.encode('utf-8'),
                         **kwargs)
        self._is_running = False

    async def start(self):
        if not self._is_running:
            await super().start()
            self._is_running = True

    async def stop(self):
        """Корректное отключение"""
        if self._is_running:
            await super().stop()
            self._is_running = False

    async def send_task_message(
        self,
        topic: str,
        key: str,
        message: BaseModel,
        partition: Optional[int] = None,
        timestamp_ms: Optional[int] = None,
        headers: Optional[List[Tuple[str, bytes]]] = None,
    ):
        """
        Отправка задачи в Kafka (наш собственный метод).
        """
        if not self._is_running:
            await self.start()
        try:
            await self.send_and_wait(
                topic,
                value=message.model_dump_json(),
                key=key.encode('utf-8'),
                partition=partition,
                timestamp_ms=timestamp_ms,
                headers=headers,
            )
        except Exception as e:
            raise e

class LlmKafkaProducer(ProducerBase):
    async def send_chat_request(self, message: LlmChatRequest):
        await self.send_task_message(
            topic=LlmKafkaTopic.CHAT_REQUEST.value,
            key=str(message.request_id),
            message=message,
        )

    async def send_chat_response(self, message: LlmChatResponse):
        await self.send_task_message(
            topic=LlmKafkaTopic.CHAT_RESPONSE.value,
            key=str(message.request_id),
            message=message,
        )

    async def send_stream_chunk(self, message: LlmStreamChunk):
        await self.send_task_message(
            topic=LlmKafkaTopic.CHAT_TOKEN.value,
            key=str(message.request_id),
            message=message,
        )