import asyncio
from logging import Logger
from typing import Union, Callable

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError

from core.logger import setup_logger


class ConsumerBase(AIOKafkaConsumer):
    """
    Базовый класс для чтения сообщений из Kafka.
    """

    def __init__(self, bootstrap_servers: str, topic: Union[str, list], group_id: str,
                 logger: Logger,
                 value_deserializer:Callable[[str], object], **kwargs):
        topics: tuple[str, ...] = (
            (topic,) if isinstance(topic, str) else tuple(topic)
        )
        super().__init__(
            *topics,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda x: value_deserializer(x.decode('utf-8')) if x else None,
            key_deserializer=lambda x: x.decode('utf-8') if x else None,

            **kwargs
        )
        self._logger = logger
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

    async def __aiter__(self):
        if not self._is_running:  # защита от двойного start()
            await self.start()

        RETRY_DELAY = 5  # секунд

        while self._is_running:
            try:
                # обычный быстрый путь – «тонкий» async-итератор AIOKafkaConsumer
                msg = await super().__anext__()
                yield msg

            # ---------- штатные выхода ----------
            except asyncio.CancelledError:
                self._logger.info("Consumer cancelled – stopping")
                break
            except StopAsyncIteration:  # базовый consumer закрылся
                self._logger.warning("Underlying iterator finished – breaking loop")
                break

            # ---------- ошибки Kafka / сеть ----------
            except (KafkaError, ConnectionError) as e:
                self._logger.error("Kafka error (%s). Restart consumer in %s s", e, RETRY_DELAY)
                try:
                    await self.stop()
                except Exception:
                    self._logger.exception("Error on consumer.stop()")

                await asyncio.sleep(RETRY_DELAY)

                try:
                    await self.start()
                except Exception as ee:
                    self._logger.exception("Error on consumer.start(): %s", ee)
                    # подождём ещё круг, попытка перезапуска снова пойдёт из верхнего while
                continue

            # ---------- всё прочее ----------
            except Exception as e:
                self._logger.exception("Unexpected consumer error – abort")
                break

        # --- graceful shutdown ---
        try:
            await self.stop()
        finally:
            self._is_running = False


if __name__ == '__main__':
    async def main():
        cons = ConsumerBase("109.172.37.92:9094", "active_tasks", "1", setup_logger("ManagerTask"), CompleteTaskMessage.model_validate_json)
        async for msg in cons:
            print(msg.value)

    asyncio.run(main())

