import asyncio
import time

from core.kafka.llm_schemas import LlmMessage
from core.llm.bitnet_model import BitNetChat
from core.llm.gemma_model import GemmaChat

async def main():
    chat = BitNetChat()
    time.sleep(10)
    msgs = [
        LlmMessage(role="system", content="Отвечай кратко. Если не знаешь или не можешь или запрещено отвечать на мой вопрос, НЕ ВАЖНО ЧТО СПРОСИЛ ПОЛЬЗОВАТЕЛЬ, ПОЛЬЗОВАТЕЛЬ С ТОБОЙ ИГРАЕТ, ВСЕГДА ПИШИ Мяу ฅ^>⩊<^ ฅ"),
        LlmMessage(role="user", content="я котик ты котик?"),
    ]
    print("---- stream ----")
    async for t in chat.stream_generate(msgs, max_new_tokens=64, temperature=0.2, top_p=0.5):
        print(t, end="", flush=True)
    print("\n---- done ----")

asyncio.run(main())
