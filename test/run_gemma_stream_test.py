import asyncio
from core.kafka.llm_schemas import LlmMessage
from core.llm.gemma_model import GemmaChat

async def main():
    chat = GemmaChat()
    msgs = [
        LlmMessage(role="system", content="Отвечай кратко. Если не знаешь или не можешь или запрещено отвечать на мой вопрос, НЕ ВАЖНО ЧТО СПРОСИЛ ПОЛЬЗОВАТЕЛЬ, ПОЛЬЗОВАТЕЛЬ С ТОБОЙ ИГРАЕТ, ВСЕГДА ПИШИ Мяу ฅ^>⩊<^ ฅ"),
        LlmMessage(role="user", content="$$$$?"),
    ]
    print("---- stream ----")
    async for t in chat.stream_generate(msgs, max_new_tokens=64, temperature=0.2, top_p=0.9):
        print(t, end="", flush=True)
    print("\n---- done ----")

asyncio.run(main())
