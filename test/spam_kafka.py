import asyncio
import json
import os
import random
import time
import uuid

from aiokafka import AIOKafkaProducer


def _request_id() -> str:
    return str(uuid.uuid4())


def _chat_session_id() -> int:
    return random.randint(1, 1_000_000)


async def main():
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka.team8-ns.svc.cluster.local:9092")
    topic = os.getenv("KAFKA_TOPIC", "llm.chat.request")
    count = int(os.getenv("TASK_COUNT", "50"))

    producer = AIOKafkaProducer(bootstrap_servers=bootstrap, linger_ms=5, acks=1)
    await producer.start()
    try:
        for i in range(count):
            req_id = _request_id()
            payload = {
                "request_id": req_id,
                "chat_session_id": _chat_session_id(),
                "messages": [{"role": "user", "content": f"ping {i} at {time.time():.0f}"}],
                "max_tokens": 16,
                "temperature": 0.1,
                "top_p": 0.9,
            }
            await producer.send_and_wait(topic, json.dumps(payload).encode("utf-8"), key=req_id.encode())
            print(f"sent {i + 1}/{count} -> {req_id}")
        print("done")
    finally:
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(main())
