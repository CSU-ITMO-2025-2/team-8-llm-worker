import asyncio
import json
import os
import random
import string
import time
import uuid

from aiokafka import AIOKafkaProducer

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka.team8-ns.svc.cluster.local:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "llm.chat.request")
COUNT = int(os.getenv("TASK_COUNT", "50"))

def _rid():
    return str(uuid.uuid4())

async def main():
    producer = AIOKafkaProducer(bootstrap_servers=BOOTSTRAP, linger_ms=5, acks=1)
    await producer.start()
    try:
        for i in range(COUNT):
            req_id = _rid()
            payload = {
                "request_id": req_id,
                "chat_session_id": _rid(),
                "messages": [{"role": "user", "content": f"ping {i} at {time.time():.0f}"}],
                "max_tokens": 16,
                "temperature": 0.1,
                "top_p": 0.9,
            }
            await producer.send_and_wait(TOPIC, json.dumps(payload).encode("utf-8"), key=req_id.encode())
            print(f"sent {i+1}/{COUNT} -> {req_id}")
        print("done")
    finally:
        await producer.stop()

if __name__ == "__main__":
    asyncio.run(main())
