import asyncio
import os
from typing import Tuple

from fastapi import FastAPI, Response

DEFAULT_BOOTSTRAP = "kafka:9092"
CONNECT_TIMEOUT_SEC = 2.0

app = FastAPI(title="dummy-llm-worker")


def _bootstrap_target() -> Tuple[str, int, str]:
    raw = os.getenv("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP)
    first = next((item.strip() for item in raw.split(",") if item.strip()), DEFAULT_BOOTSTRAP)

    host, _, port_str = first.partition(":")
    host = host or "kafka"
    try:
        port = int(port_str) if port_str else 9092
    except ValueError:
        port = 9092

    return host, port, first


async def _can_connect(host: str, port: int, timeout: float = CONNECT_TIMEOUT_SEC) -> bool:
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except Exception:
        return False


@app.get("/health/live")
async def live():
    return {"status": "live"}


@app.get("/health/ready")
async def ready(response: Response):
    host, port, raw = _bootstrap_target()
    ok = await _can_connect(host, port)

    if not ok:
        response.status_code = 503
        return {"status": "not-ready"}

    return {"status": "ready", "bootstrap": raw}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
