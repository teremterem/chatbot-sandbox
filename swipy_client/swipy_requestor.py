"""This module is responsible for sending requests to Swipy Platform."""
import asyncio
from typing import Any

import httpx
import websockets

_SWIPY_PLATFORM_URL_NO_PROTOCOL = "localhost:8000"
_SWIPY_PLATFORM_HTTP_URL = f"http://{_SWIPY_PLATFORM_URL_NO_PROTOCOL}"
_SWIPY_PLATFORM_WS_URL = f"ws://{_SWIPY_PLATFORM_URL_NO_PROTOCOL}"

_client = httpx.Client()


async def log_llm_request(caller_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    """Log a LLM request to Swipy Platform."""
    data = {
        "_swipy_caller_name": caller_name,
        **kwargs,
    }
    if args:
        data["_swipy_args"] = args

    response = _client.post(f"{_SWIPY_PLATFORM_HTTP_URL}/log_llm_request/", json=data)
    return response.json()["llm_request_id"]


async def log_llm_response(llm_request_id: int, llm_response: dict[str, Any]) -> None:
    """Log a LLM response to Swipy Platform."""
    _client.post(f"{_SWIPY_PLATFORM_HTTP_URL}/log_llm_response/{llm_request_id}/", json=llm_response)


async def run_fulfillment_client(fulfillment_handler: callable) -> None:
    """Connect to Swipy Platform and listen for fulfillment requests."""
    # pylint: disable=no-member
    # TODO reconnect if connection is lost ? how many times ? exponential backoff ?
    async with websockets.connect(f"{_SWIPY_PLATFORM_WS_URL}/fulfillment_websocket/") as websocket:
        while True:
            message = await websocket.recv()
            asyncio.create_task(fulfillment_handler(message))
