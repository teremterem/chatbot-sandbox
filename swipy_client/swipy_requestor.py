"""This module is responsible for sending requests to Swipy Platform."""
import asyncio
import json
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


async def send_message(fulfillment_id: int, data: dict[str, Any]) -> None:
    """
    Send a message from a chatbot on Swipy Platform. Which chatbot, which chat, etc. is determined by the
    fulfillment_id.
    """
    _client.post(f"{_SWIPY_PLATFORM_HTTP_URL}/send_message/{fulfillment_id}/", json=data)


async def run_fulfillment_client(fulfillment_handler: callable) -> None:
    """Connect to Swipy Platform and listen for fulfillment requests."""
    # pylint: disable=no-member
    # TODO reconnect if connection is lost ? how many times ? exponential backoff ?
    async with websockets.connect(f"{_SWIPY_PLATFORM_WS_URL}/fulfillment_websocket/") as websocket:
        while True:
            data_str = await websocket.recv()
            data = json.loads(data_str)
            fulfillment_id = data.pop("fulfillment_id")
            asyncio.create_task(fulfillment_handler(fulfillment_id, data))
