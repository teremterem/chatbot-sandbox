"""This module is responsible for sending requests to Swipy Platform."""
from typing import Any

import httpx

_SWIPY_PLATFORM_URL = "ws://localhost:8000"

_client = httpx.Client()


async def log_llm_request(caller_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    """Log a LLM request to Swipy Platform."""
    data = {
        "_swipy_caller_name": caller_name,
        **kwargs,
    }
    if args:
        data["_swipy_args"] = args

    response = _client.post(f"{_SWIPY_PLATFORM_URL}/log_llm_request/", json=data)
    return response.json()["llm_request_id"]


async def log_llm_response(llm_request_id: int, llm_response: dict[str, Any]) -> None:
    """Log a LLM response to Swipy Platform."""
    _client.post(f"{_SWIPY_PLATFORM_URL}/log_llm_response/{llm_request_id}/", json=llm_response)


# async def _websocket_client() -> None:
#     # pylint: disable=no-member
#     # TODO reconnect if connection is lost ? how many times ? exponential backoff ?
#     async with websockets.connect(f"{_SWIPY_PLATFORM_URL}/fulfillment_websocket/") as websocket:
#         while True:
#             # TODO receive fulfillment and schedule a task to fulfill it
#             pass


# asyncio.get_event_loop().create_task(_websocket_client())
