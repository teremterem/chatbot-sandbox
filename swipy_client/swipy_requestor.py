"""This module is responsible for sending requests to Swipy Platform."""
import asyncio
import json
from typing import Any

import websockets

_SWIPY_WEBSOCKET_URI = "ws://localhost:8000/swipy_bot_websocket/"
_SEND_QUEUE = asyncio.Queue()
_RECV_QUEUE = asyncio.Queue()


async def log_text_completion_request(caller_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    """Log a text completion request to Swipy Platform."""
    request = {
        "_swipy_caller_name": caller_name,
        **kwargs,
    }
    if args:
        request["_swipy_args"] = args

    await _send_data(request)
    result = await _recv_data()
    return result["text_completion_id"]


async def log_text_completion_response(text_completion_id: int, response: dict[str, Any]) -> None:
    """Log a text completion response to Swipy Platform."""
    data = {
        "text_completion_id": text_completion_id,
        "response": response,
    }
    await _send_data(data)


async def _send_data(data: Any) -> None:
    data = json.dumps(data)
    await _SEND_QUEUE.put(data)


async def _recv_data() -> Any:
    data = await _RECV_QUEUE.get()
    return json.loads(data)


async def _websocket_client() -> None:
    async with websockets.connect(_SWIPY_WEBSOCKET_URI) as websocket:  # pylint: disable=no-member
        while True:
            data_to_send = await _SEND_QUEUE.get()
            await websocket.send(data_to_send)
            data = await websocket.recv()
            await _RECV_QUEUE.put(data)


asyncio.get_event_loop().create_task(_websocket_client())
