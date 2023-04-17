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
    data = {
        "_swipy_caller_name": caller_name,
        **kwargs,
    }
    if args:
        data["_swipy_args"] = args

    response = await _call_websocket(data)
    return response["text_completion_id"]


async def log_text_completion_response(text_completion_id: int, completion_response: dict[str, Any]) -> None:
    """Log a text completion response to Swipy Platform."""
    data = {
        "text_completion_id": text_completion_id,
        "response": completion_response,
    }
    await _call_websocket(data)


async def _call_websocket(data: Any) -> Any:
    # send data
    data = json.dumps(data)
    await _SEND_QUEUE.put(data)
    # receive response
    response = await _RECV_QUEUE.get()
    return json.loads(response)


async def _websocket_client() -> None:
    async with websockets.connect(_SWIPY_WEBSOCKET_URI) as websocket:  # pylint: disable=no-member
        while True:
            # send data
            data_to_send = await _SEND_QUEUE.get()
            await websocket.send(data_to_send)
            # receive response
            data = await websocket.recv()
            await _RECV_QUEUE.put(data)


asyncio.get_event_loop().create_task(_websocket_client())
