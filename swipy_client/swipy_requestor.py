"""This module is responsible for sending requests to Swipy Platform."""
import asyncio
import json
import logging
import os
from contextvars import ContextVar
from typing import Any

import httpx
import websockets

swipy_platform_http_uri = os.getenv("SWIPY_PLATFORM_HTTP_URI", "http://localhost:8000")
swipy_platform_ws_uri = os.getenv("SWIPY_PLATFORM_WS_URI", "ws://localhost:8000")

_client = httpx.Client()
logger = logging.getLogger(__name__)

_fulfillment_id_var: ContextVar[int | None] = ContextVar("fulfillment_id", default=None)


async def log_llm_request(caller_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    """Log a LLM request to Swipy Platform."""
    data = {
        "_swipy_caller_name": caller_name,
        **kwargs,
    }
    if args:
        data["_swipy_args"] = args

    fulfillment_id = _fulfillment_id_var.get()
    if fulfillment_id is not None:
        data["_swipy_fulfillment_id"] = fulfillment_id

    response = _client.post(f"{swipy_platform_http_uri}/log_llm_request/", json=data)
    return response.json()["llm_request_id"]


async def log_llm_response(llm_request_id: int, llm_response: dict[str, Any]) -> None:
    """Log a LLM response to Swipy Platform."""
    _client.post(f"{swipy_platform_http_uri}/log_llm_response/{llm_request_id}/", json=llm_response)


async def send_message(fulfillment_id: int, **data) -> None:
    """
    Send a message from a chatbot on Swipy Platform. Which chatbot, which chat, etc. is determined by the
    fulfillment_id.
    """
    _client.post(f"{swipy_platform_http_uri}/send_message/{fulfillment_id}/", json=data)


async def _fulfillment_handler_wrapper(
    fulfillment_handler: callable, fulfillment_id: int, data: dict[str, Any]
) -> None:
    """Wrapper for fulfillment handlers to catch exceptions and report them to the user."""
    # pylint: disable=broad-exception-caught
    try:
        _fulfillment_id_var.set(fulfillment_id)
        await fulfillment_handler(fulfillment_id, data)
    except Exception as exc:
        logger.exception(exc)
        try:
            await send_message(fulfillment_id, text=f"⚠️ Oops, something went wrong ⚠️\n\n{exc}")
        except Exception as exc2:
            # TODO do we even need to log this ? it may confuse what was the original error
            logger.warning("FAILED TO REPORT ERROR TO THE USER", exc_info=exc2)


async def run_fulfillment_client(fulfillment_handler: callable) -> None:
    """Connect to Swipy Platform and listen for fulfillment requests."""
    # pylint: disable=no-member
    # TODO reconnect if connection is lost ? how many times ? exponential backoff ?
    async with websockets.connect(f"{swipy_platform_ws_uri}/fulfillment_websocket/") as websocket:
        while True:
            data_str = await websocket.recv()
            data = json.loads(data_str)
            fulfillment_id = data.pop("fulfillment_id")
            asyncio.create_task(_fulfillment_handler_wrapper(fulfillment_handler, fulfillment_id, data))
