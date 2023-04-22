"""This module is responsible for sending requests to Swipy Platform."""
import asyncio
import json
import logging
import os
from contextvars import ContextVar
from typing import Any, Callable, Awaitable

import httpx
import websockets
from httpx import Response

swipy_platform_http_uri = os.getenv("SWIPY_PLATFORM_HTTP_URI", "http://localhost:8000")
swipy_platform_ws_uri = os.getenv("SWIPY_PLATFORM_WS_URI", "ws://localhost:8000")
default_swipy_bot_token = os.getenv("DEFAULT_SWIPY_BOT_TOKEN")

_client = httpx.AsyncClient()
logger = logging.getLogger(__name__)

# TODO a unit test that checks that the contextvars are working as expected in a multi-bot and multi-request scenario
_swipy_bot_var: ContextVar["SwipyBot | None"] = ContextVar("swipy_bot", default=None)
_fulfillment_id_var: ContextVar[int | None] = ContextVar("fulfillment_id", default=None)

FulfillmentHandler = Callable[["SwipyBot", dict[str, Any]], Awaitable[None]]


async def log_llm_request(caller_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    """Log a LLM request to Swipy Platform."""
    data = {
        "_swipy_caller_name": caller_name,
        **kwargs,
    }
    if args:
        data["_swipy_args"] = args

    bot = _swipy_bot_var.get()
    swipy_bot_token = bot.swipy_bot_token if bot else default_swipy_bot_token

    fulfillment_id = _fulfillment_id_var.get()
    if fulfillment_id is not None:
        data["_swipy_fulfillment_id"] = fulfillment_id

    response = await _post("/log_llm_request/", data, swipy_bot_token)
    return response.json()["llm_request_id"]


async def log_llm_response(llm_request_id: int, llm_response: dict[str, Any]) -> None:
    """Log a LLM response to Swipy Platform."""
    bot = _swipy_bot_var.get()
    swipy_bot_token = bot.swipy_bot_token if bot else default_swipy_bot_token

    await _post(f"/log_llm_response/{llm_request_id}/", llm_response, swipy_bot_token)


async def _post(path: str, data: dict[str, Any], swipy_bot_token: str) -> Response:
    """Send a POST request to Swipy Platform."""
    return await _client.post(
        f"{swipy_platform_http_uri}{path}", json=data, headers={"X-Swipy-Bot-Token": swipy_bot_token}
    )


class SwipyBot:
    """A Swipy Platform client."""

    def __init__(self, swipy_bot_token: str) -> None:
        self.swipy_bot_token = swipy_bot_token

    async def run_fulfillment_client(self, fulfillment_handler: FulfillmentHandler) -> None:
        """Connect to Swipy Platform and listen for fulfillment requests."""
        # pylint: disable=no-member
        # TODO reconnect if connection is lost ? how many times ? exponential backoff ?
        async with websockets.connect(f"{swipy_platform_ws_uri}/fulfillment_websocket/") as websocket:
            while True:
                data_str = await websocket.recv()
                data = json.loads(data_str)
                asyncio.create_task(self._fulfillment_handler_wrapper(fulfillment_handler, data))

    async def send_message(self, **data) -> None:
        """
        Send a message from a chatbot on Swipy Platform. Which chatbot, which chat, etc. is determined by the
        fulfillment_id.
        """
        fulfillment_id = _fulfillment_id_var.get()
        await _post(f"/send_message/{fulfillment_id}/", data, self.swipy_bot_token)

    async def _fulfillment_handler_wrapper(
        self, fulfillment_handler: FulfillmentHandler, data: dict[str, Any]
    ) -> None:
        """Wrapper for fulfillment handlers to catch exceptions and report them to the user."""
        # pylint: disable=broad-exception-caught
        fulfillment_id = data["fulfillment_id"]
        _fulfillment_id_var.set(fulfillment_id)
        try:
            _swipy_bot_var.set(self)
            await fulfillment_handler(self, data)
        except Exception as exc:
            logger.exception(exc)
            try:
                await self.send_message(text=f"⚠️ Oops, something went wrong ⚠️\n\n{exc}")
            except Exception as exc2:
                # TODO do we even need to log this ? it may confuse what was the original error
                logger.warning("FAILED TO REPORT ERROR TO THE USER", exc_info=exc2)
