"""This module is responsible for sending requests to Swipy Platform."""
import asyncio
import json
import logging
import os
import time
from contextvars import ContextVar
from typing import Any, Callable, Awaitable

import httpx
from httpx import Response
from websockets.client import connect
from websockets.exceptions import ConnectionClosedError

swipy_platform_http_uri = os.getenv("SWIPY_PLATFORM_HTTP_URI", "http://localhost:8000")
swipy_platform_ws_uri = os.getenv("SWIPY_PLATFORM_WS_URI", "ws://localhost:8000")
default_swipy_bot_token = os.getenv("DEFAULT_SWIPY_BOT_TOKEN")  # TODO there should be no default bot

_client = httpx.AsyncClient()
logger = logging.getLogger(__name__)

# TODO a unit test that checks that the contextvars are working as expected in a multi-bot and multi-request scenario
_swipy_bot_var: ContextVar["SwipyBot | None"] = ContextVar("swipy_bot", default=None)
_fulfillment_id_var: ContextVar[int | None] = ContextVar("fulfillment_id", default=None)

FulfillmentHandler = Callable[["SwipyBot", dict[str, Any]], Awaitable[None]]


class SwipyBot:
    """A Swipy Platform client."""

    def __init__(self, swipy_bot_token: str, experiment_name: str = "default") -> None:
        self.swipy_bot_token = swipy_bot_token
        self.experiment_name = experiment_name

    async def run_fulfillment_client(self, fulfillment_handler: FulfillmentHandler) -> None:
        """Connect to Swipy Platform and listen for fulfillment requests."""
        # pylint: disable=no-member
        attempt = 1
        while True:
            attempt_time = time.time()
            try:
                async with connect(
                    f"{swipy_platform_ws_uri}/fulfillment_websocket/",
                    extra_headers={
                        "X-Swipy-Bot-Token": self.swipy_bot_token,
                        "X-Swipy-Experiment-Name": self.experiment_name,
                    },
                ) as websocket:
                    print(f"SWIPY BOT ONLINE ({self.experiment_name})")
                    print()
                    while True:
                        data_str = await websocket.recv()
                        data = json.loads(data_str)
                        asyncio.create_task(self._fulfillment_handler_wrapper(fulfillment_handler, data))
            except ConnectionClosedError:
                if time.time() - attempt_time > 120:
                    # if more than 2 minutes passed since the last attempt_time, then reset the attempt to 1
                    attempt = 1

                sleep_time = 2**attempt
                logger.warning("WEBSOCKET CONNECTION LOST, RETRYING IN %s SECONDS", sleep_time, exc_info=True)
                await asyncio.sleep(sleep_time)

    async def send_message(self, **data) -> None:
        """
        Send a message from a chatbot on Swipy Platform. Which chatbot, which chat, etc. is determined by the
        fulfillment_id.
        """
        # TODO investigate why sometimes message is not sent ? some kind of async "deadlock" ?
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
                await self.send_message(
                    text=f"⚠️ Oops, something went wrong ⚠️\n\n{exc}",
                    is_visible_to_bot=False,
                )
            except Exception as exc2:
                # TODO do we even need to log this ? it may confuse what was the original error
                logger.warning("FAILED TO REPORT ERROR TO THE USER", exc_info=exc2)


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

    response = await _post("/log_llm_request/", data)
    return response.json()["llm_request_id"]


async def log_llm_response(llm_request_id: int | Awaitable[int], llm_response: dict[str, Any]) -> None:
    """Log a LLM response to Swipy Platform."""
    if isinstance(llm_request_id, Awaitable):
        llm_request_id = await llm_request_id

    await _post(f"/log_llm_response/{llm_request_id}/", llm_response)


async def _post(path: str, data: dict[str, Any], swipy_bot_token: str = None) -> Response:
    """Send a POST request to Swipy Platform."""
    if not swipy_bot_token:
        bot = _swipy_bot_var.get()
        swipy_bot_token = bot.swipy_bot_token if bot else default_swipy_bot_token

        if not swipy_bot_token:
            raise ValueError(
                "Request is being made outside of a fulfillment handler context "
                "but DEFAULT_SWIPY_BOT_TOKEN env var is not set"
            )

    url = f"{swipy_platform_http_uri}{path}"
    # TODO introduce logging config so logs like the one below can be enabled/disabled
    # logger.debug("POST: %s\n%s", url, data)
    print(f"POST: {url}\n{data}\n")
    return await _client.post(url, json=data, headers={"X-Swipy-Bot-Token": swipy_bot_token})
