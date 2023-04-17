"""This module is responsible for sending requests to Swipy Platform."""
from typing import Any

import httpx

_swipy_platform_url = "http://localhost:8000"

_client = httpx.Client()


def log_text_completion_request(
    caller_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> int:
    """Log a text completion request to Swipy Platform."""
    payload = {
        "_swipy_caller_name": caller_name,
        **kwargs,
    }
    if args:
        payload["_swipy_args"] = args

    response = _client.post(
        f"{_swipy_platform_url}/text_completion_request/",
        json=payload,
    )
    return response.json()["text_completion_id"]


def log_text_completion_response(
    text_completion_id: int, response: dict[str, Any]
) -> None:
    """Log a text completion response to Swipy Platform."""
    _client.post(
        f"{_swipy_platform_url}/text_completion_response/{text_completion_id}/",
        json=response,
    )
