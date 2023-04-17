"""Patch openai to log requests to Swipy Platform."""
from typing import Any

import openai

from swipy_client.swipy_requestor import (
    log_text_completion_request,
    log_text_completion_response,
)


def patch_openai() -> None:
    """Patch openai to log requests to Swipy Platform."""
    chat_completion_acreate_original = openai.ChatCompletion.acreate

    async def swipy_chat_completion_acreate(*args, **kwargs) -> dict[str, Any]:
        text_completion_id = await log_text_completion_request("openai.ChatCompletion.create", args, kwargs)
        response = await chat_completion_acreate_original(*args, **kwargs)
        await log_text_completion_response(text_completion_id, response)
        return response

    openai.ChatCompletion.acreate = swipy_chat_completion_acreate
