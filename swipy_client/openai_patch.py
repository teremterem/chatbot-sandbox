"""Patch openai to log requests to Swipy Platform."""
from typing import Any

from swipy_client.swipy_requestor import log_llm_request, log_llm_response


def patch_openai() -> None:
    """Patch openai to log requests to Swipy Platform."""
    import openai  # pylint: disable=import-outside-toplevel

    _chat_completion_acreate_original = openai.ChatCompletion.acreate

    async def _chat_completion_acreate(*args, **kwargs) -> dict[str, Any]:
        # TODO do asyncio.create_task() instead of await ?
        llm_request_id = await log_llm_request("openai.ChatCompletion.create", args, kwargs)

        response = await _chat_completion_acreate_original(*args, **kwargs)

        # TODO do asyncio.create_task() instead of await ?
        await log_llm_response(llm_request_id, response)

        return response

    openai.ChatCompletion.acreate = _chat_completion_acreate

    # TODO patch the rest of the openai methods
