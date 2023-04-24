"""Patch openai to log requests to Swipy Platform."""
import asyncio
from functools import partial
from typing import Any

from swipy_client.swipy_requestor import log_llm_request, log_llm_response


def patch_openai() -> None:
    """Patch openai to log requests to Swipy Platform."""

    classes = ("ChatCompletion", "Completion", "Embedding")
    functions = (
        ("acreate", True),
        ("create", False),
    )
    for class_name in classes:
        for function_name, is_async in functions:
            _patch_openai_function(class_name, function_name, is_async)


def _patch_openai_function(class_name: str, function_name: str, is_async: bool) -> callable:
    import openai  # pylint: disable=import-outside-toplevel

    patch_function = _async_patch if is_async else _sync_patch
    openai_class = getattr(openai, class_name)

    setattr(
        openai_class,
        function_name,
        partial(
            patch_function,
            f"openai.{class_name}.{function_name}",
            getattr(openai_class, function_name),
        ),
    )


async def _async_patch(caller_name: str, original_function: callable, *args, **kwargs) -> Any:
    llm_request_id_task = asyncio.create_task(log_llm_request(caller_name, args, kwargs))
    response = await original_function(*args, **kwargs)
    asyncio.create_task(log_llm_response(llm_request_id_task, response))
    return response


def _sync_patch(caller_name: str, original_function: callable, *args, **kwargs) -> Any:
    llm_request_id = asyncio.get_event_loop().run_until_complete(log_llm_request(caller_name, args, kwargs))
    response = original_function(*args, **kwargs)
    asyncio.get_event_loop().run_until_complete(log_llm_response(llm_request_id, response))
    return response
