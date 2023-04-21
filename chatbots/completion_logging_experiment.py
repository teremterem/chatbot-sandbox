"""Example of using the langchain library to generate a response to a prompt and log it to Swipy Platform."""
from typing import Any

from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage

from swipy_client import run_fulfillment_client, send_message

llm_chat = ChatOpenAI()


async def fulfillment_handler(fulfillment_id: int, data: dict[str, Any]) -> None:
    """Handle fulfillment requests from Swipy Platform."""
    # TODO attach completion to fulfillment somehow
    # TODO pass user_uuid to openai ?
    # TODO make /start invisible to bot
    message = data["message"]
    print(message["content"])

    llm_context = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in data["message_history"]]
    llm_context.append(ChatMessage(role=message["role"], content=message["content"]))
    llm_result = await llm_chat.agenerate([llm_context])
    response = llm_result.generations[0][0].text

    print(response)
    await send_message(fulfillment_id, text=response)


async def main() -> None:
    """Example of using the langchain library to generate a response to a prompt and log it to Swipy Platform."""
    await run_fulfillment_client(fulfillment_handler)
