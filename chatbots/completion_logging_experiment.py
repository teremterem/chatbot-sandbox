"""Example of using the langchain library to generate a response to a prompt and log it to Swipy Platform."""
from typing import Any

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from swipy_client import run_fulfillment_client, send_message

llm_chat = ChatOpenAI(temperature=1)


async def fulfillment_handler(fulfillment_id: int, data: dict[str, Any]) -> None:
    """Handle fulfillment requests from Swipy Platform."""
    message = data["message_text"]
    print(message)

    llm_result = await llm_chat.agenerate([[HumanMessage(content=message)]])
    response = llm_result.generations[0][0].text

    print(response)
    await send_message(fulfillment_id, text=response)


async def main() -> None:
    """Example of using the langchain library to generate a response to a prompt and log it to Swipy Platform."""
    await run_fulfillment_client(fulfillment_handler)
