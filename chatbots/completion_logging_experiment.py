"""Example of using the langchain library to generate a response to a prompt and log it to Swipy Platform."""

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from swipy_client import run_fulfillment_client

llm_chat = ChatOpenAI(temperature=1)


async def fulfillment_handler(message: str) -> None:
    """Handle fulfillment requests from Swipy Platform."""
    print()
    print(message)
    llm_result = await llm_chat.agenerate([[HumanMessage(content=message)]])
    print()
    print(llm_result.generations[0][0].text)


async def main() -> None:
    """Example of using the langchain library to generate a response to a prompt and log it to Swipy Platform."""
    await run_fulfillment_client(fulfillment_handler)
