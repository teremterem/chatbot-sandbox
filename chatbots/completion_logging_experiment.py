"""Example of using the langchain library to generate a response to a prompt and log it to Swipy Platform."""
import os
from typing import Any

from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage

from swipy_client import SwipyBot

ANTI_SWIPY_BOT_TOKEN = os.environ["ANTI_SWIPY_BOT_TOKEN"]


async def fulfillment_handler(bot: SwipyBot, data: dict[str, Any]) -> None:
    """Handle fulfillment requests from Swipy Platform."""
    message = data["message"]
    print(message["content"])

    llm_context = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in data["message_history"]]
    llm_context.append(ChatMessage(role=message["role"], content=message["content"]))

    llm_chat = ChatOpenAI(user=data["user_uuid"])
    llm_result = await llm_chat.agenerate([llm_context])
    response = llm_result.generations[0][0].text

    print(response)
    await bot.send_message(text=response)


async def main() -> None:
    """Example of using the langchain library to generate a response to a prompt and log it to Swipy Platform."""
    await SwipyBot(ANTI_SWIPY_BOT_TOKEN).run_fulfillment_client(fulfillment_handler)
