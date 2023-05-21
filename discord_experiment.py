"""A simple Discord bot that reverses messages."""
import asyncio
import os
from pprint import pformat
from typing import Any

import discord
import httpx
import socketio
from discord import Message
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN_HEADER = "X-Chat-Merger-Bot-Token"

DISCORD_BOT_SECRET = os.environ["DISCORD_BOT_SECRET"]
CHAT_MERGER_BOT_TOKEN = os.environ["CHAT_MERGER_BOT_TOKEN"]

httpx_client = httpx.AsyncClient()
sio = socketio.AsyncClient()
discord_client = discord.Client(intents=discord.Intents.default())


async def connect_to_chat_merger_websocket():
    """Connect the bot to the ChatMerger websocket."""
    await sio.connect(
        "ws://localhost:8000/",
        transports=["websocket"],
        headers={BOT_TOKEN_HEADER: CHAT_MERGER_BOT_TOKEN},
    )


@discord_client.event
async def on_ready() -> None:
    """Called when the client is done preparing the data received from Discord."""
    asyncio.create_task(connect_to_chat_merger_websocket())


@discord_client.event
async def on_message(message: Message) -> None:
    """Called when a message is created and sent by a user."""
    if message.author == discord_client.user:
        # TODO is this necessary ?
        return

    # print(message.channel.id, discord_client.get_channel(message.channel.id))
    response = await httpx_client.post(
        "http://localhost:8000/bot_update",
        json={"message": message.content, "channel_id": message.channel.id},
        headers={BOT_TOKEN_HEADER: CHAT_MERGER_BOT_TOKEN},
    )
    await message.channel.send(pformat(response.json()))


@sio.event
async def bot_update(update: dict[str, Any]):
    # print(discord_client.get_channel(update["channel_id"]))
    # await discord_client.get_channel(update["channel_id"]).send(pformat(update))
    # TODO outgoing messages should go through ChatMerger too
    pass


@sio.event
async def connect():
    print("Connected to server")


@sio.event
async def disconnect():
    print("Disconnected from server")


if __name__ == "__main__":
    discord_client.run(DISCORD_BOT_SECRET)
