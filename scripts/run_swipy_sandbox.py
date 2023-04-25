"""Entry point for the chatbot sandbox."""
# pylint: disable=wrong-import-position
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_PATH = Path(__file__).parents[1]
sys.path.append(str(REPO_PATH))

from swipy_client import patch_openai

patch_openai()

from chatbots.talk_to_doc import FaissBot


async def main() -> None:
    """Run the chatbot sandbox."""
    await asyncio.gather(
        asyncio.create_task(
            FaissBot(
                os.environ["ANTI_SWIPY_BOT_TOKEN"],
                REPO_PATH / "data" / "faiss" / "this_repo",
            ).run_fulfillment_client()
        ),
        asyncio.create_task(
            FaissBot(
                os.environ["LANGCHAIN_BOT_TOKEN"],
                REPO_PATH / "data" / "faiss" / "langchain",
            ).run_fulfillment_client()
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
