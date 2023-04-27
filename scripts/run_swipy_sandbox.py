"""Entry point for the chatbot sandbox."""
# pylint: disable=wrong-import-position,import-error
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_PATH = Path(__file__).parents[1]
sys.path.append(str(REPO_PATH))

# from swipy_client import patch_openai
#
# patch_openai()

from chatbots.talk_to_doc import FaissBot
from swipy_client import SwipyBot


async def main() -> None:
    """Run the chatbot sandbox."""

    langchain_src_bot = FaissBot(
        REPO_PATH / "data" / "faiss" / "langchain",
        pretty_path_prefix="langchain/",
    )
    langchain_doc_bot = FaissBot(
        REPO_PATH / "data" / "faiss" / "langchain_docs",
        pretty_path_prefix="langchain/docs/",
    )
    this_repo_bot = FaissBot(
        REPO_PATH / "data" / "faiss" / "this_repo",
        pretty_path_prefix="chatbot-sandbox/",
    )

    fulfillment_tasks = [
        asyncio.create_task(
            SwipyBot(os.environ["LANGCHAIN_SRC_BOT_TOKEN"]).run_fulfillment_client(
                langchain_src_bot.refine_fulfillment_handler
            )
        ),
        asyncio.create_task(
            SwipyBot(os.environ["LANGCHAIN_DOC_BOT_TOKEN"]).run_fulfillment_client(
                langchain_doc_bot.refine_fulfillment_handler
            )
        ),
        asyncio.create_task(
            SwipyBot(os.environ["ANTI_SWIPY_BOT_TOKEN"]).run_fulfillment_client(
                this_repo_bot.refine_fulfillment_handler
            )
        ),
    ]
    print()
    await asyncio.gather(*fulfillment_tasks)


if __name__ == "__main__":
    asyncio.run(main())
