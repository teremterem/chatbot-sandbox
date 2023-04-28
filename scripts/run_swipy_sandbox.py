"""Entry point for the chatbot sandbox."""
# pylint: disable=wrong-import-position,import-error
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain import FAISS

load_dotenv()

REPO_PATH = Path(__file__).parents[1]
sys.path.append(str(REPO_PATH))

# from swipy_client import patch_openai
#
# patch_openai()

from chatbots.talk_to_doc import TalkToDocBot, get_embeddings
from swipy_client import SwipyBot


def create_langchain_experiments(exp_name_suffix: str, faiss_subfolder: str) -> list[TalkToDocBot]:
    """Create the experiments for the langchain_src bot."""
    embeddings = get_embeddings()

    langchain_src_faiss = FAISS.load_local(
        str(REPO_PATH / "data" / "faiss" / faiss_subfolder / "langchain_src"),
        embeddings,
    )
    langchain_test_faiss = FAISS.load_local(
        str(REPO_PATH / "data" / "faiss" / faiss_subfolder / "langchain_tests"),
        embeddings,
    )
    langchain_doc_faiss = FAISS.load_local(
        str(REPO_PATH / "data" / "faiss" / faiss_subfolder / "langchain_docs"),
        embeddings,
    )

    def create_gpt_4_3_experiments(_exp_name_suffix: str, use_gpt4: bool) -> list[TalkToDocBot]:
        langchain_src_bot = TalkToDocBot(
            SwipyBot(os.environ["LANGCHAIN_SRC_BOT_TOKEN"], experiment_name=f"lc_src_{_exp_name_suffix}"),
            langchain_src_faiss,
            use_gpt4=use_gpt4,
            pretty_path_prefix="langchain/",
        )
        langchain_test_bot = TalkToDocBot(
            SwipyBot(os.environ["LANGCHAIN_SRC_BOT_TOKEN"], experiment_name=f"lc_test_{_exp_name_suffix}"),
            langchain_test_faiss,
            use_gpt4=use_gpt4,
            pretty_path_prefix="langchain/tests/",
        )
        langchain_doc_bot = TalkToDocBot(
            SwipyBot(os.environ["LANGCHAIN_SRC_BOT_TOKEN"], experiment_name=f"lc_doc_{_exp_name_suffix}"),
            langchain_doc_faiss,
            use_gpt4=use_gpt4,
            pretty_path_prefix="langchain/docs/",
        )
        return [langchain_src_bot, langchain_test_bot, langchain_doc_bot]

    return [
        *create_gpt_4_3_experiments(exp_name_suffix + "_gpt4", use_gpt4=True),
        *create_gpt_4_3_experiments(exp_name_suffix + "_gpt3", use_gpt4=False),
    ]


async def main() -> None:
    """Run the chatbot sandbox."""

    # Create the bots
    bots = [
        *create_langchain_experiments("2k", "2000-400"),
        *create_langchain_experiments("1k", "1000-200"),
    ]
    bots[0].swipy_bot.experiment_name += "-default"

    print()
    await asyncio.gather(*(asyncio.create_task(bot.run_fulfillment_client()) for bot in bots))


if __name__ == "__main__":
    asyncio.run(main())
