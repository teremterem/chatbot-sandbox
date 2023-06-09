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

from chatbots.talk_to_doc import ConvRetrievalBot, StuffConvRetrievalBot
from chatbots.ingestion_utils import get_embeddings
from swipy_client import SwipyBot


def create_langchain_experiments(exp_name_suffix: str, faiss_subfolder: str) -> list[ConvRetrievalBot]:
    """Create the experiments for the langchain_src bot."""
    embeddings = get_embeddings()

    langchain_faiss = FAISS.load_local(
        str(REPO_PATH / "data" / "faiss" / faiss_subfolder / "langchain_full"),
        embeddings,
    )
    auto_gpt_faiss = FAISS.load_local(
        str(REPO_PATH / "data" / "faiss" / faiss_subfolder / "auto_gpt_full"),
        embeddings,
    )
    # langchain_src_faiss = FAISS.load_local(
    #     str(REPO_PATH / "data" / "faiss" / faiss_subfolder / "langchain_src"),
    #     embeddings,
    # )
    # langchain_test_faiss = FAISS.load_local(
    #     str(REPO_PATH / "data" / "faiss" / faiss_subfolder / "langchain_tests"),
    #     embeddings,
    # )
    # langchain_doc_faiss = FAISS.load_local(
    #     str(REPO_PATH / "data" / "faiss" / faiss_subfolder / "langchain_docs"),
    #     embeddings,
    # )

    def create_gpt_4_3_experiments(_exp_name_suffix: str, use_gpt4: bool) -> list[ConvRetrievalBot]:
        langchain_bot = StuffConvRetrievalBot(
            SwipyBot(os.environ["LANGCHAIN_SRC_BOT_TOKEN"], experiment_name=f"lc_{_exp_name_suffix}"),
            langchain_faiss,
            use_gpt4=use_gpt4,
            pretty_path_prefix="langchain/",
        )
        auto_gpt_bot = StuffConvRetrievalBot(
            SwipyBot(os.environ["ANTI_SWIPY_BOT_TOKEN"], experiment_name=f"ag_{_exp_name_suffix}"),
            auto_gpt_faiss,
            use_gpt4=use_gpt4,
            pretty_path_prefix="Auto-GPT/",
        )
        return [langchain_bot, auto_gpt_bot]
        # langchain_src_bot = StuffConvRetrievalBot(
        #     SwipyBot(os.environ["LANGCHAIN_SRC_BOT_TOKEN"], experiment_name=f"lc_src_{_exp_name_suffix}"),
        #     langchain_src_faiss,
        #     use_gpt4=use_gpt4,
        #     pretty_path_prefix="langchain/",
        # )
        # langchain_test_bot = StuffConvRetrievalBot(
        #     SwipyBot(os.environ["LANGCHAIN_SRC_BOT_TOKEN"], experiment_name=f"lc_test_{_exp_name_suffix}"),
        #     langchain_test_faiss,
        #     use_gpt4=use_gpt4,
        #     pretty_path_prefix="langchain/tests/",
        # )
        # langchain_doc_bot = StuffConvRetrievalBot(
        #     SwipyBot(os.environ["LANGCHAIN_SRC_BOT_TOKEN"], experiment_name=f"lc_doc_{_exp_name_suffix}"),
        #     langchain_doc_faiss,
        #     use_gpt4=use_gpt4,
        #     pretty_path_prefix="langchain/docs/",
        # )
        # return [langchain_src_bot, langchain_test_bot, langchain_doc_bot]

    return [
        *create_gpt_4_3_experiments("gpt3_" + exp_name_suffix, use_gpt4=False),
        *create_gpt_4_3_experiments("gpt4_" + exp_name_suffix, use_gpt4=True),
    ]


async def main() -> None:
    """Run the chatbot sandbox."""

    # Create the bots
    bots = [
        *create_langchain_experiments("2k", "2000-1000"),
        *create_langchain_experiments("1k", "1000-500"),
        # *create_langchain_experiments("2k", "2000-400"),
        # *create_langchain_experiments("1k", "1000-200"),
    ]
    bots[0].swipy_bot.experiment_name += "-default"
    bots[1].swipy_bot.experiment_name += "-default"

    print()
    await asyncio.gather(*(asyncio.create_task(bot.run_fulfillment_client()) for bot in bots))


if __name__ == "__main__":
    asyncio.run(main())
