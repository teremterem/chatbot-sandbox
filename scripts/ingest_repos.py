"""Ingest the repos into FAISS and save FAISS indices to disk."""
# pylint: disable=wrong-import-position
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_PATH = Path(__file__).parents[1]
sys.path.append(str(REPO_PATH))

from chatbots.talk_to_doc import repo_to_faiss


def main() -> None:
    """Ingest the repos into FAISS and save FAISS indices to disk."""
    repo_to_faiss(
        REPO_PATH,
        source_url_base="https://github.com/teremterem/chatbot-sandbox/blob/main/",
    ).save_local(
        str(REPO_PATH / "data" / "faiss" / "this_repo"),
    )
    repo_to_faiss(
        REPO_PATH / ".." / "langchain" / "docs",
        source_url_base="https://github.com/hwchase17/langchain/blob/v0.0.148/docs/",
    ).save_local(
        str(REPO_PATH / "data" / "faiss" / "langchain_docs"),
    )
    repo_to_faiss(
        REPO_PATH / ".." / "langchain",
        source_url_base="https://github.com/hwchase17/langchain/blob/v0.0.148/",
    ).save_local(
        str(REPO_PATH / "data" / "faiss" / "langchain"),
    )


if __name__ == "__main__":
    main()
