"""Ingest the repos into FAISS and save FAISS indices to disk."""
# pylint: disable=wrong-import-position,import-error
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_PATH = Path(__file__).parents[1]
sys.path.append(str(REPO_PATH))

from chatbots.ingestion_utils import repo_to_faiss

LANGCHAIN_VERSION = "v0.0.151"


def _ingest_repos(subfolder: str, chunk_size: int, chunk_overlap: int) -> None:
    repo_to_faiss(
        REPO_PATH,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_url_base="https://github.com/teremterem/chatbot-sandbox/blob/main/",
    ).save_local(
        str(REPO_PATH / "data" / "inbox" / "faiss" / subfolder / "this_repo"),
    )
    repo_to_faiss(
        REPO_PATH / ".." / "langchain" / "docs",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_url_base=f"https://github.com/hwchase17/langchain/blob/{LANGCHAIN_VERSION}/docs/",
    ).save_local(
        str(REPO_PATH / "data" / "inbox" / "faiss" / subfolder / "langchain_docs"),
    )
    repo_to_faiss(
        REPO_PATH / ".." / "langchain" / "tests",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_url_base=f"https://github.com/hwchase17/langchain/blob/{LANGCHAIN_VERSION}/tests/",
    ).save_local(
        str(REPO_PATH / "data" / "inbox" / "faiss" / subfolder / "langchain_tests"),
    )
    repo_to_faiss(
        REPO_PATH / ".." / "langchain",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        additional_gitignore_content="docs/\ntests/",
        source_url_base=f"https://github.com/hwchase17/langchain/blob/{LANGCHAIN_VERSION}/",
    ).save_local(
        str(REPO_PATH / "data" / "inbox" / "faiss" / subfolder / "langchain_src"),
    )


def main() -> None:
    """Ingest the repos into FAISS and save FAISS indices to disk."""
    _ingest_repos("1000-200", chunk_size=1000, chunk_overlap=200)
    _ingest_repos("2000-400", chunk_size=2000, chunk_overlap=400)


if __name__ == "__main__":
    main()
