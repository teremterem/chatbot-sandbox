"""Ingest the repos into FAISS and save FAISS indices to disk."""
# pylint: disable=wrong-import-position,import-error
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_PATH = Path(__file__).parents[2]
sys.path.append(str(REPO_PATH))

from chatbots.ingestion_utils import repo_to_faiss

AUTO_GPT_VERSION = "v0.2.2"
LANGCHAIN_VERSION = "v0.0.151"


def _ingest_repos(chunk_size: int, chunk_overlap: int) -> None:
    subfolder = f"{chunk_size}-{chunk_overlap}"
    repo_to_faiss(
        repo_path=REPO_PATH / ".." / "repos-to-ingest" / "Auto-GPT",
        save_to_dir=REPO_PATH / "data" / "inbox" / "faiss" / subfolder / "auto_gpt_full",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_url_base=f"https://github.com/Significant-Gravitas/Auto-GPT/blob/{AUTO_GPT_VERSION}/",
    )
    repo_to_faiss(
        repo_path=REPO_PATH / ".." / "repos-to-ingest" / "langchain",
        save_to_dir=REPO_PATH / "data" / "inbox" / "faiss" / subfolder / "langchain_full",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_url_base=f"https://github.com/hwchase17/langchain/blob/{LANGCHAIN_VERSION}/",
    )


def main() -> None:
    """Ingest the repos into FAISS and save FAISS indices to disk."""
    _ingest_repos(chunk_size=1000, chunk_overlap=500)
    _ingest_repos(chunk_size=2000, chunk_overlap=1000)


if __name__ == "__main__":
    main()
