"""
This chatbot is based on materials from the following video by Prompt Engineering YouTube channel
(checkout video description for more links): https://www.youtube.com/watch?v=TLf90ipMzfE
"""
import itertools
import os
from pathlib import Path
from typing import Any

import chardet
import magic
from PyPDF2 import PdfReader
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import VectorStore
from pathspec import pathspec

from swipy_client import SwipyBot


class TalkToDocBot:
    """A chatbot that answers questions about a PDF document."""

    def __init__(self, swipy_bot_token: str, vector_store: VectorStore) -> None:
        self.swipy_bot_token = swipy_bot_token
        self.vector_store = vector_store

    async def run_fulfillment_client(self):
        """Connect to Swipy Platform and listen for fulfillment requests."""
        print("BOT STARTED")
        print()
        await SwipyBot(self.swipy_bot_token).run_fulfillment_client(self._fulfillment_handler)

    async def _fulfillment_handler(self, bot: SwipyBot, data: dict[str, Any]) -> None:
        """Handle fulfillment requests from Swipy Platform."""
        print("USER:", data["message"]["content"])
        print()

        query = self._build_query(data)

        llm_chat = ChatOpenAI(
            user=data["user_uuid"],
            temperature=0,
        )
        chain = load_qa_chain(llm_chat, chain_type="stuff")
        docs = await self.vector_store.asimilarity_search(query)
        response = await chain.arun(
            input_documents=docs,
            question=query,
            stop=[
                "\n\nUSER:",
                "\n\nASSISTANT:",
            ],
        )

        print("ASSISTANT:", response)
        print()
        await bot.send_message(
            text=response,
            parse_mode="Markdown",
        )

    @staticmethod
    def _build_query(data: dict[str, Any]) -> str:
        query_parts = []
        for msg in itertools.chain(data["message_history"], (data["message"],)):
            query_parts.append(msg["role"].upper())
            query_parts.append(": ")
            query_parts.append(msg["content"])
            query_parts.append("\n\n")
        query_parts.append("ASSISTANT:")
        return "".join(query_parts)

    @staticmethod
    def build_embeddings() -> Embeddings:
        """Build LangChain's Embeddings object."""
        return OpenAIEmbeddings(allowed_special="all")


class FaissBot(TalkToDocBot):
    """A chatbot that answers questions using a FAISS index that was saved locally."""

    def __init__(self, swipy_bot_token: str, faiss_folder_path: str | Path) -> None:
        embeddings = self.build_embeddings()
        faiss = FAISS.load_local(faiss_folder_path, embeddings)

        super().__init__(
            swipy_bot_token=swipy_bot_token,
            vector_store=faiss,
        )


def pdf_to_faiss(pdf_path: str | Path) -> FAISS:
    """Ingest a PDF and return a FAISS instance."""
    reader = PdfReader(pdf_path)
    raw_text_parts = [page.extract_text() for page in reader.pages]
    raw_text = "\n".join(raw_text_parts)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    for text in texts:
        print(text)
        print("    LENGTH:", len(text), "CHARS")
        print()

    embeddings = FaissBot.build_embeddings()
    return FAISS.from_texts(texts, embeddings)


def repo_to_faiss(repo_path: str | Path) -> FAISS:
    """Ingest a git repository and return a FAISS instance."""
    repo_path = Path(repo_path).resolve()

    print()
    print("REPO:", repo_path)
    print()
    print("================================================================================")
    print()

    filepaths = _list_files_in_repo(repo_path)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = []
    for filepath in filepaths:
        print(filepath, end=" - ")

        with open(repo_path / filepath, "rb") as file:
            raw_bytes = file.read()
        detected_encoding = chardet.detect(raw_bytes).get("encoding") or "utf-8"
        print(detected_encoding)

        try:
            raw_text = raw_bytes.decode(detected_encoding)
        except UnicodeDecodeError as exc:
            print(f"ERROR! SKIPPING A FILE! {exc}")
        else:
            texts.extend(text_splitter.split_text(raw_text))

    print()
    print("================================================================================")
    print()

    # for text in texts:
    #     print(text)
    #     print()
    #     print("================================================================================")
    #     print()
    print()
    print(len(filepaths), "FILES")
    print(len(texts), "SNIPPETS")
    print()
    print("INDEXING...")

    embeddings = FaissBot.build_embeddings()
    faiss = FAISS.from_texts(texts, embeddings)

    print("DONE")
    print()
    return faiss


def _list_files_in_repo(repo_path: str | Path) -> list[Path]:
    repo_path = Path(repo_path)

    # ".*\n" means skip all "hidden" files and directories too
    gitignore_content = ".*\n" + _read_gitignore(repo_path)
    spec = pathspec.PathSpec.from_lines("gitwildmatch", gitignore_content.splitlines())

    files_list = []
    for root, dirs, files in os.walk(repo_path):
        root = Path(root)
        # Remove excluded directories from the list to prevent os.walk from processing them
        dirs[:] = [d for d in dirs if not spec.match_file(root / d)]

        for file in files:
            file_path = root / file
            if not spec.match_file(file_path) and _is_text_file(file_path):
                files_list.append(file_path.relative_to(repo_path))

    return files_list


def _read_gitignore(repo_path: str | Path) -> str:
    gitignore_path = Path(repo_path) / ".gitignore"
    if not gitignore_path.is_file():
        return ""

    with open(gitignore_path, "r", encoding="utf-8") as file:
        gitignore_content = file.read()
    return gitignore_content


def _is_text_file(file_path: str | Path):
    file_mime = magic.from_file(file_path, mime=True)
    # TODO is this an exhaustive list of mime types that we want to index ?
    return file_mime.startswith("text/") or file_mime.startswith("application/json")
