"""
This chatbot is based on materials from the following video by Prompt Engineering YouTube channel
(checkout video description for more links): https://www.youtube.com/watch?v=TLf90ipMzfE
"""
import itertools
import os
from pathlib import Path
from typing import Any

from PyPDF2 import PdfReader
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
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
        await SwipyBot(self.swipy_bot_token).run_fulfillment_client(self._fulfillment_handler)

    async def _fulfillment_handler(self, bot: SwipyBot, data: dict[str, Any]) -> None:
        """Handle fulfillment requests from Swipy Platform."""
        print("USER:", data["message"]["content"])
        print()

        query = self._build_query(data)

        llm_chat = ChatOpenAI(user=data["user_uuid"])
        chain = load_qa_chain(llm_chat, chain_type="stuff")
        docs = self.vector_store.similarity_search(query)
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
        await bot.send_message(text=response)

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


def pdf_to_faiss(pdf_filename: str) -> FAISS:
    """Ingest a PDF and return a FAISS instance."""
    reader = PdfReader(pdf_filename)
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

    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts, embeddings)


def repo_to_faiss(repo_path: str) -> FAISS:
    """Ingest a git repository and return a FAISS instance."""
    filepaths = _list_files_in_repo(repo_path)
    print()
    for filepath in filepaths:
        print(filepath)
    print()
    print("TOTAL:", len(filepaths), "FILES")
    print()
    print("================================================================================")
    print()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = []
    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as file:
            raw_text = file.read()

        texts.extend(text_splitter.split_text(raw_text))

    for text in texts:
        print(text)
        print()
        print("================================================================================")
        print()

    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts, embeddings)


def _list_files_in_repo(repo_path: str | Path) -> list[Path]:
    repo_path = Path(repo_path)

    gitignore_content = ".*\n" + _read_gitignore(repo_path)
    spec = pathspec.PathSpec.from_lines("gitwildmatch", gitignore_content.splitlines())

    files_list = []
    for root, dirs, files in os.walk(repo_path):
        root = Path(root)
        # Remove excluded directories from the list to prevent os.walk from processing them
        dirs[:] = [d for d in dirs if not spec.match_file(root / d)]

        for file in files:
            file_path = root / file
            if not spec.match_file(file_path):
                files_list.append(file_path.relative_to(repo_path))

    return files_list


def _read_gitignore(repo_path: str | Path) -> str:
    gitignore_path = os.path.join(repo_path, ".gitignore")
    if not os.path.isfile(gitignore_path):
        return ""

    with open(gitignore_path, "r", encoding="utf-8") as file:
        gitignore_content = file.read()
    return gitignore_content
