"""A chatbot that answers questions about a repo, a PDF document etc."""
import os
from pathlib import Path
from pprint import pprint
from typing import Any

import chardet
import magic
from PyPDF2 import PdfReader
from langchain import FAISS
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document, BaseMessage, ChatMessage, HumanMessage, AIMessage, SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import VectorStore
from pathspec import pathspec

from chatbots.langchain_customizations import load_swipy_conv_retrieval_chain
from swipy_client import SwipyBot


class TalkToDocBot:
    """A chatbot that answers questions about a repo, a PDF document etc."""

    def __init__(
        self,
        swipy_bot: SwipyBot,
        vector_store: VectorStore,
        use_gpt4: bool = False,
        pretty_path_prefix: str = "",
    ) -> None:
        self.swipy_bot = swipy_bot
        self.vector_store = vector_store
        self.use_gpt4 = use_gpt4
        self.pretty_path_prefix = pretty_path_prefix

    async def refine_fulfillment_handler(self, bot: SwipyBot, data: dict[str, Any]) -> None:
        """Handle fulfillment requests from Swipy Platform."""
        print("USER:")
        pprint(data)
        print()

        llm_chat = PromptLayerChatOpenAI(
            model_name="gpt-4" if self.use_gpt4 else "gpt-3.5-turbo",
            temperature=0,
            user=data["user_uuid"],
            pl_tags=[f"ff{data['fulfillment_id']}"],
        )
        qna = load_swipy_conv_retrieval_chain(
            llm_chat,
            self.vector_store.as_retriever(search_kwargs={"k": 4}),
            bot,
            pretty_path_prefix=self.pretty_path_prefix,
            verbose=False,
        )

        chat_history = _openai_msg_history_to_langchain(data["message_history"])
        result = await qna.acall({"question": data["message"]["content"], "chat_history": chat_history})

        print("ASSISTANT:")
        pprint(result)
        print()
        await bot.send_message(
            text=result["answer"],
            # parse_mode="Markdown",
        )

    async def run_fulfillment_client(self) -> None:
        """Run the fulfillment client."""
        await self.swipy_bot.run_fulfillment_client(self.refine_fulfillment_handler)


def get_embeddings() -> Embeddings:
    """Build LangChain's Embeddings object."""
    return OpenAIEmbeddings(allowed_special="all")


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

    embeddings = get_embeddings()
    return FAISS.from_texts(texts, embeddings)


def repo_to_faiss(
    repo_path: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    additional_gitignore_content: str = "",
    source_url_base: str = None,
) -> FAISS:
    """Ingest a git repository and return a FAISS instance."""
    # pylint: disable=too-many-locals
    repo_path = Path(repo_path).resolve()
    if source_url_base:
        source_url_base = source_url_base.strip()
        source_url_base = source_url_base.rstrip("/")

    print()
    print("REPO:", repo_path)
    print()
    print("================================================================================")
    print()

    filepaths = _list_files_in_repo(repo_path, additional_gitignore_content)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    documents: list[Document] = []
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
            text_snippets = text_splitter.split_text(raw_text)
            for snippet_idx, text_snippet in enumerate(text_snippets):
                filepath_posix = filepath.as_posix()

                source = filepath_posix
                if source_url_base:
                    source = f"{source_url_base}/{filepath_posix}"

                documents.append(
                    Document(
                        page_content=text_snippet,
                        metadata={
                            "source": source,
                            "path": filepath_posix,
                            "snippet_idx": snippet_idx,
                            "snippets_total": len(text_snippets),
                        },
                    )
                )

    print()
    print("================================================================================")
    print()

    print()
    print(len(filepaths), "FILES")
    print(len(documents), "SNIPPETS")
    print()
    print("INDEXING...")

    embeddings = get_embeddings()
    faiss = FAISS.from_documents(documents, embeddings)

    print("DONE")
    print()
    return faiss


def _list_files_in_repo(repo_path: str | Path, additional_gitignore_content: str) -> list[Path]:
    repo_path = Path(repo_path)

    # ".*\n" means skip all "hidden" files and directories too
    gitignore_content = f".*\n{additional_gitignore_content}\n{_read_gitignore(repo_path)}"
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


def _openai_msg_history_to_langchain(message_history: list[dict[str, str]]) -> list[BaseMessage]:
    """Convert OpenAI's message history format to LangChain's format."""
    langchain_history = []
    for message in message_history:
        if message["role"] == "user":
            langchain_history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            langchain_history.append(AIMessage(content=message["content"]))
        elif message["role"] == "system":
            langchain_history.append(SystemMessage(content=message["content"]))
        else:
            langchain_history.append(ChatMessage(role=message["role"], content=message["content"]))

    return langchain_history
