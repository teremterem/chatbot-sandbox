"""Utilities for ingesting data into LangChain."""
import os
from pathlib import Path

import magic
from PyPDF2 import PdfReader
from langchain import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from pathspec import pathspec

from chatbots.langchain_customizations import SwipyCodeTextSplitter


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


def repo_to_faiss(  # pylint: disable=too-many-arguments
    repo_path: str | Path,
    save_to_dir: str | Path,
    chunk_size: int,
    chunk_overlap: int,
    source_url_base: str,
    additional_gitignore_content: str = "",
) -> None:
    """Ingest a git repository and return a FAISS instance."""
    # pylint: disable=too-many-locals
    # TODO log all problems to a file, not just print them

    repo_path = Path(repo_path).resolve()
    save_to_dir = Path(save_to_dir).resolve()
    source_url_base = source_url_base.strip()
    source_url_base = source_url_base.rstrip("/")

    print()
    print("REPO:", repo_path)
    print()
    print("================================================================================")
    print()

    filepaths = _list_files_in_repo(repo_path, additional_gitignore_content)
    text_splitter = SwipyCodeTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents: list[Document] = []

    save_to_dir.mkdir(parents=True, exist_ok=True)
    with open(save_to_dir / "snippet_log.txt", "w", encoding="utf-8") as snippet_file:
        for filepath in filepaths:
            # print(filepath, end=" - ")

            with open(repo_path / filepath, "rb") as file:
                raw_bytes = file.read()
            detected_encoding = "utf-8"  # chardet.detect(raw_bytes).get("encoding") or "utf-8"
            # print(detected_encoding)

            try:
                raw_text = raw_bytes.decode(detected_encoding)
            except UnicodeDecodeError as exc:
                print(f"ERROR! SKIPPING A FILE! {filepath} - {exc}")
            else:
                text_snippets = text_splitter.split_text(raw_text)
                for snippet_idx, text_snippet in enumerate(text_snippets):
                    filepath_posix = filepath.as_posix()

                    source = filepath_posix
                    if source_url_base:
                        source = f"{source_url_base}/{filepath_posix}"

                    text_snippet = f"{source} - SNIPPET {snippet_idx + 1}/{len(text_snippets)}:\n" f"{text_snippet}"

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
                    snippet_file.write(text_snippet)
                    snippet_file.write(
                        "\n================================================================================\n"
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
    faiss.save_local(str(save_to_dir))

    print("DONE")
    print()


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
