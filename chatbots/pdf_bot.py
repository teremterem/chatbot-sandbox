"""
This chatbot is based on materials from the following video by Prompt Engineering YouTube channel
(checkout video description for more links): https://www.youtube.com/watch?v=TLf90ipMzfE
"""
import os
from typing import Any

from PyPDF2 import PdfReader
from langchain import FAISS, OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import ChatMessage
from langchain.text_splitter import CharacterTextSplitter

from swipy_client import SwipyBot

ANTI_SWIPY_BOT_TOKEN = os.environ["ANTI_SWIPY_BOT_TOKEN"]


async def pdf_bot_handler(bot: SwipyBot, data: dict[str, Any]) -> None:
    """Handle fulfillment requests from Swipy Platform."""
    message = data["message"]
    print("USER:", message["content"])

    llm_context = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in data["message_history"]]
    llm_context.append(ChatMessage(role=message["role"], content=message["content"]))

    llm_chat = OpenAI(user=data["user_uuid"])
    chain = load_qa_chain(llm_chat, chain_type="stuff")
    docs = docsearch.similarity_search(message)
    response = chain.run(input_documents=docs, question=message)

    print("BOT:", response)
    await bot.send_message(text=response)


def ingest_pdf(filename: str) -> FAISS:
    """Ingest a PDF and return a FAISS instance."""
    reader = PdfReader(filename)
    raw_text_parts = [page.extract_text() for page in reader.pages]
    raw_text = "\n".join(raw_text_parts)
    print()
    print()
    print(raw_text)
    print()
    print()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts, embeddings)


docsearch = ingest_pdf("2023_GPT4All_Technical_Report.pdf")


async def main() -> None:
    """Example of using the langchain library to generate a response to a prompt and log it to Swipy Platform."""
    await SwipyBot(ANTI_SWIPY_BOT_TOKEN).run_fulfillment_client(pdf_bot_handler)
