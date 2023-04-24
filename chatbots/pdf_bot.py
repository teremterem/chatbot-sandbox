"""
This chatbot is based on materials from the following video by Prompt Engineering YouTube channel
(checkout video description for more links): https://www.youtube.com/watch?v=TLf90ipMzfE
"""
from typing import Any

from PyPDF2 import PdfReader
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from swipy_client import SwipyBot


class PdfBot:
    """A chatbot that answers questions about a PDF document."""

    def __init__(self, swipy_bot_token: str, pdf_filename: str) -> None:
        self.swipy_bot_token = swipy_bot_token
        self.docsearch = self._ingest_pdf(pdf_filename)

    async def run_fulfillment_client(self):
        """Connect to Swipy Platform and listen for fulfillment requests."""
        await SwipyBot(self.swipy_bot_token).run_fulfillment_client(self._fulfillment_handler)

    async def _fulfillment_handler(self, bot: SwipyBot, data: dict[str, Any]) -> None:
        """Handle fulfillment requests from Swipy Platform."""
        message = data["message"]
        query = message["content"]
        print("USER:", query)

        # llm_context = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in data["message_history"]]
        # llm_context.append(ChatMessage(role=message["role"], content=query))

        llm_chat = ChatOpenAI(user=data["user_uuid"])
        chain = load_qa_chain(llm_chat, chain_type="stuff")
        docs = self.docsearch.similarity_search(query)
        response = await chain.arun(input_documents=docs, question=query)

        print("BOT:", response)
        print()
        await bot.send_message(text=response)

    @staticmethod
    def _ingest_pdf(pdf_filename: str) -> FAISS:
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
            print(len(text))
            print(text)
            print()
            print()

        embeddings = OpenAIEmbeddings()
        return FAISS.from_texts(texts, embeddings)
