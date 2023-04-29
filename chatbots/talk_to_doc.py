"""A chatbot that answers questions about a repo, a PDF document etc."""
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Any

from langchain import LLMChain
from langchain.callbacks import AsyncCallbackManager
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chat_models import PromptLayerChatOpenAI, ChatOpenAI
from langchain.schema import BaseMessage, ChatMessage, HumanMessage, AIMessage, SystemMessage
from langchain.vectorstores import VectorStore

from chatbots.langchain_customizations import load_swipy_refine_chain, load_swipy_stuff_chain
from swipy_client import SwipyBot


class ThinkingCallbackHandler(AsyncCallbackHandler):
    """A callback handler that sends a message when the bot is thinking."""

    def __init__(self, swipy_bot: SwipyBot) -> None:
        self.swipy_bot = swipy_bot

    async def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        await self.swipy_bot.send_message(
            text="_Thinking..._",
            parse_mode="Markdown",
            disable_notification=True,
            disable_web_page_preview=True,
            is_visible_to_bot=False,
            keep_typing=True,
        )


class ConvRetrievalBot(ABC):
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

    @abstractmethod
    def load_doc_chain(self, chat_llm: ChatOpenAI) -> BaseCombineDocumentsChain:
        """Load a chain that combines documents."""

    async def run_llm_chain(self, chat_llm: ChatOpenAI, data: dict[str, Any]) -> dict[str, Any]:
        """Build LLM chain and run it on a message."""
        condense_question_chain = LLMChain(
            llm=chat_llm,
            prompt=CONDENSE_QUESTION_PROMPT,
            verbose=False,
            callback_manager=AsyncCallbackManager([ThinkingCallbackHandler(self.swipy_bot)]),
        )
        qna = ConversationalRetrievalChain(
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            combine_docs_chain=self.load_doc_chain(chat_llm),
            question_generator=condense_question_chain,
        )

        chat_history = _openai_msg_history_to_langchain(data["message_history"])
        result = await qna.acall({"question": data["message"]["content"], "chat_history": chat_history})
        return result

    async def fulfillment_handler(self, _, data: dict[str, Any]) -> None:
        """Handle fulfillment requests from Swipy Platform."""
        print("USER:")
        pprint(data)
        print()

        chat_llm = PromptLayerChatOpenAI(
            model_name="gpt-4" if self.use_gpt4 else "gpt-3.5-turbo",
            temperature=0,
            user=data["user_uuid"],
            pl_tags=[f"ff{data['fulfillment_id']}"],
        )
        result = await self.run_llm_chain(chat_llm, data)

        print("ASSISTANT:")
        pprint(result)
        print()
        # TODO support parse_mode="Markdown" somehow ? Some kind of Markdown escaping is needed for that ?
        await self.swipy_bot.send_message(text=result["answer"])

    async def run_fulfillment_client(self) -> None:
        """Run the fulfillment client."""
        # TODO get rid of copy-paste in the two fulfillment handlers
        # await self.swipy_bot.run_fulfillment_client(self.refine_fulfillment_handler)
        await self.swipy_bot.run_fulfillment_client(self.fulfillment_handler)


class StuffConvRetrievalBot(ConvRetrievalBot):
    """Conversational Retrieval Bot that uses "stuff" pattern."""

    def load_doc_chain(self, chat_llm: ChatOpenAI) -> BaseCombineDocumentsChain:
        doc_chain = load_swipy_stuff_chain(
            chat_llm,
            self.swipy_bot,
            pretty_path_prefix=self.pretty_path_prefix,
            verbose=False,
        )
        return doc_chain


class RefineConvRetrievalBot(ConvRetrievalBot):
    """Conversational Retrieval Bot that uses "refine" pattern."""

    def load_doc_chain(self, chat_llm: ChatOpenAI) -> BaseCombineDocumentsChain:
        doc_chain = load_swipy_refine_chain(
            chat_llm,
            self.swipy_bot,
            pretty_path_prefix=self.pretty_path_prefix,
            verbose=False,
        )
        return doc_chain


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
