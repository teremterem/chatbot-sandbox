"""A chatbot that answers questions about a repo, a PDF document etc."""
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Any

import promptlayer.prompts
from langchain import LLMChain
from langchain.callbacks import AsyncCallbackManager
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.question_answering import stuff_prompt
from langchain.chat_models import PromptLayerChatOpenAI, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseMessage, ChatMessage, HumanMessage, AIMessage, SystemMessage
from langchain.vectorstores import VectorStore

from chatbots.langchain_customizations import (
    load_swipy_refine_chain,
    load_swipy_stuff_chain,
    ThinkingCallbackHandler,
    SwipyConversationalRetrievalChain,
)
from swipy_client import SwipyBot


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
            # TODO download prompt from PromptLayer only once ? or just hardcode the prompt when done experimenting ?
            prompt=promptlayer.prompts.get("condense_question_prompt", langchain=True),
            verbose=True,
            callback_manager=AsyncCallbackManager([ThinkingCallbackHandler(self.swipy_bot)]),
        )
        qna = SwipyConversationalRetrievalChain(
            swipy_bot=self.swipy_bot,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            combine_docs_chain=self.load_doc_chain(chat_llm),
            question_generator=condense_question_chain,
            get_chat_history=_get_chat_history,
        )

        chat_history = _openai_msg_history_to_langchain(data["message_history"])
        result = await qna.acall({"question": data["message"]["content"], "chat_history": chat_history})
        return result

    async def fulfillment_handler(self, _, data: dict[str, Any]) -> None:
        """Handle fulfillment requests from Swipy Platform."""
        print("HUMAN:")
        pprint(data)
        print()

        chat_llm = PromptLayerChatOpenAI(
            model_name="gpt-4" if self.use_gpt4 else "gpt-3.5-turbo",
            temperature=0,
            stop=[
                "\n\nHUMAN:",
                "\n\nASSISTANT:",
            ],
            request_timeout=300,
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
        # TODO download prompt from PromptLayer only once ? or just hardcode the prompt when done experimenting ?
        stuff_chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(stuff_prompt.system_template),
                HumanMessagePromptTemplate(prompt=promptlayer.prompts.get("stuff_prompt_question", langchain=True)),
            ]
        )
        doc_chain = load_swipy_stuff_chain(
            chat_llm,
            self.swipy_bot,
            pretty_path_prefix=self.pretty_path_prefix,
            prompt=stuff_chat_prompt,
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


_ROLE_MAP = {"human": "HUMAN", "ai": "ASSISTANT"}


def _get_chat_history(chat_history: list[BaseMessage]) -> str:
    parts = []
    for dialogue_turn in chat_history:
        role_prefix = _ROLE_MAP.get(dialogue_turn.type, dialogue_turn.type)
        parts.append(f"{role_prefix.upper()}: {dialogue_turn.content}")
    return "\n\n".join(parts)
