"""Customizations to the LangChain library for Swipy."""
# pylint: disable=too-many-arguments
from typing import Any

from langchain import BasePromptTemplate, LLMChain
from langchain.callbacks import BaseCallbackManager
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.chains.question_answering import refine_prompts, stuff_prompt
from langchain.schema import Document, BaseLanguageModel

from swipy_client import SwipyBot


def load_swipy_stuff_chain(
    llm: BaseLanguageModel,
    swipy_bot: SwipyBot,
    pretty_path_prefix: str = "",
    prompt: BasePromptTemplate = None,
    document_variable_name: str = "context",
    verbose: bool = None,
    callback_manager: BaseCallbackManager = None,
    **kwargs: Any,
) -> "SwipyStuffDocumentsChain":
    """
    A modification of:
    langchain/chains/question_answering/__init__.py::_load_stuff_chain()
    """
    _prompt = prompt or stuff_prompt.PROMPT_SELECTOR.get_prompt(llm)
    llm_chain = LLMChain(llm=llm, prompt=_prompt, verbose=verbose, callback_manager=callback_manager)
    return SwipyStuffDocumentsChain(
        swipy_bot=swipy_bot,
        pretty_path_prefix=pretty_path_prefix,
        llm_chain=llm_chain,
        document_variable_name=document_variable_name,
        verbose=verbose,
        callback_manager=callback_manager,
        **kwargs,
    )


def load_swipy_refine_chain(
    llm: BaseLanguageModel,
    swipy_bot: SwipyBot,
    pretty_path_prefix: str = "",
    question_prompt: BasePromptTemplate = None,
    refine_prompt: BasePromptTemplate = None,
    document_variable_name: str = "context_str",
    initial_response_name: str = "existing_answer",
    refine_llm: BaseLanguageModel = None,
    verbose: bool = None,
    callback_manager: BaseCallbackManager = None,
    **kwargs: Any,
) -> "SwipyRefineDocumentsChain":
    """
    A modification of:
    langchain/chains/question_answering/__init__.py::_load_refine_chain()
    """
    # pylint: disable=too-many-locals
    _question_prompt = question_prompt or refine_prompts.QUESTION_PROMPT_SELECTOR.get_prompt(llm)
    _refine_prompt = refine_prompt or refine_prompts.REFINE_PROMPT_SELECTOR.get_prompt(llm)
    initial_chain = LLMChain(
        llm=llm,
        prompt=_question_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
    )
    _refine_llm = refine_llm or llm
    refine_chain = LLMChain(
        llm=_refine_llm,
        prompt=_refine_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
    )
    return SwipyRefineDocumentsChain(
        swipy_bot=swipy_bot,
        pretty_path_prefix=pretty_path_prefix,
        initial_llm_chain=initial_chain,
        refine_llm_chain=refine_chain,
        document_variable_name=document_variable_name,
        initial_response_name=initial_response_name,
        verbose=verbose,
        callback_manager=callback_manager,
        **kwargs,
    )


class SwipyStuffDocumentsChain(StuffDocumentsChain):
    """A stuff chain that reports the documents that are being used to answer to the user."""

    swipy_bot: SwipyBot
    pretty_path_prefix: str = ""

    async def acombine_docs(self, docs: list[Document], **kwargs: Any) -> tuple[str, dict]:
        """Stuff all documents into one prompt and pass to LLM."""
        doc_source_set = set()
        non_duplicate_docs = []
        for doc in docs:
            if doc.metadata["source"] not in doc_source_set:
                non_duplicate_docs.append(doc)
                doc_source_set.add(doc.metadata["source"])

        doc_list = [
            f"- [{self.pretty_path_prefix}{doc.metadata['path']}]({doc.metadata['source']})"
            for doc in non_duplicate_docs
        ]
        line_separator = "\n"
        await self.swipy_bot.send_message(
            text=f"_Looking at:_\n{line_separator.join(doc_list)}",
            parse_mode="Markdown",
            disable_notification=True,
            disable_web_page_preview=True,
            is_visible_to_bot=False,
            keep_typing=True,
        )
        return await super().acombine_docs(docs, **kwargs)


class SwipyRefineDocumentsChain(RefineDocumentsChain):
    """A refine chain that reports the document it is currently processing to the user."""

    swipy_bot: SwipyBot
    pretty_path_prefix: str = ""

    async def acombine_docs(self, docs: list[Document], **kwargs: Any) -> tuple[str, dict]:
        """Combine by mapping first chain over all, then stuffing into final chain."""
        inputs = self._construct_initial_inputs(docs, **kwargs)
        await self._report_doc_processing(docs[0])  # modification
        res = await self.initial_llm_chain.apredict(**inputs)
        refine_steps = [res]
        for doc in docs[1:]:
            base_inputs = self._construct_refine_inputs(doc, res)
            inputs = {**base_inputs, **kwargs}
            await self._report_doc_processing(doc)  # modification
            res = await self.refine_llm_chain.apredict(**inputs)
            refine_steps.append(res)
        return self._construct_result(refine_steps, res)

    async def _report_doc_processing(self, doc: Document) -> None:
        await self.swipy_bot.send_message(
            text=f"_Looking at_ [{self.pretty_path_prefix}{doc.metadata['path']}]({doc.metadata['source']})",
            parse_mode="Markdown",
            disable_notification=True,
            disable_web_page_preview=True,
            is_visible_to_bot=False,
            keep_typing=True,
        )


class SwipyConversationalRetrievalChain(ConversationalRetrievalChain):
    """
    Chain for chatting with an index. Unlike the original version, this one uses the whole chat history to answer,
    and not just the "summarized" question (the "summarized" question is used only for doc retrieval).
    """

    swipy_bot: SwipyBot

    async def _acall(self, inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])
        if chat_history_str:
            new_question = await self.question_generator.arun(question=question, chat_history=chat_history_str)
        else:
            new_question = question
        docs = await self._aget_docs(new_question, inputs)
        new_inputs = inputs.copy()
        # new_inputs["question"] = new_question  # modification: use new_question only for doc retrieval
        new_inputs["chat_history"] = chat_history_str
        answer = await self.combine_docs_chain.arun(input_documents=docs, **new_inputs)
        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        return {self.output_key: answer}


class ThinkingCallbackHandler(AsyncCallbackHandler):
    """A callback handler that sends a message when the bot is thinking."""

    def __init__(self, swipy_bot: SwipyBot) -> None:
        self.swipy_bot = swipy_bot

    async def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        await self.swipy_bot.send_message(
            text="_Thinking_ ⏳",
            parse_mode="Markdown",
            disable_notification=True,
            disable_web_page_preview=True,
            is_visible_to_bot=False,
            keep_typing=True,
        )

    async def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        await self.swipy_bot.send_message(
            text=f"{outputs['text']} 🤔",
            disable_notification=True,
            disable_web_page_preview=True,
            is_visible_to_bot=False,
            keep_typing=True,
        )
