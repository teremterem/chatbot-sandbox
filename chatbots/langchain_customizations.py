"""Customizations to the langchain library for Swipy."""
# pylint: disable=too-many-arguments
from typing import Any

from langchain import BasePromptTemplate, LLMChain
from langchain.callbacks import BaseCallbackManager
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import refine_prompts
from langchain.schema import Document, BaseLanguageModel, BaseRetriever

from swipy_client import SwipyBot


def load_swipy_conv_retrieval_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
    swipy_bot: SwipyBot,
    condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
    verbose: bool = False,
    combine_docs_chain_kwargs: dict = None,
    **kwargs: Any,
) -> ConversationalRetrievalChain:
    """
    A modification of:
    langchain/chains/conversational_retrieval/base.py::ConversationalRetrievalChain.from_llm()
    """
    combine_docs_chain_kwargs = combine_docs_chain_kwargs or {}
    doc_chain = load_swipy_refine_chain(
        llm,
        swipy_bot,
        verbose=verbose,
        **combine_docs_chain_kwargs,
    )
    condense_question_chain = LLMChain(llm=llm, prompt=condense_question_prompt, verbose=verbose)
    return ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=doc_chain,
        question_generator=condense_question_chain,
        **kwargs,
    )


def load_swipy_refine_chain(
    llm: BaseLanguageModel,
    swipy_bot: SwipyBot,
    question_prompt: BasePromptTemplate = None,
    refine_prompt: BasePromptTemplate = None,
    document_variable_name: str = "context_str",
    initial_response_name: str = "existing_answer",
    refine_llm: BaseLanguageModel = None,
    verbose: bool = None,
    callback_manager: BaseCallbackManager = None,
    **kwargs: Any,
) -> RefineDocumentsChain:
    """
    A modification of:
    langchain/chains/question_answering/__init__.py::_load_refine_chain()
    """
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
        initial_llm_chain=initial_chain,
        refine_llm_chain=refine_chain,
        document_variable_name=document_variable_name,
        initial_response_name=initial_response_name,
        verbose=verbose,
        callback_manager=callback_manager,
        **kwargs,
    )


class SwipyRefineDocumentsChain(RefineDocumentsChain):
    """A refine chain that reports the document it is currently processing to the user."""

    swipy_bot: SwipyBot

    async def acombine_docs(self, docs: list[Document], **kwargs: Any) -> tuple[str, dict]:
        """Combine by mapping first chain over all, then stuffing into final chain."""
        inputs = self._construct_initial_inputs(docs, **kwargs)
        await self._report_doc_processing(docs[0])  # modification
        res = await self.initial_llm_chain.apredict(**inputs)
        refine_steps = [res]
        for doc in docs[1:]:
            base_inputs = self._construct_refine_inputs(doc, res)
            await self._report_doc_processing(doc)  # modification
            inputs = {**base_inputs, **kwargs}
            res = await self.refine_llm_chain.apredict(**inputs)
            refine_steps.append(res)
        return self._construct_result(refine_steps, res)

    async def _report_doc_processing(self, doc: Document) -> None:
        await self.swipy_bot.send_message(
            text=f"_Looking at_ [{doc.metadata['path']}]({doc.metadata['source']})",
            parse_mode="Markdown",
            disable_web_page_preview=True,
            is_visible_to_bot=False,
            keep_typing=True,
        )
