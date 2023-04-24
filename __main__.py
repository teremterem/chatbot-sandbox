"""Entry point for the chatbot sandbox."""
# pylint: disable=wrong-import-position
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from swipy_client import patch_openai

patch_openai()

from chatbots.talk_to_doc import TalkToDocBot, pdf_to_faiss


async def main() -> None:
    """Run the chatbot sandbox."""
    await asyncio.gather(
        asyncio.create_task(
            TalkToDocBot(
                os.environ["GPT4ALL_PDF_BOT_TOKEN"],
                pdf_to_faiss("2023_GPT4All_Technical_Report.pdf"),
            ).run_fulfillment_client()
        ),
        # asyncio.create_task(
        #     TalkToDocBot(
        #         os.environ["INSTRUCT_GPT_PDF_BOT_TOKEN"],
        #         pdf_to_faiss("2203.02155.pdf"),
        #     ).run_fulfillment_client()
        # ),
    )


if __name__ == "__main__":
    asyncio.run(main())
