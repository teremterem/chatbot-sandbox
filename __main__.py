"""Entry point for the chatbot sandbox."""
# pylint: disable=wrong-import-position
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from swipy_client import patch_openai

patch_openai()

from chatbots.pdf_bot import PdfBot


async def main() -> None:
    """Run the chatbot sandbox."""
    gpt4all_pdf_bot = asyncio.create_task(
        PdfBot(os.environ["GPT4ALL_PDF_BOT_TOKEN"], "2023_GPT4All_Technical_Report.pdf").run_fulfillment_client()
    )
    instruct_gpt_pdf_bot = asyncio.create_task(
        PdfBot(os.environ["INSTRUCT_GPT_PDF_BOT_TOKEN"], "2203.02155.pdf").run_fulfillment_client()
    )
    await asyncio.gather(gpt4all_pdf_bot, instruct_gpt_pdf_bot)


if __name__ == "__main__":
    asyncio.run(main())
