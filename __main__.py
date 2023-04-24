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
    pdf_bot = asyncio.create_task(
        PdfBot(os.environ["PDF_BOT_TOKEN"], "2023_GPT4All_Technical_Report.pdf").run_fulfillment_client()
    )
    await asyncio.gather(pdf_bot)


if __name__ == "__main__":
    asyncio.run(main())
