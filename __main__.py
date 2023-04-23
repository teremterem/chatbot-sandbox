"""Entry point for the chatbot sandbox."""
# pylint: disable=wrong-import-position
import asyncio

from dotenv import load_dotenv

load_dotenv()

from swipy_client import patch_openai

patch_openai()

from chatbots.pdf_bot import main

if __name__ == "__main__":
    asyncio.run(main())
