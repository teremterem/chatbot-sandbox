"""Entry point for the chatbot sandbox."""
import asyncio

from dotenv import load_dotenv

from swipy_client import patch_openai

load_dotenv()
patch_openai()

if __name__ == "__main__":
    from chatbots.completion_logging_experiment import main

    asyncio.run(main())
