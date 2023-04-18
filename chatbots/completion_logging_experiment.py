"""Example of using the langchain library to generate a response to a prompt and log it to Swipy Platform."""
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


async def main() -> None:
    """Example of using the langchain library to generate a response to a prompt and log it to Swipy Platform."""
    llm_chat = ChatOpenAI(temperature=1)
    for _ in range(2):
        llm_result = await llm_chat.agenerate([[HumanMessage(content="Hi, how are you? What's your name?")]])
        print(llm_result.generations[0][0].text)
