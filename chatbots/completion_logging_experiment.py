"""Example of using the langchain library to generate a response to a prompt and log it to Swipy Platform."""
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


async def main() -> None:
    """Example of using the langchain library to generate a response to a prompt and log it to Swipy Platform."""
    llm_chat = ChatOpenAI(temperature=1)
    llm_result = await llm_chat.agenerate(
        [[HumanMessage(content="What would be a good company name for a company that makes colorful socks?")]]
    )
    print(llm_result.generations[0][0].text)
