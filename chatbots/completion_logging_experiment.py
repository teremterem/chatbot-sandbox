from dotenv import load_dotenv

load_dotenv()

from swipy_client import patch_openai

patch_openai()

from langchain.chat_models import ChatOpenAI

from langchain.schema import HumanMessage

llm_chat = ChatOpenAI()
llm_result = await llm_chat.agenerate(
    [[HumanMessage(content="What would be a good company name for a company that makes colorful socks?")]]
)
print(llm_result.generations[0][0].text)
