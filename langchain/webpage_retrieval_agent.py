import re

import requests
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool


@tool
def fetch_url(url: str) -> str:
    """Fetch text content from a URL"""
    response = requests.get(url, timeout=10.0)
    response.raise_for_status()
    content = re.sub(r'<script.*?>.*?</script>', ' ', response.text, flags=re.DOTALL)
    content = re.sub(r'<style.*?>.*?</style>', ' ', content, flags=re.DOTALL)
    content = re.sub(r'<.*?>', ' ', content)
    return content

system_prompt = """
Use fetch_url when you need to fetch information from a web-page; quote relevant snippets.
"""

agent = create_agent(
    model='deepseek-chat',
    tools=[fetch_url], # A tool for retrieval
    system_prompt=system_prompt,
)

result = agent.invoke({'messages': HumanMessage("""
Read the following documentation. What is RAG? What are the types of RAG architecture?
https://docs.langchain.com/oss/python/langchain/retrieval
""")})
for msg in result['messages']:
    print(f'{msg.type}: {msg.content}')
