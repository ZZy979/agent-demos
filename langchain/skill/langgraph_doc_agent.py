import requests
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from markdownify import markdownify

ALLOWED_DOMAINS = ['https://langchain-ai.github.io/', 'https://docs.langchain.com']


@tool
def fetch_documentation(url: str) -> str:
    """Fetch and convert documentation from a URL"""
    if not any(url.startswith(domain) for domain in ALLOWED_DOMAINS):
        return f"Error: URL not allowed. Must start with one of: {', '.join(ALLOWED_DOMAINS)}"
    response = requests.get(url, timeout=10.0)
    return markdownify(response.text)


model = init_chat_model('deepseek-v4-flash', extra_body={'thinking': {'type': 'disabled'}})
backend = FilesystemBackend(root_dir='.', virtual_mode=True)
agent = create_deep_agent(
    model=model,
    tools=[fetch_documentation],
    backend=backend,
    skills=['skills/']
)

result = agent.invoke(
    {'messages': [{'role': 'user', 'content': 'What is LangGraph?'}]},
    config={'configurable': {'thread_id': '1'}},
)
for msg in result['messages']:
    print(f'{msg.type}: {msg.content}')
