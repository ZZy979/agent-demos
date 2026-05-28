from langchain.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek

# 初始化DeepSeek模型
llm = ChatDeepSeek(
    model='deepseek-v4-flash',
    temperature=0,
    extra_body={'thinking': {'type': 'disabled'}}
)

messages = [
    SystemMessage('你是一个专业的AI助手'),
    HumanMessage('介绍一下LangChain框架的主要功能'),
]

response = llm.invoke(messages)
print(response.content)
