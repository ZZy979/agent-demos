import random

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    res = random.choices(['晴天', '多云', '下雨'], weights=[80, 15, 5], k=1)[0]
    return city + res


agent = create_agent(
    model='deepseek-chat',
    tools=[get_weather],
    system_prompt='你是一个天气预报助手',
)

# 运行agent
result = agent.invoke(
    {'messages': [HumanMessage('北京的天气怎么样？')]}
)
for msg in result['messages']:
    print(f'{msg.type}: {msg.content}')
