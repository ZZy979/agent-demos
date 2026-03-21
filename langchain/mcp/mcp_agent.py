import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    client = MultiServerMCPClient({
        'math': {
            'transport': 'stdio',  # Local subprocess communication
            'command': 'python',
            # Absolute path to your math_server.py file
            'args': ['/path/to/math_server.py'],
        },
        'weather': {
            'transport': 'http',  # HTTP-based remote server
            # Ensure you start your weather server on port 8000
            'url': 'http://localhost:8000/mcp',
        }
    })

    tools = await client.get_tools()
    agent = create_agent('deepseek-chat', tools)

    math_response = await agent.ainvoke(
        {'messages': [{'role': 'user', 'content': 'What is (3 + 5) * 12?'}]}
    )
    print('Math response:')
    for msg in math_response['messages']:
        print(f'{msg.type}: {msg.content}')

    weather_response = await agent.ainvoke(
        {'messages': [{'role': 'user', 'content': 'What is the weather in nyc?'}]}
    )
    print('\nWeather response:')
    for msg in weather_response['messages']:
        print(f'{msg.type}: {msg.content}')


if __name__ == '__main__':
    asyncio.run(main())
