import os

import requests

DEEPSEEK_API_KEY = os.environ['DEEPSEEK_API_KEY']

url = 'https://api.deepseek.com/chat/completions'
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
}

data = {
    'model': 'deepseek-chat',  # DeepSeek-V3.2的非思考模式
    'messages': [
        {'role': 'system', 'content': '你是一个专业的AI助手'},
        {'role': 'user', 'content': '请用一句话解释什么是量子计算'}
    ],
    'stream': False
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    print(result['choices'][0]['message']['content'])
else:
    print(f"请求失败，错误码：{response.status_code}")
