import os
import wandb
from openai import OpenAI

# 移除所有不必要的代理变量
os.environ.pop('ALL_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ.pop('SOCKS_PROXY', None)
os.environ.pop('socks_proxy', None)

# 设置 HTTP/HTTPS 代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'https://127.0.0.1:7890'

# 打印环境变量以确认
print("HTTP_PROXY:", os.environ.get('HTTP_PROXY'))
print("HTTPS_PROXY:", os.environ.get('HTTPS_PROXY'))
print("ALL_PROXY:", os.environ.get('ALL_PROXY'))
print("SOCKS_PROXY:", os.environ.get('SOCKS_PROXY'))

import httpx

proxies = {
    "http://": "http://127.0.0.1:7890",
    "https://": "http://127.0.0.1:7890",
}

# 测试请求
try:
    url = "http://openai.com"
    response = httpx.get(url, proxies=proxies)
    print(response.status_code)
except httpx.InvalidURL as e:
    print(f"Invalid URL error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

import os
import wandb

os.environ['OPENAI_API_BASE'] = "https://api.openai.com/v1"
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'https://127.0.0.1:7890'
#os.environ['OPENAI_API_KEY'] = 'sk-proj-FiIig4OafMQgY2WD7CiiT3BlbkFJQ0RPHMGoSOrtknAeZV2C'
os.environ['WANDB_API_ENTITY']='61037bbd5a5cb503d3646e2868c2c2d4e2b721eb'
os.environ['OPENAI_API_KEY'] ='sk-proj-asxX9MfDsQ5Fez0VtRDwT3BlbkFJ8J6bL1PfOOgtLUpTkcED'
CLIENT = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
# api = wandb.Api()
# api.entity = os.environ['WANDB_API_ENTITY']
def oai_predict(prompt):
    """Predict with GPT-4 model."""

    if isinstance(prompt, str):
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = prompt

    output = CLIENT.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        max_tokens=200,
    )
    response = output.choices[0].message.content
    return response
print(oai_predict("hello"))


