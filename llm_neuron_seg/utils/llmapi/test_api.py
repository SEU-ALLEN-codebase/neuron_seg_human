import os
import wandb

import openai
import logging
from openai import OpenAI
from data_test2 import data
print(data)
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
# os.environ['OPENAI_API_BASE'] = "https://www.gptapi.us/v1"
# os.environ['OPENAI_API_KEY'] = 'sk-eaoLQwfe7CdoQ3ia97C309655a9847CaB50bD4DfDbE02bF6'
os.environ['OPENAI_API_BASE'] = "https://api.openai.com/v1"

os.environ['WANDB_API_ENTITY']='61037bbd5a5cb503d3646e2868c2c2d4e2b721eb'
os.environ['OPENAI_API_KEY'] ='sk-proj-asxX9MfDsQ5Fez0VtRDwT3BlbkFJ8J6bL1PfOOgtLUpTkcED'
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.environ['OPENAI_API_KEY'],
    base_url=os.environ['OPENAI_API_BASE']
)
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hi, who are you?"}]
)
print(completion)
api = wandb.Api()
api.entity = os.environ['WANDB_API_ENTITY']
