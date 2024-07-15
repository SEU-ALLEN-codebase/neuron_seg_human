import openai
import time
import sys
import io
import requests

class OutputCapturer:
    def __enter__(self):
        self.new_stdout = io.StringIO()
        self.new_stderr = io.StringIO()
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.new_stdout
        sys.stderr = self.new_stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.output = self.new_stdout.getvalue()
        self.errors = self.new_stderr.getvalue()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

def send_output_to_llm(output, errors, api_url, api_key):
    payload = {
        "output": output,
        "errors": errors,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

def outputcatch():
    api_url = "https://example.com/api/llm"  
    api_key = "your_api_key"  

    with OutputCapturer() as capturer:
        # 在这里执行你的代码
        print("这是一条标准输出消息")
        print("这是一条错误消息", file=sys.stderr)

    output = capturer.output
    errors = capturer.errors

    print("捕获到的输出：", output)
    print("捕获到的错误：", errors)

    # 发送捕获的输出到 LLM
    response = send_output_to_llm(output, errors, api_url, api_key)
    print("LLM API 响应：", response)

if __name__ == "__main__":
    main()

def ai(question:str):
  openai.api_base = "https://api.openai.com/v1"
  # openai.api_key = "sk-xxxxxxxxxxxx"
  # openai.api_key = "sk - dN38iZueS69gm1iAiGzbT3BlbkFJStfihtVDXlvLq8uZxamH"
#   openai.api_key = "sk - v2Yh0lpIQgtSwzeVwgUmT3BlbkFJbBt27TbwhpR5yo50KxyD"
  openai.api_key ="sk-proj-asxX9MfDsQ5Fez0VtRDwT3BlbkFJ8J6bL1PfOOgtLUpTkcED"


  model = "gpt-3.5-turbo"
  response = openai.ChatCompletion.create(
    model = model,
    messages = [
      {'role': 'system', 'content': "你是一名开发者"}, # 给gpt定义一个角色，也可以不写
      {'role': 'user', 'content': question} # 问题
    ],
    temperature = 0,
    stream = True
  )

  collected_chunks = []
  collected_messages = []

  print(f"OpenAI({model}) :  ",end="")
  for chunk in response:
    time.sleep(0.1)
    message = chunk["choices"][0]["delta"].get("content","")
    print(message,end="")

    collected_chunks.append(chunk)

    chunk_message = chunk["choices"][0]["delta"]
    collected_messages.append(chunk_message)

  # full_reply_content = ''.join([m.get("content","") for m in collected_messages])
  # print(full_reply_content)

if __name__ == '__main__':
  print("hello")
  while True:
    question = input("what's weather like today in nj")
    startTime = time.time()

    # 请求
    ai(question)

    print("耗时:",time.time()-startTime)

    def txt_prompt_process(prompt1):
    llm_prompt = prompt1 + "进行任务解决规划，并将你的思路分为任务目标及功能描述、算法步骤及实现细节、特定约束三部分"
    return str(llm_prompt)

def txt_response_process(txt_prompt):
    response = get_llm_text(txt_prompt)
    txt1 = process1(response)
    txt2 = process2(response)
    txt3 = process3(response)

    return txt1,txt2,txt3
    
def code_prompt_process(prompt2,option):
    code_prompt = prompt2+f'进行基于{option}的脚本撰写'
    return str(code_prompt)