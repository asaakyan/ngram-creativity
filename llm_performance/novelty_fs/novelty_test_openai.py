import pandas as pd
import pprint
import json
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from openai import OpenAI
api_key = ''
client = OpenAI(api_key = api_key)

@retry(wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10))
def get_novelty_response(client, model_name, prompt, passage):
  filled_prompt = prompt.replace("{passage}", passage)
  if model_name == "gpt5":
    model_id = "gpt-5-2025-08-07"
    reasoning_effort = "high"
    verbosity = "high"
  elif model_name == 'gpt51':
    # model_id = "gpt-5.1-chat-latest"
    # reasoning_effort = "medium"
    # verbosity = "medium"
    model_id = "gpt-5.1-2025-11-13"
    reasoning_effort = "high"
    verbosity = "high"
  else:
    raise ValueError(f"Unknown model name: {model_name}")
  response = client.responses.create(
    model=model_id,
    input=[
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": filled_prompt
          }
        ]
      }
    ],
    text={
      "format": {
        "type": "text"
      },
      "verbosity": verbosity
    },
    reasoning={
      "effort": reasoning_effort
    },
    tools=[],
    store=True,
  )
  print(response)
  resp_json = json.loads(response.output_text)
  return resp_json

FEW_SHOT = True
# MODEL_NAME = "gpt5"
MODEL_NAME = "gpt51"

all_paras_df = pd.read_csv("../../all_paras.csv")
print(all_paras_df.shape)
all_paras_df.head(2)

save_dir = "../../../data/llm_performance/fs/gens/novelty"
assert os.path.exists(save_dir)
if FEW_SHOT:
    save_path = f"{save_dir}/novelty_{MODEL_NAME}_fewshot_responses.json"
    with open("novelty_few_shot_para_ids.txt", "r") as f:
        paras_for_few_shot = {line.strip() for line in f.readlines()}
    print(f"Few-shot ids: {paras_for_few_shot}")
    with open("novelty_fewshot_prompt.txt", "r") as f:
        prompt = f.read()
else:
    paras_for_few_shot = set()
    save_path = f"{save_dir}/novelty_{MODEL_NAME}_responses.json"
    with open("novelty_prompt.txt", "r") as f:
        prompt = f.read()
print(prompt)

if os.path.exists(save_path):
    with open(save_path, "r") as f:
        data = json.load(f)
else:
    data = []
already_done_ids = {item['id'] for item in data}
print(f"Already done {len(already_done_ids)} items")

for i, row in all_paras_df.iterrows():
    print(f"Processing row {i}")
    if row['id'] in already_done_ids:
        print("Already done, skipping")
        continue
    if row['id'] in paras_for_few_shot:
        print("Few-shot example, skipping")
        continue
    resp = get_novelty_response(client, MODEL_NAME, prompt, row['para'])
    # resp = 'test'
    pprint.pprint(resp)
    data.append({
        'id': row['id'],
        'para': row['para'],
        'response': resp
    })
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)    
    print("***"*10)
    # break