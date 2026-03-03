import anthropic
import pandas as pd
import pprint
import json
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="",
)

@retry(wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(20))
def get_novelty_response(client, model_name, prompt, passage, few_shot = False):
    if model_name == "claude":
        model_id = "claude-opus-4-1-20250805"
    elif model_name == "claude-45":
        model_id = "claude-sonnet-4-5-20250929"
    filled_prompt = prompt.replace("{passage}", passage)
    message = client.messages.create(
    model=model_id,
    max_tokens=5000,
    temperature=1,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": filled_prompt
                }
            ]
        }
        ]
    )
    # pprint.pprint(filled_prompt)
    pprint.pprint(message)
    if message.stop_reason == 'refusal':
        print(message)
        return [{"error": "PROHIBITED_CONTENT"}]
    generated_text = ""
    for content_block in message.content:
        if content_block.type == "text":
            generated_text += content_block.text
    if few_shot:
        # get the fitst [{
        if "[{" in generated_text:
            generated_text = "[{" + generated_text.split("[{", 1)[1]
        elif "```json" in generated_text:
            generated_text = generated_text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(generated_text)
    else:
        if "```json" in generated_text:
            json_string = generated_text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
            if "```json" in json_string:
                jsons = json_string.split("```json")
                for js in jsons:
                    js = js.rsplit("```", 1)[0].strip()
                json_string = "[" + "\n".join(jsons) + "]"
        else:
            json_objects_only = generated_text.split("{", 1)[1]
            json_objects_only = "{" + json_objects_only.strip()
            json_string = "[" + json_objects_only.replace("}\n\n{", "},\n{") + "]"
        print("Preprocessed output: ")
        pprint.pprint(json_string)
        data = json.loads(json_string)
    return data

all_paras_df = pd.read_csv("../../all_paras.csv")
print(all_paras_df.shape)
# all_paras_df.head(2)

# FEW_SHOT = True
FEW_SHOT = False
# MODEL_NAME = "claude"
MODEL_NAME = "claude-45"
SAVE_DIR = "../../../data/llm_performance/fs/gens/novelty"
assert os.path.exists(SAVE_DIR)
if FEW_SHOT:
    save_path = f"{SAVE_DIR}/novelty_{MODEL_NAME}_fewshot_responses.json"
    # assert parent dir exists
    assert os.path.exists("../../../data/llm_performance/fs/gens/novelty/")
    with open("novelty_few_shot_para_ids.txt", "r") as f:
        paras_for_few_shot = {line.strip() for line in f.readlines()}
    print(f"Few-shot ids: {paras_for_few_shot}")
    with open("novelty_fewshot_prompt.txt", "r") as f:
        prompt = f.read()
else:
    paras_for_few_shot = set()
    save_path = f"{SAVE_DIR}/novelty_{MODEL_NAME}_responses.json"
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
    resp = get_novelty_response(client, MODEL_NAME, prompt, row['para'], few_shot=FEW_SHOT)
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