import pandas as pd
import pprint
import json
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from google import genai
from google.genai import types

client = genai.Client(
        api_key="",
    )

@retry(wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10))
def get_novelty_response(client, model_name, prompt, passage):
    if model_name == "gemini":
        model_id = "gemini-2.5-pro"
    elif model_name == "gemini-3":
        model_id = "gemini-3-pro-preview"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    filled_prompt = prompt.replace("{passage}", passage)

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=filled_prompt)
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=-1,
        ),
    )
    generated_text = ""
    for chunk in client.models.generate_content_stream(
        model=model_id,
        contents=contents,
        config=generate_content_config,
    ):
        if "PROHIBITED_CONTENT" in str(chunk.prompt_feedback):
            return [{"error": "PROHIBITED_CONTENT"}]
        if chunk.text:
            generated_text += chunk.text
        else:
            print(f"Received non-text chunk: {chunk}")
    print(generated_text)
    if "```json" in generated_text:
        generated_text = generated_text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
    resp_json = json.loads(generated_text)
    return resp_json

all_paras_df = pd.read_csv("../../all_paras.csv")
print(all_paras_df.shape)
# all_paras_df.head(2)

# FEW_SHOT = True
FEW_SHOT = False
# MODEL_NAME = "gemini"
MODEL_NAME = "gemini-3"
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
    if row['id'] in paras_for_few_shot:
        print("Few-shot example, skipping")
        continue
    if row['id'] in already_done_ids:
        print("Already done, skipping")
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