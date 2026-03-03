import pandas as pd
import pprint
import json
import os
from generation_utils import gen_openai, gen_gemini, gen_claude
import argparse

all_paras_df = pd.read_csv("../../all_paras.csv")
print(all_paras_df.shape)
all_paras_df.head(2)

parser = argparse.ArgumentParser(description="Run pragmatic LLM evaluation")
parser.add_argument("--fs", action="store_true", help="Enable few-shot mode")
parser.add_argument(
    "--model",
    choices=["gpt5", "gpt51", "gemini", "gemini-3", "claude", "claude-45"],
    default="gpt5",
    help="Which model backend to use",
)
parser.add_argument("--save_dir", type=str, 
                    default="../../../data/llm_performance/fs/gens/prag", 
                    help="Directory to save results")
args = parser.parse_args()
SAVE_DIR = args.save_dir
assert os.path.exists(SAVE_DIR)
FEW_SHOT = args.fs
MODEL_NAME = args.model

if MODEL_NAME in ["gpt5", "gpt51"]:
    from openai import OpenAI
    api_key = ''
    client = OpenAI(api_key = api_key)
elif MODEL_NAME in ["gemini", "gemini-3"]:
    from google import genai
    client = genai.Client(
        api_key="",
    )
elif MODEL_NAME in ["claude", "claude-45"]:
    import anthropic
    client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=""
    )

if FEW_SHOT:
    save_path = f"{SAVE_DIR}/pragmatic_{MODEL_NAME}_fewshot_responses.json"
    with open("prag_few_shot_para_ids.txt", "r") as f:
        paras_for_few_shot = {line.strip() for line in f.readlines()}
    print(f"Few-shot ids: {paras_for_few_shot}")
    with open("pragmatic_fewshot_prompt.txt", "r") as f:
        prompt = f.read()
else:
    paras_for_few_shot = set()
    save_path = f"{SAVE_DIR}/pragmatic_{MODEL_NAME}_responses.json"
    with open("pragmatic_prompt.txt", "r") as f:
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
    if MODEL_NAME in ["gpt5", "gpt51"]:
        resp = gen_openai(client, MODEL_NAME, prompt, row['para'])
    elif MODEL_NAME in ["gemini", "gemini-3"]:
        resp = gen_gemini(client, MODEL_NAME, prompt, row['para'])
    elif MODEL_NAME in ["claude", "claude-45"]:
        resp = gen_claude(client, MODEL_NAME, prompt, row['para'])
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