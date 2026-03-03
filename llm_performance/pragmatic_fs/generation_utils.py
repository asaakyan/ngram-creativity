from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import json
from google.genai import types

@retry(wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(20))
def gen_openai(client, model_name, prompt, passage):
  if model_name == "gpt5":
    model_id = "gpt-5-2025-08-07"
    reasoning_effort = "high"
    verbosity = "high"
  elif model_name == "gpt51":
    model_id = "gpt-5.1-2025-11-13"
    reasoning_effort = "high"
    verbosity = "high"
  else:
    raise ValueError(f"Unknown model name: {model_name}")
  filled_prompt = prompt.replace("{passage}", passage)
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
  resp_json = json.loads(response.output_text)
  return resp_json

@retry(wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10))
def gen_gemini(client, model_name, prompt, passage):

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

@retry(wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10))
def gen_claude(client, model_name, prompt, passage):

    if model_name == "claude":
        model_id = "claude-opus-4-1-20250805"
    elif model_name == "claude-45":
        model_id = "claude-sonnet-4-5-20250929"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    filled_prompt = prompt.replace("{passage}", passage)
    # print("FILLED PROMPT:")
    # print(filled_prompt)
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
#   print(message.content)
    generated_text = ""
    for content_block in message.content:
        if content_block.type == "text":
            generated_text += content_block.text
    # find the first '{' and parse the JSON from there
    print(generated_text)
    if "[{" in generated_text:
            generated_text = "[{" + generated_text.split("[{", 1)[1].rsplit("]", 1)[0].strip() + "]"
            print("processed generated text:")
            print("-" + generated_text + "-")
    elif "```json" in generated_text:
        generated_text = generated_text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        print("processed generated text:")
        print("-" + generated_text + "-")

    data = json.loads(generated_text)

    return data