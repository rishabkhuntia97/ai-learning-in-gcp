# 2026-11-06 — Gen AI SDK on Vertex AI: Text Prompts, System Instructions, Chat, Structured Output, Streaming, Async & Count Tokens

**Notebook:** [genai-sdk-text-prompt-system-instruction-chat-structured-output-streaming-async-count-tokens.ipynb](genai-sdk-text-prompt-system-instruction-chat-structured-output-streaming-async-count-tokens.ipynb)
**Source:** Google Cloud's official tutorial [intro_genai_sdk.ipynb](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_genai_sdk.ipynb) (Apache 2.0, by Eric Dong). I worked through it and retyped several cells myself.
**Environment:** Vertex AI Workbench, Python 3.12
**Model:** `gemini-3.5-flash-lite`, location `global`
**SDK:** `google-genai` (`from google import genai`)

## Summary

My first day with the Google Gen AI SDK, following Google's getting-started tutorial. I connected to Vertex AI from Workbench and ran seven core operations: a basic text prompt, a system instruction, a multi-turn chat, structured JSON output, streaming, an async request, and token counting.

## What I ran

### 1. Text prompt
```python
client = genai.Client(enterprise=True, project=PROJECT_ID, location="global")
response = client.models.generate_content(model=MODEL_ID, contents="What is the largest planet in our solar system?")
print(response.text)   # -> Jupiter, with mass and volume facts
```
`display(Markdown(response.text))` renders the answer as formatted Markdown instead of plain text.

### 2. System instruction
A `system_instruction` in `GenerateContentConfig` sets the model's role for the whole request. With "You are a helpful language translator... English to French", the input "I like bagels." returned **"J'aime les bagels."**

### 3. Multi-turn chat
```python
chat = client.chats.create(model=MODEL_ID, config=GenerateContentConfig(system_instruction=..., temperature=0.5))
chat.send_message("Write a function that checks if a year is a leap year.")
chat.send_message("Okay, write a unit test of the generated function.")
```
The second message says only "the generated function", and the model knew which function I meant, because the chat keeps the history. It wrote pytest tests covering 2024, 2023, 1900 and 2000.

### 4. Structured output (Pydantic)
```python
class Recipe(BaseModel):
    name: str
    description: str
    ingredients: list[str]
config=GenerateContentConfig(response_mime_type="application/json", response_schema=Recipe)
```
It returned valid JSON for "Classic Chocolate Chip Cookies". I asked for "a few" recipes but got only one, because the schema is a single `Recipe`. To get several, the schema should be `list[Recipe]`.

### 5. Streaming
`client.models.generate_content_stream(...)` returns the answer in chunks as they're generated, so I printed each chunk with a separator to see where they split. This is useful for chat UIs, where users see text appear immediately.

### 6. Async request
```python
response = await client.aio.models.generate_content(...)
```
`client.aio` has async versions of every method. Jupyter lets you use `await` directly in a cell. The squirrel time-travel song came back fine.

### 7. Count tokens
```python
client.models.count_tokens(model=MODEL_ID, contents="What's the highest mountain in Africa?")
# -> total_tokens=9
```
This counts input tokens before sending a request. It's useful for estimating cost and checking limits.

## Covered in the tutorial but not run yet

These cells have no output in my notebook:
- **Multimodal prompts:** an image via PIL, and `Part.from_uri` for an image URL
- **Model parameters:** `temperature`, `top_p`, `top_k`, `seed`, `stop_sequences`, penalties
- **Safety filters:** `SafetySetting` with `HarmCategory` / `HarmBlockThreshold`, and reading `safety_ratings`
- **JSON schema as a dict:** the product-review sentiment example with an `enum`
- **`compute_tokens`**
- **Function calling:** the `get_destination` example
- **Context caching:** create, use and delete a cache over two research PDFs
- **Text embeddings:** `embed_content` with `output_dimensionality=128`

## Gotchas

- **The client uses `enterprise=True`.** On 8 Nov I used `vertexai=True`; both connect to Vertex AI in my project.
- **Restart the kernel after `%pip install`**, or the new SDK version isn't loaded.
- **Cell 41 is unfinished:** `from pydantic import ` with nothing after it is a syntax error, so delete it before running the whole notebook. The next cell has the full import.
- **`WARNING: Ignoring invalid distribution ~oogle-auth`** during install comes from an earlier interrupted install. It's harmless.

## Next time

- [ ] Run the cells I skipped, starting with multimodal prompts and model parameters.
- [ ] Change the Recipe schema to `list[Recipe]` and confirm I get several recipes.
- [ ] Try safety filters and look at the safety ratings.
