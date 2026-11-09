# 2026-11-08 — Vertex AI: Google Gen AI SDK basics

**Notebook:** [genai-sdk-practice-1.ipynb](genai-sdk-practice-1.ipynb)
**Environment:** Vertex AI Workbench, Python 3.12
**Model:** `gemini-3.1-flash-lite`, location `global`
**SDK:** `google-genai` (`from google import genai`)

## Summary

My first hands-on day with the Google Gen AI SDK on Vertex AI. I connected a client to my GCP project and tried five ways of calling Gemini: multi-turn chat, a cheap health check, structured JSON output, Google Search grounding, and function calling with my own function declaration.

## What I learned

### 1. Connecting the client
```python
from google import genai
client = genai.Client(vertexai=True, project=project, location="global")
```
`vertexai=True` sends requests through Vertex AI in my GCP project, using the instance's credentials, with no API key. On Workbench, `google.auth.default()` returns the project ID, so it doesn't need to be hard-coded.

### 2. Multi-turn chat
```python
chat = client.chats.create(model=MODEL_ID)
chat.send_message("...")
chat.get_history()
```
The chat object keeps the conversation history, so follow-up messages have context.

### 3. Checking the client works without spending tokens
I wanted a way to confirm the SDK is ready each time I start the instance without paying for a full generation. `count_tokens` works:
```python
client.models.count_tokens(model=MODEL_ID, contents="test")
# -> Client is ready! Token count: 1
```
It goes through authentication, the project, the region and the model name, but generates nothing.

### 4. Structured output with Pydantic
Pass a Pydantic model as `response_schema` with `response_mime_type="application/json"`, and the model returns JSON in that shape. I combined it with a `system_instruction` that gives the model a persona (a tour guide who recommends lesser-known places).

### 5. Built-in tool: Google Search grounding
```python
tools = [Tool(google_search=types.GoogleSearch())]
```
This lets the model search the web before answering. It correctly pointed out that Hugging Face uses "likes", not "stars".

### 6. Function calling with a custom `FunctionDeclaration`
I defined a `hugging_face_models` function with a JSON schema (an array of objects with `model_name`, `description`, `date_developed`). The model doesn't run the function. It returns arguments that match the schema, which I read with `response.function_calls[0].args`.

## Gotchas

- **404 NOT_FOUND: "Publisher model ... was not found".** Older model names (`gemini-2.0-flash-lite`, `gemini-1.5-flash`) in `us-central1` weren't available to my project. Switching to `gemini-3.1-flash-lite` with location `global` fixed it. The 404 outputs in the notebook are left over from those earlier runs; I didn't re-run those cells after the fix.
- **Function-call arguments aren't facts.** Without a real data source, the model made up the values: it said Llama 2 was "developed by OpenAI" and gave wrong dates. Function calling only shapes the output. For real data, my code has to run the function (or the model needs grounding) and pass the results back.
- **Empty `description` on the function declaration.** The model uses the description to decide when and how to call the function, so it should always be filled in.
- **`WARNING: Ignoring invalid distribution ~oogle-auth`** during `pip install` comes from an earlier interrupted install. It's harmless.

## Next time

- [ ] Re-run the chat and structured-output cells with the working model and check the JSON output.
- [ ] Finish function calling: actually run the function, send the result back with `Part.from_function_response`, and get a final answer.
- [ ] Try the features I imported but didn't use yet: `SafetySetting` / `HarmCategory`, `EmbedContentConfig` (embeddings), and `CreateCachedContentConfig` (context caching).
