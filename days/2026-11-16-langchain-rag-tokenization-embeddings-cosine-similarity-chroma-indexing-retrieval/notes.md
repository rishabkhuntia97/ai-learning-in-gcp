# 2026-11-16 — LangChain RAG with Gemini on Vertex AI: Tokenization, Embeddings, Cosine Similarity, Chroma Indexing & Retrieval

**Notebook:** [langchain-rag-tokenization-embeddings-cosine-similarity-chroma-indexing-retrieval.ipynb](langchain-rag-tokenization-embeddings-cosine-similarity-chroma-indexing-retrieval.ipynb)
**Environment:** my Mac (VS Code + `.venv`, Python 3.10), Vertex AI via gcloud login (no GCP instance)
**Models:** `gemini-3.5-flash-lite` (chat), `text-embedding-005` (embeddings), location `global`
**Libraries:** `google-genai`, `langchain-google-genai`, `langchain-text-splitters`, `langchain-community` (WebBaseLoader, Chroma), `langchain-classic`, `chromadb`, `tiktoken`
**Source document:** Lilian Weng's blog post ["LLM Powered Autonomous Agents"](https://lilianweng.github.io/posts/2023-06-23-agent/)

## Summary

My first day building **RAG (Retrieval-Augmented Generation)** with LangChain. I followed a RAG-from-scratch tutorial that was written for OpenAI and moved it to Gemini on Vertex AI. I built the whole pipeline end to end first, then went back and studied each part: tokens, embeddings and similarity, then indexing and retrieval.

## Part 1: The full RAG pipeline (overview)

```
web page → load → split into chunks → embed → Chroma vector store → retriever
                                                                        ↓
question ──────────────────────────────────────────────────→ prompt + context → Gemini → answer
```

- **Load:** `WebBaseLoader` with a `bs4.SoupStrainer` keeps only the post's content, title and header.
- **Split:** `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` → **63 chunks**.
- **Embed and store:** `Chroma.from_documents(splits, GoogleGenerativeAIEmbeddings(model="text-embedding-005", vertexai=True, ...))`. The store count was **63**, the same as the number of chunks.
- **Chain** (LCEL pipe syntax):
  ```python
  rag_chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt | llm | StrOutputParser()
  )
  rag_chain.invoke("What is Task Decomposition?")
  ```
- **Answer:** "Task decomposition is a technique used to break down a complicated or hard task into smaller, simpler, and more manageable steps…" It was grounded in the blog's Chain of Thought section.
- `retriever.invoke(...)` on its own returns the matching **chunks** (Documents), not an answer. The retriever is a wrapper around the vector store's similarity search.

### Moving the tutorial from OpenAI to Gemini / LangChain 1.x

| Tutorial (old) | What I used |
|---|---|
| `ChatOpenAI`, `OpenAIEmbeddings` | `ChatGoogleGenerativeAI`, `GoogleGenerativeAIEmbeddings` with `vertexai=True` |
| `from langchain import hub` | `from langchain_classic import hub` |
| `from langchain.text_splitter import ...` | `from langchain_text_splitters import ...` |
| `hub.pull("rlm/rag-prompt")` | My own `ChatPromptTemplate`. Public hub pulls are now blocked by default. |

## Part 2: Understanding the pieces

### Tokens: tiktoken vs Gemini
For the question "What kind of pets do I like?":
- `tiktoken` with `cl100k_base` (OpenAI's tokenizer) → **8 tokens**
- Gemini `client.models.count_tokens` → **8 tokens**
- Gemini `client.models.compute_tokens` shows the actual pieces and IDs:
  `['What', ' kind', ' of', ' pets', ' do', ' I', ' like', '?']` → `[3689, 2712, 529, 26757, 776, 564, 1133, 236881]`

The two counts happen to match for this sentence. In general, each model family has its own tokenizer, so counts can differ. Use the model's own `count_tokens` for cost estimates.

### Embeddings and cosine similarity
- `embed.embed_query(text)` turns text into a **vector of 768 numbers**: a point in a 768-dimensional space where similar meanings sit close together.
- **Cosine similarity** = dot(a, b) / (‖a‖ × ‖b‖), from −1 to 1; higher means more similar.
- "What kind of pets do I like?" vs "My favorite peet is a cat." → **0.618**. They're related even though they share almost no words.

### Indexing
- I split again with a **token-based** splitter: `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)` → 1 document → **50 chunks**.
- In my own words, from the notebook: each chunk goes through the embedding model and becomes one entry in the vector store (id, text, vector, metadata). So **number of chunks = number of entries**: `len(splits) == vectorStore._collection.count()` (50 = 50).

## Part 3: Retrieval (started)

- `vectorStore.as_retriever(search_kwargs={"k": 1})` returns only the **single** most similar chunk; `len(docs)` → 1.
- `k` controls how many chunks go into the prompt as context. More chunks give more coverage but use more tokens and add more noise.

## Gotchas

- **Never put API keys in a notebook.** The LangSmith key is now in the repo's `.env` (gitignored) and loaded with `load_dotenv()`, because this repo is public.
- **`format_docs` has a typo:** `"/n/n".join(...)` should be `"\n\n".join(...)`. Right now, the chunks are glued together with a literal `/n/n` between them.
- **The Part 1 loader uses `"post_title"` (underscore)**, while Part 2 uses `"post-title"` (hyphen), which is the class name the blog uses. With the underscore, the page title is skipped.
- **The temperature warning:** "Model 'gemini-3.5-flash-lite' uses fixed sampling defaults; temperature will be ignored." On this model, `temperature=0` has no effect.
- **The `langchain-community` deprecation warning:** it's being sunset. Its loaders and Chroma integration still work for now, but standalone packages (like `langchain-chroma`) are the long-term path.
- **`hub.pull` is blocked by default in newer LangChain.** Write the prompt with `ChatPromptTemplate` instead.

## Next time

- [ ] Write down my understanding of the full chain (the placeholder in cell 14).
- [ ] Fix `format_docs` (`"\n\n"`) and the `post_title` class name, then re-run Part 1.
- [ ] Finish **Part 3 Retrieval**: try different `k` values and look at which chunks come back.
- [ ] **Part 4 Generation:** connect the retriever to the prompt and LLM step by step, and check traces in LangSmith.
- [ ] Try `langchain-chroma` instead of `langchain_community.vectorstores.Chroma`.

---

## Continued on 2026-11-21: Generation, Query Translation & Routing (Parts 4–10)

**Summary:** I finished the basic RAG loop with **Part 4 (Generation)**, then moved on to **query translation**: rewriting or expanding the user's question before retrieval, so the vector search finds better chunks. I wrote all five techniques from the tutorial (Parts 5–9), moved each from OpenAI to Gemini, and fixed a few issues along the way. The retriever is still `k=1`, so each query brings back one chunk. I finished with **Part 10 (Routing)**: sending a question to the right data source or prompt, by LLM classification (logical) or by embedding similarity (semantic).

| Part | Technique | Idea in one line | Status in my notebook |
|---|---|---|---|
| 4 | Generation | Put retrieved chunks into a prompt, then the LLM answers | Ran ✅ |
| 5 | Multi-Query | LLM writes 5 versions of the question → retrieve for each → unique union | Code written, no saved outputs |
| 6 | RAG Fusion | Several queries → combine the ranked lists with **Reciprocal Rank Fusion** | Code written, no saved outputs |
| 7 | Decomposition | Split into sub-questions → answer one at a time, feeding earlier Q&A forward | Code written, no saved outputs |
| 8 | Step-Back | Ask a more general "step-back" question, then retrieve for **both** questions | Ran ✅ |
| 9 | HyDE | LLM writes a hypothetical answer passage → retrieve using that passage | Ran ✅ |
| 10 | Routing (logical + semantic) | Pick the data source / prompt: LLM structured output, or closest prompt embedding | Ran ✅ |

### Part 4: Generation
```python
prompt = ChatPromptTemplate.from_template("Answer the question based only on the following context:\n{context}\n\nQuestion: {question}")
chain = prompt | llm
chain.invoke({"context": docs, "question": "What is Task Decomposition?"})
```
- The answer came only from the retrieved chunk: task decomposition is done (1) by an LLM with simple prompting, (2) with task-specific instructions, or (3) with human input.
- The `AIMessage.content` from this model is a **list of parts**, not a plain string. Use `result.content[0]["text"]`, or add `StrOutputParser()` to get plain text.
- A test I set up but haven't run: give it unrelated context ("Cats are mammals.") to check that the model admits it doesn't know.

### Part 5: Multi-Query
- The prompt asks for **5 alternative phrasings** to get past the limits of distance-based similarity search.
- `generate_queries | retriever.map() | get_unique_union`: `.map()` runs the retriever once per query. `get_unique_union` removes duplicates by serialising documents with `dumps` and `loads`.
- The final chain feeds `retrieval_chain` into `{context}` and uses `itemgetter("question")` for `{question}`.
- My own note to try: compare **token usage** between single-query and multi-query retrieval.

### Part 6: RAG Fusion
- 4 generated queries → one ranked list of chunks for each query → **Reciprocal Rank Fusion**:
  `score(doc) = Σ 1 / (rank + k)` over every list the doc appears in, with `k = 60`.
  Chunks that rank well across many queries end up on top.
- My version keys documents on `(page_content, source)` instead of `dumps()`, and keeps the original `Document` objects.
- `fused_to_context` drops the scores and joins the text before it goes into the prompt.

### Part 7: Decomposition
- The LLM splits "What are the main components of an LLM-powered autonomous agent system?" into **3 sub-questions**.
- A loop answers each sub-question with its own retrieval, plus all **earlier Q&A pairs** as background (`q_a_pairs`), so later answers build on earlier ones. The last `answer` is the final response.
- I added `format_docs` so the prompt gets clean text instead of `Document(...)` reprs.

### Part 8: Step-Back
- **Few-shot prompting** with `FewShotChatMessagePromptTemplate`: two example pairs teach the model to turn a specific question into a more general one.
- "What is task decomposition for LLM agents?" → step-back question **"what is task decomposition?"**
- The final chain retrieves for **both** the original question (`normal_context`) and the step-back question (`step_back_context`), then answers using both.
- **Debugging I added:**
  - A `tap()` RunnableLambda that prints the query going into each retriever.
  - A `safe_step_back` guard that falls back to the original question if the step-back output is blank. I checked the raw response: `finish_reason: STOP`, 88 input / 5 output tokens, and the text wasn't blank on that run.

### Part 9: HyDE (Hypothetical Document Embeddings)
- Questions and documents read differently, so HyDE has the LLM write a **"scientific paper passage"** answering the question, then searches with that passage instead of the question. A passage-shaped query sits closer to real passages in embedding space.
- The generated passage was long and academic. Retrieval with it found the blog's **"Component One: Planning / Task Decomposition"** chunk, and the final answer was grounded in it.
- `safe_hyde` guard: if the passage is blank, fall back to the question. The retrieval chain is named `retrieval_chain_hyde` so it doesn't overwrite the multi-query chain.

### Part 10: Routing
Routing decides **where a question should go** before retrieval or generation.

**Logical routing (LLM decides):**
```python
class RouteQuery(BaseModel):
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(..., description="...")

structured_llm = router_llm.with_structured_output(RouteQuery)
router = prompt_route | structured_llm
full_chain = router | RunnableLambda(choose_route)
```
- `with_structured_output` makes Gemini return a `RouteQuery` object instead of text. The `Literal` type limits it to the three allowed values.
- A question with LangChain Python code → `RouteQuery(datasource='python_docs')`, and `full_chain` returned **"chain for python_docs"**.
- `choose_route` is a plain function wrapped in `RunnableLambda`. In a real app each branch would return a different retrieval chain.
- Moved from the tutorial: `langchain_core.pydantic_v1` was removed in newer langchain-core, so I import `BaseModel, Field` straight from `pydantic`.

**Semantic routing (embeddings decide):**
- Embed two prompt templates (physics professor, mathematician) once with `embed_documents`.
- For each question, `embed_query` it, compute cosine similarity against both prompt embeddings, and use the prompt with the highest score (`argmax`).
- I wrote my own NumPy `cosine_similarity(X, Y)`: normalise each row to length 1, then `X @ Y.T` gives every pairwise similarity at once.
- "What's a black hole" → **Using PHYSICS**, and the answer was a short, simple explanation in the professor's style ("the ultimate cosmic trapdoor").

**Logical vs semantic:** logical routing costs one LLM call but can reason about the question; semantic routing is just embeddings and a dot product, so it's cheaper and faster but only as good as how distinct the route descriptions are.

### Gotchas (21 Nov)
- **Part 5, `print(docs.text)` will fail:** `docs` is a list of Documents, so use `for d in docs: print(d.page_content)`.
- **Run order matters:** Parts 5–7 reuse names like `llm`, `prompt`, `generate_queries`, `docs` and `final_rag_chain`. Running parts out of order quietly uses the wrong object, which is why I gave HyDE its own chain name.
- **`k=1` limits these techniques:** each query returns only one chunk, so Multi-Query and Fusion can't gather much. Try `k=3`–`5`.
- **The Part 4 LLM was created without `vertexai=True`.** It still worked, but the other parts pass it explicitly, which is safer.
- **The temperature warning** appears on every call: `gemini-3.5-flash-lite` ignores `temperature`.
- Typos in prompts ("Answert", "useer", "militations"). The model copes, but it's worth cleaning up.
- **Part 10, `choose_route`'s `else` branch** returns `"golang_docs"`, not `"chain for golang_docs"` like the other two. Harmless here, but inconsistent.
- **Part 10 reuses `router_llm`** for the semantic chain's answer, so the temperature warning shows up there too.

### Next time
- [ ] Run Parts 5, 6 and 7 and save the outputs.
- [ ] Fix `print(docs.text)` in Part 5.
- [ ] Raise `k` to 3–5 and compare answers for single-query, Multi-Query, Fusion, Step-Back and HyDE on the same question.
- [ ] Compare token usage (LangSmith traces or `usage_metadata`) between single and multi-query.
- [ ] Make each routing branch return a real retrieval chain (e.g. a separate vector store per docs source).
- [ ] Try semantic routing on a question that's both math and physics, and print the two similarity scores.
- [ ] Next in the tutorial: **query construction** (turning a question into metadata filters).
