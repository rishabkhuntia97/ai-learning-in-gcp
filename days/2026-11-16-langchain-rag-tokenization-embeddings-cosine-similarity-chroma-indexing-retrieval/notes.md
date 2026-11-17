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
