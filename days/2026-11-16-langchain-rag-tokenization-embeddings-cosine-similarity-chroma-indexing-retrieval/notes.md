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

---

## Continued on 2026-11-23: Query Structuring (Part 11)

**Summary:** Query structuring (also called query construction) turns a plain-English question into a **structured database query**: a text search plus **metadata filters** such as view count, publish date and video length. The LLM fills in a Pydantic schema, so "videos published in 2023" becomes real date filters instead of just words in a similarity search.

### Loading YouTube metadata (didn't work)
- Installed `youtube-transcript-api` and `pytube`, then tried `YoutubeLoader.from_youtube_url(..., add_video_info=True)`.
- It failed with **`HTTPError: HTTP Error 400: Bad Request`**. `add_video_info=True` uses `pytube` to read the title, views and length, and pytube 15.0.0 often breaks when YouTube changes its site. The rest of Part 11 doesn't need this cell.

### The schema: `TutorialSearch`
```python
class TutorialSearch(BaseModel):
    content_search: str            # similarity search over transcripts
    title_search: str              # short keyword version for titles
    min_view_count: Optional[int] = None
    max_view_count: Optional[int] = None
    earliest_publish_date: Optional[datetime.date] = None   # inclusive
    latest_publish_date: Optional[datetime.date] = None     # exclusive
    min_length_sec: Optional[int] = None
    max_length_sec: Optional[int] = None
```
- Each field's `description` tells the LLM what it means, and the filters say "Only use if explicitly specified", so the model leaves them empty unless the question asks for one.
- `pretty_print()` shows only the fields that were filled in. I moved it from the deprecated `self.__fields__` to pydantic v2's `type(self).model_fields`, and imported from `pydantic` instead of `langchain_core.pydantic_v1`.

### The query analyzer
```python
structured_llm = llm_structured.with_structured_output(TutorialSearch)
query_analyzer = prompt_structured | structured_llm
```
The system prompt says to convert questions into database queries and **not to rephrase unfamiliar acronyms or words**.

| Question | What Gemini returned |
|---|---|
| "rag from scratch" | content and title search `rag from scratch`, no filters |
| "videos on chat langchain published in 2023" | search `chat langchain`, `earliest_publish_date: 2023-01-01`, `latest_publish_date: 2024-01-01` |
| "…chat langchain that are published before 2024" | search `chat langchain`, only `latest_publish_date: 2024-01-01` |
| "how to use multi-modal models in an agent, only videos under 5 minutes" | content search the full phrase, title search `multi-modal models agent`, `max_length_sec: 300` |

- "In 2023" became a **date range** (1 Jan 2023 up to, but not including, 1 Jan 2024), which matches the inclusive/exclusive rules in the field descriptions.
- "Under 5 minutes" became **300 seconds**: the model did the unit conversion.
- The title search is shorter than the content search, as its description asks.

### Gotchas (23 Nov)
- **`YoutubeLoader` with `add_video_info=True` gave HTTP 400** (pytube). Try `add_video_info=False`, or get the metadata another way.
- **`pip install` inside a notebook:** restart the kernel afterwards. `%pip install` is safer than `pip install` because it installs into the kernel's own environment.
- The temperature warning again: `gemini-3.5-flash-lite` ignores `temperature=0`.

### Next time
- [ ] Get the YouTube loader working (`add_video_info=False`, or a newer pytube fork) and look at real metadata.
- [ ] Turn the `TutorialSearch` output into a real Chroma `where` filter, and run a filtered search.
- [ ] Try LangChain's `SelfQueryRetriever`, which does query structuring and filtering in one step.
- [ ] Next in the tutorial: **indexing techniques** (multi-representation, RAPTOR, ColBERT).

---

## Continued on 2026-11-25: Indexing — Multi-Representation, RAPTOR, ColBERT + CRAG & Adaptive RAG notes (Parts 12–17)

**Summary:** I moved from *query-side* tricks to **indexing**, which is about how documents are stored so the right thing can be found. I ran **multi-representation indexing** end to end with Gemini (search over summaries, return full documents). I wrote up **RAPTOR** (hierarchical summary trees) as notes, and tried **ColBERT** with RAGatouille, which failed on an import error. Then I wrote notes on two "self-checking" RAG designs: **Corrective RAG (CRAG)** and **Adaptive RAG**.

| Part | Technique | Idea in one line | Status in my notebook |
|---|---|---|---|
| 12 | Multi-representation indexing | Embed a **summary** of each doc; return the **full doc** it points to | Ran ✅ |
| 13 | RAPTOR | Cluster and summarise chunks into a tree; index every level together | Notes only |
| 14 | ColBERT (RAGatouille) | One embedding **per token**, scored with late interaction | Failed ❌ (`ModuleNotFoundError`) |
| 16 | CRAG | Grade retrieved docs; keep, discard and web-search, or use both | Notes only |
| 17 | Adaptive RAG | Classify the question's complexity first: no, single or multi-step retrieval | Notes only |

(There's no Part 15 in my notebook; I numbered CRAG as 16.)

### Part 12: Multi-representation indexing
In my own words, from the notebook: when the store has lots of documents, each document gets **summarised**, and only the summary (plus a doc id) goes into the vector store. A question is embedded and matched against the summaries, then the doc id fetches the **full document** from a separate byte store. Use it when you want whole documents back, not chunks.

```
docs ──LLM summarise──→ summaries (+ doc_id) ──embed──→ Chroma  (search here)
  └────────────────────── full docs keyed by doc_id ──→ InMemoryByteStore  (return from here)
```

- **Load:** two Lilian Weng posts ("LLM Powered Autonomous Agents" and "human data quality") with `WebBaseLoader`. That's 2 documents, with no splitting.
- **Summarise:** `chain.batch(docs, {"max_concurrency": 5})` runs the summary chain on both docs in parallel. Gemini produced long, structured Markdown summaries.
- **Store:**
  ```python
  mv_retriever = MultiVectorRetriever(
      vectorstore=mv_vectorstore,          # Chroma collection "summaries"
      byte_store=InMemoryByteStore(),      # full docs
      id_key="doc_id",
      search_kwargs={"k": 1},
  )
  mv_retriever.vectorstore.add_documents(summary_docs)        # summaries with metadata doc_id
  mv_retriever.docstore.mset(list(zip(doc_ids, docs)))        # doc_id → full doc
  ```
- **Query "Memory in agents":**
  - `mv_vectorstore.similarity_search(query, k=1)` returns the **summary** of the agents post, with its `doc_id` in the metadata.
  - `mv_retriever.invoke(query)` returns the **full original page** for that id: the raw text starting "LLM Powered Autonomous Agents | Lil'Log… Table of Contents…".
- **Moved to LangChain 1.x:**

  | Tutorial | What I used |
  |---|---|
  | `from langchain.storage import InMemoryByteStore` | `from langchain_core.stores import InMemoryByteStore` |
  | `from langchain.retrievers.multi_vector import MultiVectorRetriever` | `from langchain_classic.retrievers.multi_vector import MultiVectorRetriever` |
  | `retriever.get_relevant_documents(q)` | `retriever.invoke(q)` |
  | `n_results=1` | `search_kwargs={"k": 1}` |

### Part 13: RAPTOR (hierarchical indexing): my notes
- **Like a textbook:** pages (chunks), chapter summaries (clusters of related chunks), and a book summary (a summary of the chapter summaries).
- **The problem it solves:** plain chunk RAG fails on **broad** questions ("what are the main themes?"), because no single chunk holds the big picture.
- **How the tree is built:** embed → cluster by similarity → summarise each cluster → repeat on the summaries until there are few enough.
  `63 chunks → 8 summaries → 3 summaries` = **74 nodes**, all indexed together (the **"collapsed tree"** approach). Broad questions match high-level summaries, and detailed questions match the original chunks.

### Part 14: ColBERT with RAGatouille (failed)
- **The idea:** instead of one vector per chunk, ColBERT keeps **one vector per token**. A query is scored by matching each query token to its best document token and adding up the matches ("late interaction"). This catches precise word-level matches that a single chunk vector can blur.
- **What I set up:** `pip install -U ragatouille`, `RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")`, a Wikipedia API helper to fetch the "Hayao Miyazaki" article, `RAG.index(..., max_document_length=180, split_documents=True)`, then `RAG.as_langchain_retriever(k=3)`.
- **Why it failed:** importing `ragatouille` raised **`ModuleNotFoundError: No module named 'langchain.retrievers'`**. RAGatouille 0.0.9 imports from the old LangChain layout, which LangChain 1.x removed. Because the import failed, `RAG` was never created, so the next two cells gave `NameError: name 'RAG' is not defined`.
- The Wikipedia helper sends a descriptive `User-Agent`, as Wikipedia's API asks. I swapped the tutorial author's email for a placeholder.

### Part 16: Corrective RAG (CRAG): my notes
- **One line:** check the retrieved documents **before** using them, and correct course if they look bad.
- A lightweight **retrieval evaluator** scores relevance → one of three actions:
  ```
  question → retrieve → GRADE the documents
                           ├─ correct   → refine into relevant "knowledge strips" → generate
                           ├─ incorrect → discard, use web search instead        → generate
                           └─ ambiguous → use both                               → generate
  ```
- It's plug-and-play: the paper uses it alongside standard RAG and Self-RAG. The LangGraph versions usually grade with an LLM and fall back to query rewriting and web search.
- **Trade-offs:** extra grading calls and latency, and it depends on the grader and on web search quality.

### Part 17: Adaptive RAG: my notes
- **One line:** decide **how much** retrieval a question needs before doing any.
- A small classifier model predicts query complexity:
  ```
  question → CLASSIFY complexity
                ├─ simple             → answer directly, no retrieval
                ├─ medium             → one retrieve-and-answer step
                └─ complex (multi-hop) → iterative retrieval, step by step
  ```
- The paper's example of a multi-hop question is "What currency is in Billy Giles' birthplace?" (find the birthplace, then the currency). Simple questions skip the expensive pipeline.

### Gotchas (25 Nov)
- **RAGatouille doesn't work with LangChain 1.x** (`langchain.retrievers` is gone). Options: a separate venv with `langchain<1`, or the newer PyLate backend that RAGatouille 0.0.10 is moving to (its own warning says so). Also, the install pulled in **torch, faiss, llama-index and colbert-ai**, which is a lot of extra size in `.venv`.
- **The full doc returned in Part 12 is raw page text** with lots of blank lines, because this loader has no `bs4.SoupStrainer`. Add the same filter as in Part 1 for cleaner documents.
- **`k=1` in Part 12** returns one whole document, so it's fine with only 2 docs, but with many docs use `k=2`–`3`.
- **The `Chroma` deprecation warning** again: switch to the `langchain-chroma` package.
- **Kernel restart between Parts 12 and 13:** the execution counts reset to 1 at Part 13, so Parts 13–14 ran in a fresh kernel. Any cell that uses `MODEL_ID`, `PROJECT_ID` and so on needs the setup cells run first.
- Using `! pip install` in a notebook: `%pip install` installs into the kernel's own environment, which is safer.

### Next time
- [ ] Get ColBERT working: a separate venv with `langchain<1`, or try PyLate, then run the Miyazaki question.
- [ ] Write the RAPTOR code (embed → cluster → summarise loop) instead of notes only.
- [ ] Add `bs4.SoupStrainer` to the Part 12 loader and compare the full-doc output.
- [ ] Build a small CRAG graph in **LangGraph**: an LLM grader → if it's irrelevant, rewrite the query and web-search.
- [ ] Try a simple Adaptive RAG router: an LLM classifies a question as simple, medium or complex and picks the pipeline (reusing my Part 10 routing code).
