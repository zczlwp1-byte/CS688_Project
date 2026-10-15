# CS688 — NewsInsight: Local-LLM News Research Agent

A Boston University MET CS688 term project combining live news collection, local language models, retrieval-augmented generation (RAG), sentiment analysis, and Streamlit.

Application source: [`news-insight-agent/`](news-insight-agent/). Seven course reference PDFs are preserved at the root. These are supporting materials rather than application outputs.

## Features and architecture

```text
Streamlit interface
  → NewsInsightAgent: up to six model/tool steps per question
      ├─ fetch_news → DuckDuckGo → scraping / excerpt fallback
      │                          → chunking → Ollama embeddings → FAISS
      ├─ retrieve → top-four chunks → local answer generation → sources
      └─ analyze_sentiment → VADER scores and article summary
  → Insights: article sentiment and top-15 frequent keywords
```

The agent loop is implemented directly in Python without an agent orchestration framework. Model inference uses Ollama; news search and article downloads require external network requests, so the complete application is not offline.

| File | Purpose |
|---|---|
| [`app.py`](news-insight-agent/app.py) | Sidebar, chat, article links, sentiment and keyword charts |
| [`agent.py`](news-insight-agent/agent.py) | Tool registry, output parsing, execution loop and error observations |
| [`llm.py`](news-insight-agent/llm.py) | Ollama chat, generation, embeddings and availability check |
| [`tools/scraper.py`](news-insight-agent/tools/scraper.py) | News search, paragraph extraction and excerpt fallback |
| [`tools/rag.py`](news-insight-agent/tools/rag.py) | Chunking, normalized FAISS search and contextual answering |
| [`tools/sentiment.py`](news-insight-agent/tools/sentiment.py) | VADER polarity, summaries and word frequencies |
| [`requirements.txt`](news-insight-agent/requirements.txt) | Python dependencies |
| [Original README](news-insight-agent/README.md) | Original project overview and notes |

## Setup

The original project specifies Python 3.10+ and a local Ollama installation. From the repository root:

```powershell
cd news-insight-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
streamlit run app.py
```

Ensure Ollama is running; if needed, start `ollama serve` separately. Defaults in `llm.py` are `qwen2.5:7b` for chat and `nomic-embed-text` for embeddings.

Dependencies include Streamlit, Ollama's Python client, FAISS CPU, NumPy, pandas, Requests, BeautifulSoup, `ddgs` and VADER. Requirements specify minimum versions rather than a locked environment.

## Usage

1. Start Ollama and the Streamlit application.
2. Enter a topic such as `renewable energy` and select **Fetch & index**.
3. Inspect indexed article links in the sidebar.
4. Ask, for example, `What themes recur across these articles?` or `Summarize the sentiment of the fetched articles.`
5. Inspect the answer, tool observations and returned source links.
6. Open **Insights** to compare sentiment and frequent keywords.

These are suggested demonstration prompts, not recorded results. Actual outputs depend on news availability, scraping success and installed models.

## Implementation settings

| Setting | Current value |
|---|---|
| Agent loop | Maximum six steps per question |
| News collection | Up to five search results per fetch |
| Article request timeout | 15 seconds |
| Scrape fallback | Search excerpt if fetching fails or text is shorter than 200 characters |
| Paragraph filter | Retains paragraphs longer than 60 characters |
| Chunking | Approximately 800 characters with a 150-character tail overlap |
| Embeddings | `nomic-embed-text` through Ollama |
| Search | FAISS `IndexFlatIP` on L2-normalized vectors, equivalent to cosine ranking |
| Retrieved context | Top four chunks by default |
| Temperature | 0.2 by default |
| Sentiment input | First 5,000 characters per article |
| Polarity labels | Positive ≥ 0.05; negative ≤ −0.05; otherwise neutral |
| Keywords | Top 15 token counts after stopword filtering |
| Storage | Articles and vectors in Streamlit session memory |

## Evidence and limitations

The archive contains source code and a descriptive README, with no benchmark dataset, execution logs, measured latency, accuracy table or saved demonstration results. The original README reports qualitative observations about model formatting and scraping; these were not independently reproduced during upload.

The source establishes the following constraints:

- **Grounding is prompt-based.** RAG requests answers based only on retrieved context, but does not verify claims or citations mechanically.
- **Model settings have limited scope.** The sidebar model controls agent planning calls; RAG answer generation uses the default model in `llm.py`.
- **The article slider is not connected.** The UI shows a 3–10 slider, while the fetch tool hardcodes five results.
- **Evidence coverage varies.** Failed or short scrapes use excerpts, which may omit context.
- **Storage is temporary.** Restarting the session loses the index. Repeated fetches append records without deduplication or a reset control.
- **Conversation context is limited.** Previous messages are displayed, but each agent run begins a new model conversation containing the current question.
- **Parsing is lightweight.** Recognized malformed Action JSON can trigger an error observation; unrecognized output may be returned as a final answer. Recovery is not guaranteed.
- **Sentiment is descriptive.** VADER may misread sarcasm or specialized language; keyword extraction uses frequency counts rather than TF-IDF or semantic ranking.
- **Runtime validation remains outstanding.** No end-to-end run with downloaded models or live news was performed during publishing.

## Reference materials

Seven root PDFs cover agent basics, AutoGen tools, data-science agents, local LLMs, Python interfaces, student examples and term-project guidance. The application does not require these PDFs at runtime.

Python bytecode, virtual environments, environment files and local Streamlit secrets are excluded by `.gitignore`.
