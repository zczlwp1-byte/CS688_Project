# NewsInsight — Local-LLM News Research Agent

**BU MET CS688 Term Project** — an autonomous AI agent that searches the web for news
on any topic, mines the articles, and answers questions about them using a **local LLM**,
**RAG** (FAISS vector search), and a from-scratch **ReAct agent**, all wrapped in a
**Streamlit** app.

## Why (Context)

LLMs cannot access real-time data — ask one "what is the market saying about Tesla this
week?" and it will hallucinate. This project solves that by combining web mining
(scraping fresh news), retrieval-augmented generation (grounding answers in the scraped
text), and an autonomous agent that decides *when* to search, retrieve, or analyze.
Everything runs locally via Ollama: private and free.

## Architecture

```
User (Streamlit UI)
        │
        ▼
 NewsInsightAgent  ← from-scratch ReAct loop (Thought → Action → Observation)
        │
        ├── fetch_news ──────► DuckDuckGo search + BeautifulSoup scraping
        ├── retrieve ────────► FAISS vector store + local embeddings (RAG)
        └── analyze_sentiment ► VADER sentiment + keyword extraction
        │
        ▼
   Local LLM (Ollama: qwen2.5 / llama3, nomic-embed-text for embeddings)
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: fetch panel, agent chat with visible reasoning steps, insights dashboard |
| `agent.py` | ReAct agent built from scratch (no LangChain/CrewAI) with a 3-tool registry |
| `llm.py` | Ollama wrapper — the only file touching the model backend |
| `tools/scraper.py` | Web mining: DuckDuckGo news search + article scraping |
| `tools/rag.py` | Chunking, local embeddings, FAISS index, grounded Q&A with sources |
| `tools/sentiment.py` | VADER sentiment scoring + TF keyword extraction |

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:

   ```bash
   ollama pull qwen2.5:7b        # or qwen2.5:3b / llama3.1:8b on smaller machines
   ollama pull nomic-embed-text
   ```

2. Install Python dependencies (Python 3.10+):

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

## Usage

1. In the sidebar, enter a topic (e.g. *"Tesla stock"*) and click **Fetch & index** —
   the app searches recent news, scrapes each article, and indexes it into FAISS.
2. In the **Ask the agent** tab, ask questions such as *"What is the overall market
   sentiment about Tesla?"*. The agent's Thought / Action / Observation steps are
   shown live; answers cite their source articles.
3. The **Insights** tab shows per-article sentiment scores and top keywords.

## Libraries Used

`streamlit` (UI) · `ollama` (local LLM + embeddings) · `faiss-cpu` (Meta's similarity
search) · `ddgs` (DuckDuckGo search) · `requests` + `beautifulsoup4` (scraping) ·
`vaderSentiment` (lexicon sentiment) · `pandas`/`numpy` (data handling)

## Findings & Limitations

- The from-scratch ReAct loop works reliably with 7B models; 3B models occasionally
  emit malformed Action JSON (handled by re-prompting via the observation).
- Scraping quality varies by site; paywalled articles fall back to search excerpts.
- VADER is fast but lexicon-based; sarcasm and finance jargon can be misread.
- Answers are grounded in retrieved chunks only — if the articles don't cover a
  question, the agent says so instead of hallucinating.
