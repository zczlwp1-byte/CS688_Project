"""
app.py — Streamlit UI for the NewsInsight agent.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import llm
from agent import NewsInsightAgent
from tools import sentiment

st.set_page_config(page_title="NewsInsight Agent", page_icon="📰", layout="wide")

# ---------------- session state ----------------
if "agent" not in st.session_state:
    st.session_state.agent = NewsInsightAgent()
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {"role", "content"}

agent: NewsInsightAgent = st.session_state.agent

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("📰 NewsInsight")
    st.caption("Local-LLM news research agent — CS688 Term Project")

    if not llm.is_available():
        st.error("Ollama server not reachable. Start it with `ollama serve`.")

    agent.model = st.text_input("Ollama chat model", value=agent.model)

    st.divider()
    st.subheader("1 · Fetch articles")
    topic = st.text_input("Topic", placeholder="e.g. Tesla stock")
    n_articles = st.slider("Articles to fetch", 3, 10, 5)
    if st.button("Fetch & index", use_container_width=True) and topic:
        with st.spinner("Searching, scraping and indexing…"):
            result = agent._tool_fetch_news(topic)
        st.success(result.splitlines()[0])

    if agent.articles:
        st.divider()
        st.subheader("Indexed articles")
        for art in agent.articles:
            st.markdown(f"- [{art['title']}]({art['url']})")

# ---------------- main area ----------------
tab_chat, tab_insights = st.tabs(["💬 Ask the agent", "📊 Insights"])

with tab_chat:
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask anything about the fetched news…"):
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            steps_box = st.status("Agent is reasoning…", expanded=True)

            def on_step(kind: str, text: str):
                icons = {"thought": "🤔", "action": "🔧", "observation": "👀"}
                steps_box.markdown(f"{icons.get(kind, '•')} **{kind.title()}:** {text}")

            answer = agent.run(question, on_step=on_step)
            steps_box.update(label="Done", state="complete", expanded=False)
            st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})

with tab_insights:
    if not agent.articles:
        st.info("Fetch some articles first (sidebar) to see sentiment and keywords.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sentiment per article")
            results = sentiment.analyze_articles(agent.articles)
            df = pd.DataFrame(results)
            st.bar_chart(df.set_index("title")["compound"])
            st.dataframe(df, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Top keywords")
            keywords = sentiment.extract_keywords(agent.articles)
            kw_df = pd.DataFrame(keywords, columns=["keyword", "count"])
            st.bar_chart(kw_df.set_index("keyword")["count"])
