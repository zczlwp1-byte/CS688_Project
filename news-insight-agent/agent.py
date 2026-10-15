"""
agent.py — A ReAct AI agent built from scratch (no agent framework).

The agent follows the Thought -> Action -> Observation loop:
  1. The LLM receives the user query plus a system prompt describing its tools.
  2. It "thinks", then either emits an Action (a JSON tool call) or a Final Answer.
  3. Tool results are fed back as Observations until it can answer.

This mirrors the from-scratch agent architecture covered in the CS688
"AI Agents in Python" slides.
"""

from __future__ import annotations

import json
import re

import llm
from tools import scraper, sentiment
from tools.rag import VectorStore

MAX_STEPS = 6

SYSTEM_PROMPT = """You are NewsInsight, an autonomous research agent that analyzes news.

You run in a loop of Thought, Action, Observation. At each step, output EXACTLY one of:

Thought: <your reasoning about what to do next>
Action: {"tool": "<tool_name>", "input": "<tool input string>"}

or, when you have enough information:

Thought: <your reasoning>
Final Answer: <your complete answer to the user>

Available tools:
- fetch_news: input is a topic string. Searches the web for recent news on the topic,
  scrapes the articles, and indexes them into the knowledge base. Returns the titles found.
- retrieve: input is a question string. Searches the indexed articles and returns an
  answer grounded in them, with sources. Use ONLY after articles have been fetched.
- analyze_sentiment: input is ignored (use ""). Returns a sentiment report
  (positive/negative/neutral) over all fetched articles.

Rules:
- If the knowledge base is empty and the user asks about a topic, fetch_news first.
- Never invent facts; base answers on tool observations.
- After "Action:" output valid JSON on a single line and STOP — wait for the Observation.
"""


class NewsInsightAgent:
    """Holds the tool implementations and the ReAct loop state."""

    def __init__(self, model: str = llm.CHAT_MODEL):
        self.model = model
        self.store = VectorStore()
        self.articles: list[dict] = []

    # ---------------- tool implementations ----------------

    def _tool_fetch_news(self, topic: str) -> str:
        articles = scraper.collect_articles(topic, max_results=5)
        if not articles:
            return f"No articles found for '{topic}'."
        added = 0
        for art in articles:
            added += self.store.add_article(art)
        self.articles.extend(articles)
        titles = "\n".join(f"- {a['title']} ({a['source']})" for a in articles)
        return f"Fetched and indexed {len(articles)} articles ({added} chunks):\n{titles}"

    def _tool_retrieve(self, question: str) -> str:
        result = self.store.answer(question)
        src = "\n".join(f"  [{s['title']}] {s['url']}" for s in result["sources"])
        return f"{result['answer']}\nSOURCES:\n{src}" if src else result["answer"]

    def _tool_analyze_sentiment(self, _: str) -> str:
        return sentiment.summarize_sentiment(self.articles)

    def _run_tool(self, name: str, tool_input: str) -> str:
        tools = {
            "fetch_news": self._tool_fetch_news,
            "retrieve": self._tool_retrieve,
            "analyze_sentiment": self._tool_analyze_sentiment,
        }
        if name not in tools:
            return f"Unknown tool '{name}'. Available: {', '.join(tools)}."
        try:
            return tools[name](tool_input)
        except Exception as exc:  # surface tool errors as observations
            return f"[tool error] {exc}"

    # ---------------- ReAct loop ----------------

    @staticmethod
    def _parse(reply: str) -> tuple[str, str | None, str | None]:
        """Return (thought, action_json, final_answer) — one of the last two is None."""
        thought_match = re.search(r"Thought:\s*(.*?)(?:\n|$)", reply)
        thought = thought_match.group(1).strip() if thought_match else ""
        final_match = re.search(r"Final Answer:\s*(.*)", reply, re.DOTALL)
        if final_match:
            return thought, None, final_match.group(1).strip()
        action_match = re.search(r"Action:\s*(\{.*?\})", reply, re.DOTALL)
        if action_match:
            return thought, action_match.group(1).strip(), None
        # Neither found: treat whole reply as the final answer.
        return thought, None, reply.strip()

    def run(self, query: str, on_step=None) -> str:
        """Answer a user query. `on_step(kind, text)` is an optional callback
        used by the UI to stream Thought/Action/Observation steps."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        for _ in range(MAX_STEPS):
            reply = llm.chat(messages, model=self.model)
            thought, action_json, final = self._parse(reply)
            if on_step and thought:
                on_step("thought", thought)
            if final is not None:
                return final

            # Execute the requested tool.
            try:
                action = json.loads(action_json)
                tool_name = action.get("tool", "")
                tool_input = str(action.get("input", ""))
            except (json.JSONDecodeError, AttributeError):
                observation = "Could not parse Action JSON. Use {\"tool\": ..., \"input\": ...}."
                tool_name = "?"
            else:
                if on_step:
                    on_step("action", f"{tool_name}({tool_input})")
                observation = self._run_tool(tool_name, tool_input)

            if on_step:
                on_step("observation", observation[:1500])
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return "I reached the maximum number of reasoning steps without a final answer."
