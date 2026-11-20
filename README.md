# Agentic Workflow with LangGraph

A terminal assistant built with LangGraph. An LLM classifies each message and routes it to the right agent: chat, company knowledge (RAG), web search, coding via Claude Code, or memory.

![Graph](graph.png)

## Features

- **Chat**: casual conversation. Lasting facts about you are saved to memory automatically.
- **Knowledge (RAG)**: answers from the Markdown files in `docs/`, indexed with FAISS. If the docs don't have the answer, it falls back to web search.
- **Web search**: DuckDuckGo, no API key needed. Answers start with `(from web)`.
- **Coding**: rewrites your request for Claude Code and asks for your approval (`yes` / `no` / a revised request), then runs it in `workspace/`.
- **Memory**: `remember ...` and `forget ...` manage `MEMORY.md`, which every agent reads.
- **Streaming**: chat, RAG and web answers are printed token by token.

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and the [Claude Code](https://claude.com/claude-code) CLI (for coding requests).

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
uv run main.py
```

## Usage examples

```
When was Zentrix founded?            -> RAG
How much is the Enterprise plan?     -> RAG
What's the capital of France?        -> web
Add a hello function to utils.py     -> coding (asks for approval)
Remember that I prefer TypeScript    -> memory
Forget that I prefer TypeScript      -> forget
```

## Notes

- **Updating docs**: after editing files in `docs/`, delete `faiss_index/` and restart to rebuild the index.
- **Web search** scrapes search engines, so it can fail occasionally. It retries 3 times before answering "I don't know".
- Conversation history is kept in memory only and is lost on exit. `MEMORY.md` persists.
