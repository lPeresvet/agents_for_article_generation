# Article agents

A **multi-agent article pipeline** built with [LangGraph](https://github.com/langchain-ai/langgraph). A **manager** (main editor) coordinates **researcher**, **correspondent**, and **reviewer** agents, then **finalize** produces the published article. LLM calls use **Ollama** by default; **GigaChat** is used for predominantly Russian topics when credentials are configured.

## Features

- **LangGraph workflow**: `memory_gate` → `manager` → (`researcher` \| `correspondent` \| `reviewer` \| `finalize`) with conditional routing and a hard cap on manager steps (`MAX_MANAGER_STEPS = 6`).
- **Model routing**: Cyrillic-heavy topics → GigaChat; otherwise Ollama. Override with `MODEL_ROUTE_OVERRIDE=gigachat` or `ollama`.
- **Web search**: Researcher and correspondent can call DuckDGo search via LangChain tool `internet_search_DDGO`.
- **Long-term memory (ChromaDB)**: Stores finalized articles per topic; retrieves similar past runs for the manager and can short-circuit the graph when an article for the same topic already exists (cache).
- **Observability**: Optional LangSmith traces when `LANGCHAIN_TRACING_V2` and `LANGCHAIN_API_KEY` are set; stderr tracing with `ARTICLE_AGENTS_TRACE` or `-v` / `--trace`.

## Requirements

- **Python** 3.12+ (see `Dockerfile`).
- **Ollama** with chat and embedding models matching `.env` (defaults: `qwen3.5:2b`, `nomic-embed-text`).
- **GigaChat** (optional): base64 `client_id:client_secret` in `GIGACHAT_CREDENTIALS` for Russian routing.
- **ChromaDB** (optional but recommended for memory): HTTP API, e.g. via Docker Compose.

## Setup

1. Clone the repo and create a virtual environment.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy environment template and fill in values:

   ```bash
   cp .env.example .env
   ```

   Key variables are documented in `.env.example` (GigaChat, Ollama base URL/model, LangSmith, Chroma host/port).

4. Start **Ollama** locally and pull the models you reference in `.env`.

5. (Optional) Start **ChromaDB** — for example with the provided Compose file:

   ```bash
   docker compose up -d chromadb
   ```

## CLI usage

Run as a module:

```bash
python -m article_agents "Your article topic here"
```

- **Topic**: first positional argument, or **stdin** if not a TTY (e.g. `echo "Topic" | python -m article_agents`).
- **`-v` / `--verbose` or `--trace`**: stream node updates to stderr and enable timestamped trace lines (`ARTICLE_AGENTS_TRACE=1`).
- **`-i` / `--interactive`**: prompt for multiple topics until empty line, `exit`, or EOF.

The **final article** is printed to **stdout**. Exit code `0` on success, `1` if no `final_article` was produced.

### Docker

Build and run the full stack (Chroma + app) with the default Compose command (interactive, verbose):

```bash
docker compose up --build
```

The `article-agents` service expects Ollama at `OLLAMA_BASE_URL` (default `http://host.docker.internal:11434` on macOS/Windows). Ensure Ollama runs on the host with the configured models.

## Configuration highlights

| Area | Variables (examples) |
|------|----------------------|
| Ollama | `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_TEMPERATURE`, `OLLAMA_REQUEST_TIMEOUT`, `OLLAMA_NUM_PREDICT` |
| GigaChat | `GIGACHAT_CREDENTIALS`, `GIGACHAT_SCOPE`, `GIGACHAT_MODEL`, `GIGACHAT_TEMPERATURE`, `GIGACHAT_VERIFY_SSL_CERTS` |
| Routing | `MODEL_ROUTE_OVERRIDE` |
| Chroma | `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_COLLECTION`, connect retries/delay |
| Manager memory | `MANAGER_MEMORY_ENABLED`, `MANAGER_MEMORY_K`, `MANAGER_MEMORY_MIN_SIMILARITY`, `MANAGER_MEMORY_CANDIDATES_K` |
| Cache shortcut | `MEMORY_CACHE_RETURN` (skip full run if a stored article matches the topic), `MEMORY_SCAN_MAX`, `MEMORY_CACHE_SEMANTIC_FIRST` |
| LangSmith | `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, `LANGCHAIN_TAGS`, `LANGSMITH_TOPIC_PREVIEW_CHARS`, `LANGSMITH_RUN_NAME_PREFIX` |

## Architecture

```text
START → memory_gate → manager ⟷ researcher / correspondent / reviewer → finalize → END
                              (manager routes by NEXT: … line + safety rules)
```

- **memory_gate**: If Chroma already has a final article for this topic, sets `final_article` and ends.
- **manager**: Reads topic, research, draft, reviewer notes; optional RAG snippets from past articles; outputs routing via a trailing `NEXT: researcher|correspondent|reviewer|finalize`.
- **researcher / correspondent**: Use bound tools (DuckDGo search) in a short multi-turn loop.
- **reviewer**: Critique without tools; expects `VERDICT: APPROVE` or `VERDICT: REVISE`.
- **finalize**: Polishes the draft and **writes** the result to Chroma.

Agent system prompts live in `article_agents/prompts.py`; optional extra tuning is loaded from `article_agents/agents/agent_*.md` (manager, researcher, correspondent, reviewer).

## Graph visualization

Export the compiled graph (PNG or Mermaid):

```bash
python -m article_agents.visualize_graph -o article_graph.png
python -m article_agents.visualize_graph -m graph.mmd
```

Local PNG generation prefers the system `dot` binary (Graphviz). See `visualize_graph.py` for fallbacks (Mermaid API, Jupyter helper `display_article_graph`).

## Project layout

```text
article_agents/
  __init__.py          # exports build_graph
  __main__.py          # CLI entry
  cli.py               # argparse, load_dotenv, run loop
  graph.py             # StateGraph wiring
  state.py             # ArticleState, limits, route literals
  nodes.py             # LangGraph nodes + tool loop
  routing.py           # parse NEXT:, compute_next_route
  prompts.py           # system prompts + agent_*.md tuning
  llm.py               # Ollama + GigaChat factories
  language.py          # Russian detection, resolve_model_route
  memory.py            # ManagerMemory (Chroma + Ollama embeddings)
  tools.py             # internet_search_DDGO
  trace.py             # ARTICLE_AGENTS_TRACE stderr logging
  observability.py     # LangSmith run config (metadata, tags, run_name)
  visualize_graph.py   # graph export utilities
  agents/              # optional markdown tuning per role
docker-compose.yml
Dockerfile
requirements.txt
.env.example
```

## Dependencies

Pinned roughly in `requirements.txt`: `langgraph`, `langchain-core`, `langsmith`, `langchain-ollama`, `langchain-gigachat`, `duckduckgo-search`, `chromadb`, `python-dotenv`, `grandalf`, `graphviz`.
