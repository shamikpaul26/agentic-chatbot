# LangGraph Streamlit Chatbot

> **Version 5.0** | `chat_model_v4.py` + `chatbot_v5.py`

A production-ready conversational AI chatbot built with **LangGraph**, **Streamlit**, and **OpenAI**. Features persistent multi-session memory (SQLite), real-time streaming responses, and MCP (Model Context Protocol) tool integration for external service connectivity.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Module Breakdown](#3-module-breakdown)
4. [State & Persistence](#4-state--persistence)
5. [MCP Tool Integration](#5-mcp-tool-integration)
6. [Async / Event Loop Design](#6-async--event-loop-design)
7. [Configuration Reference](#7-configuration-reference)
8. [Known Issues & Improvements](#8-known-issues--improvements)
9. [File Structure](#9-file-structure)

---

## 1. Project Overview

This project is a production-ready conversational AI chatbot built with LangGraph, Streamlit, and OpenAI. It features persistent multi-session memory (SQLite), real-time streaming responses, and MCP tool integration for external service connectivity — specifically an expense-tracker tool.

### Key Features

- Multi-turn conversation with persistent SQLite checkpointing
- Real-time token streaming via Streamlit's `st.write_stream`
- Multi-session sidebar with conversation history
- MCP tool integration (expense-tracker via stdio transport)
- LangSmith observability tracing on the chat node
- Thread-safe async architecture using a single persistent event loop

### Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-3.5-Turbo via `langchain-openai` |
| Orchestration | LangGraph `StateGraph` |
| Persistence | SQLite via `AsyncSqliteSaver` + `aiosqlite` |
| UI | Streamlit |
| Tools | MCP via `langchain-mcp-adapters` |
| Observability | LangSmith `@traceable` |

---

## 2. Architecture Overview

The application follows a layered architecture with a strict separation between the async graph engine and Streamlit's synchronous rendering thread. All async operations are routed through a single persistent background event loop to avoid event-loop lifecycle conflicts.

### 2.1 High-Level Architecture Diagram

```
+------------------------------------------------------------------+
|                    STREAMLIT UI  (chatbot_v5.py)                  |
|                                                                    |
|   Sidebar                    Chat Window                          |
|   [Thread List]              [Message History]                    |
|   [New Chat Btn]             [st.chat_input]                      |
|        |                           |                               |
+--------|---------------------------|--------------------------------+
         |                           |
         v                           v
+------------------------------------------------------------------+
|           run_in_loop(coro)  /  stream_in_loop(coro)              |
|        Thread-safe bridge: asyncio.run_coroutine_threadsafe()     |
+------------------------------------------------------------------+
         |                           |
         v                           v
+------------------------------------------------------------------+
|             BACKGROUND EVENT LOOP  (daemon thread)                |
|                                                                    |
|   get_chatbot()  -->  AsyncSqliteSaver  <-->  chatbot.db          |
|        |                                                           |
|        v                                                           |
|   LangGraph StateGraph                                             |
|   +------------------+                                             |
|   | START            |                                             |
|   |     |            |                                             |
|   |  chat_node()     |                                             |
|   |     |            |                                             |
|   | END              |                                             |
|   +------------------+                                             |
|        |                                                           |
+--------|------------------------------------------------------------+
         |
         v
+---------------------------+     +-------------------------------+
| ChatOpenAI (gpt-3.5-turbo)|     | MCP Client (expense-tracker)  |
| llm.bind_tools(mcp_tools) |     | transport: stdio / uv run     |
+---------------------------+     +-------------------------------+
         |
         v
+------------------+
| LangSmith Tracer |
| @traceable       |
+------------------+
```

### 2.2 Request Data Flow

```
User types message
     |
     v
chatbot_v5.py: st.chat_input captures text
     |
     v
HumanMessage created, appended to session state
     |
     v
stream_response() generator starts
     |
     +--> asyncio.run_coroutine_threadsafe(_stream(), _loop)
               |
               v
          chatbot.astream(messages, config, stream_mode='messages')
               |
               v
          chat_node() invoked via LangGraph
               |
          +----+----+
          |         |
       LLM call   Tool call? (if tool_use in response)
          |         |
          +---------+
               |
               v
          AsyncSqliteSaver checkpoints state to chatbot.db
               |
               v
          chunk_queue.put(message_chunk.content)
               |
     <---------+  (streamed back via Queue)
     |
     v
st.write_stream renders tokens in real-time
     |
     v
Full response appended to message_history
```

---

## 3. Module Breakdown

### 3.1 `chat_model_v4.py` — Core Engine

This module owns all async AI infrastructure. It is never imported from Streamlit's main thread in an async context — all calls are bridged through `chatbot_v5.py`'s persistent background loop.

| Component | Description |
|---|---|
| `llm` | `ChatOpenAI(gpt-3.5-turbo, temp=0.7)` — base language model |
| `MultiServerMCPClient` | Connects to MCP servers via stdio; loads tools at startup |
| `mcp_tools` | Dynamic tool list fetched from expense-tracker MCP server |
| `llm_with_tools` | LLM bound with MCP tools via `.bind_tools()`; falls back to plain LLM |
| `ChatbotState` | TypedDict with `messages: Annotated[list[BaseMessage], add_messages]` |
| `chat_node()` | `@traceable` async node; invokes `llm_with_tools.ainvoke(messages)` |
| `build_graph()` | Constructs linear StateGraph: `START -> chat_node -> END` |
| `get_chatbot()` | Lazy singleton; initialises `aiosqlite` + `AsyncSqliteSaver` on first call |
| `retrive_all_thread()` | Async; iterates checkpointer history to recover thread IDs |
| `_init_lock` | `threading.Lock` (NOT `asyncio.Lock`) — prevents double-init race |

#### Key Design Decision: `threading.Lock` vs `asyncio.Lock`

`asyncio.Lock` captures the running event loop at creation time. At module-import time Streamlit's main-thread loop is active, so an `asyncio.Lock` would be bound to the wrong loop. `threading.Lock` has no event-loop binding and is safe to use here.

```python
_init_lock = threading.Lock()   # threading, NOT asyncio — no loop binding
```

### 3.2 `chatbot_v5.py` — Streamlit Frontend

This module renders the UI and bridges synchronous Streamlit code with the async LangGraph engine through a single persistent background event loop.

| Component | Description |
|---|---|
| `_loop` | `asyncio.new_event_loop()` — created once, lives for process lifetime |
| `_loop_thread` | Daemon thread running `_loop.run_forever()` |
| `run_in_loop(coro)` | Submits coroutine; blocks until result via `future.result()` |
| `generate_thread_id()` | Returns `str(uuid.uuid4())` for new chat sessions |
| `reset_chat()` | Creates new `thread_id`, clears `message_history` |
| `load_conversation()` | Fetches full message history for a thread from checkpointer |
| `get_chat_title()` | Reads first `HumanMessage` content (truncated to 30 chars) |
| `stream_response()` | Generator; runs `_stream()` in background loop, yields via Queue |
| `chunk_queue` | `threading.Queue` bridges async stream to sync Streamlit iterator |
| `CONFIG` | LangGraph run config: `thread_id`, metadata, tags, `run_name` |
| `st.write_stream` | Renders `stream_response()` tokens incrementally in the UI |

---

## 4. State & Persistence

### 4.1 LangGraph State

The graph state is a `TypedDict` with a single `messages` field. The `add_messages` reducer merges new messages into the existing list rather than replacing it, enabling multi-turn conversation continuity.

```python
class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

### 4.2 SQLite Checkpointing

Persistence is handled by `AsyncSqliteSaver` backed by an `aiosqlite` connection. The database file `chatbot.db` is created in the working directory. The checkpointer is initialised lazily on the first `get_chatbot()` call and reused for all subsequent requests.

```
Thread ID (UUID)  -->  LangGraph Config  -->  AsyncSqliteSaver
                                                      |
                                            chatbot.db (SQLite)
                                                      |
                          +--------------------------+--------------------------+
                          |                          |                          |
                     checkpoint                 metadata                 channel data
                     (messages)                (thread_id, ts)          (tool results)
```

### 4.3 Streamlit Session State

| Key | Description |
|---|---|
| `message_history` | `List[dict]` with `role`/`content` pairs for current chat display |
| `thread_id` | Active LangGraph thread UUID |
| `chat_thread` | List of all known thread UUIDs (loaded from DB at startup) |

---

## 5. MCP Tool Integration

The chatbot integrates with external tools via the Model Context Protocol (MCP). Tools are loaded at startup and bound to the LLM. If the MCP server is unavailable, `load_mcp_tools()` returns an empty list and the LLM operates without tools.

| Property | Value |
|---|---|
| Server name | `expense-tracker` |
| Transport | `stdio` |
| Command | `uv run --project <path> expense-tracker-mcp` |
| Tool binding | `llm.bind_tools(mcp_tools)` |
| Fallback | Plain LLM if tools list is empty |

### Tool Dispatch Flow

```
chat_node invoked
     |
     v
llm_with_tools.ainvoke(messages)
     |
     +--[ tool_use in response? ]
     |         |
     |        YES
     |         v
     |   MCP Client dispatches to expense-tracker server
     |   (stdio subprocess via uv run)
     |         |
     |         v
     |   ToolMessage result added to messages
     |         |
     +<--------+
     |
     v
Final AI response returned
```

---

## 6. Async / Event Loop Design

This is the most critical architectural decision in the project. All async operations share a single persistent background event loop to ensure that `AsyncSqliteSaver` and its internal `asyncio.Lock` are always called from the same loop that created them.

### 6.1 Why a Persistent Background Loop?

- `asyncio.run()` creates a **new loop each call** — the checkpointer's internal `Lock` becomes bound to the wrong loop
- Streamlit reruns the script on every interaction — it cannot own a persistent loop
- A daemon thread running `_loop.run_forever()` solves both problems

### 6.2 Thread Safety Boundary

```
STREAMLIT MAIN THREAD (sync)          BACKGROUND LOOP THREAD (async)
─────────────────────────────         ──────────────────────────────
run_in_loop(coro)                      _loop.run_forever()
  │                                         │
  ├─ run_coroutine_threadsafe() ──────────> coroutine executes here
  │                                         │
  └─ future.result() blocks ─────────────< result returned

stream_response() generator            _stream() async generator
  │                                         │
  └─ run_coroutine_threadsafe() ──────────> chunks pushed to Queue
  chunk_queue.get() iterates <──────────── chunk_queue.put(chunk)
```

---

## 7. Configuration Reference

### 7.1 LangGraph Run Config

```python
CONFIG = {
    "configurable": { "thread_id": st.session_state.thread_id },
    "metadata":     { "thread_id": ..., "session": "streamlit_chat" },
    "tags":         ["streamlit", "langgraph", "chatbot"],
    "run_name":     "chatBot-analysis"
}
```

### 7.2 Environment Variables

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Authentication for `ChatOpenAI` (loaded via `python-dotenv`) |
| `LANGCHAIN_API_KEY` | LangSmith tracing (optional) |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing — set to `true` |
| `LANGCHAIN_PROJECT` | LangSmith project name for trace grouping |

### 7.3 Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | Graph state machine orchestration |
| `langchain-openai` | OpenAI LLM integration |
| `langchain-mcp-adapters` | MCP tool loading and binding |
| `langsmith` | `@traceable` decorator for observability |
| `aiosqlite` | Async SQLite driver for checkpointer |
| `streamlit` | Web UI framework |
| `python-dotenv` | Environment variable loading |

### 7.4 Installation

```bash
pip install langgraph langchain-openai langchain-mcp-adapters \
            langsmith aiosqlite streamlit python-dotenv
```

### 7.5 Running the App

```bash
streamlit run chatbot_v5.py
```

---

## 8. Known Issues & Recommended Improvements

### 8.1 Known Issues

- **`run_async` NameError** — `load_mcp_tools()` calls `run_async()` which is not defined in the provided code. This will raise a `NameError` at startup. Should use `run_in_loop()` from `chatbot_v5.py` or inline asyncio logic.
- **`BaseTool` not imported** — the type annotation in `load_mcp_tools()` references `BaseTool` which is never imported — will raise `NameError`.
- **Typo** — `retrive_all_thread` should be `retrieve_all_threads` for clarity.
- **Performance** — `get_chat_title()` calls `load_conversation()` for every sidebar thread on every Streamlit rerun, which can be expensive with many threads.

### 8.2 Recommended Improvements

- Cache `get_chat_title()` results in session state to avoid repeated DB reads
- Add a `ToolNode` to the LangGraph so tool results are handled within the graph (currently tools are bound but not executed in the graph loop)
- Add error boundaries in the streaming path to surface LLM errors gracefully
- Implement conversation search / filter in the sidebar
- Add session expiry or pagination for the thread list
- Switch to `gpt-4o-mini` or `gpt-4o` for better reasoning quality

---

## 9. File Structure

```
project/
├── chat_model_v4.py        # Async core: LLM, graph, checkpointer, MCP
├── chatbot_v5.py           # Streamlit UI + event-loop bridge
├── chatbot.db              # SQLite persistence (auto-created at runtime)
├── .env                    # API keys (OPENAI_API_KEY, etc.)
└── requirements.txt        # Python dependencies

External:
└── expense_tracker_mcp/    # Separate uv project for MCP server
     └── (managed by uv at D:\shamik\projects\expencess_tracker_mcp)
```

---
