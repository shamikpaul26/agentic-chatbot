"""
chat_model_v5.py  —  PERMANENT FIX + SHORT-TERM MEMORY SUBGRAPH

ROOT CAUSE OF THE ORIGINAL ERROR
--------------------------------
Error:
  "Synchronous calls to AsyncSqliteSaver are only allowed from a different
   thread. From the main thread, use the async interface."

`AsyncSqliteSaver` captures `self.loop = asyncio.get_running_loop()` at
construction. Its SYNC methods (`get_tuple` / `list` / `delete_thread`)
raise `InvalidStateError` when called from INSIDE that same loop.

LangGraph's pregel engine has sync code paths (pregel/_loop.py,
pregel/main.py) that internally call `checkpointer.get_tuple(...)`
SYNCHRONOUSLY even when the outer call is `graph.astream(...)`. Because
our background thread runs the async graph on `self.loop`, those
internal sync `get_tuple` calls trigger the main-thread check and raise
— producing the error on every chat turn.

Swapping to plain `SqliteSaver` doesn't work either — its `aget_tuple`
/ `alist` / `aput` raise `NotImplementedError`, so `graph.astream` and
`graph.aget_state` break.

THE FIX — HybridSqliteSaver
---------------------------
Subclass `SqliteSaver` and add async methods that delegate to the sync
ones via `asyncio.to_thread`. Best of both worlds:

  * Sync methods (`get_tuple`, `list`, `put`, `put_writes`): inherited
    from `SqliteSaver` — thread-safe, no loop-affinity trap.
  * Async methods (`aget_tuple`, `alist`, `aput`, `aput_writes`):
    implemented here by dispatching to the sync methods on a worker
    thread. LangGraph's `astream` / `aget_state` see a fully functional
    async checkpointer.
  * Pregel's internal SYNC `get_tuple()` calls — which previously
    crashed `AsyncSqliteSaver` — now just run the sync path directly.
    No `InvalidStateError`, no thread-mismatch, ever.

We open the sqlite connection with `check_same_thread=False` so the same
connection is usable from the main Streamlit thread, the background
loop thread, and the `to_thread` worker threads.

SHORT-TERM MEMORY SUBGRAPH
---------------------------
A dedicated `memory_subgraph` sits between message ingestion and the
chat node. It runs ONLY when total token count > TOKEN_LIMIT (5000).

Algorithm:
  1. Count tokens for all messages (system excluded).
  2. If count <= TOKEN_LIMIT → pass through unchanged.
  3. If count > TOKEN_LIMIT:
       a. Pop the oldest messages (oldest-first) until we are back
          under TOKEN_LIMIT.
       b. Ask the LLM to summarize the popped messages into one
          compact paragraph.
       c. Merge the new summary with any pre-existing summary.
       d. Store the merged summary in state["summary"].
       e. Return the trimmed messages list.
  4. chat_node reads state["summary"] and prepends it as a
     SystemMessage BEFORE the main SYSTEM_PROMPT so the LLM always
     has the context of what was discussed earlier.

No existing logic is modified — the subgraph is inserted as a new node
between "encrypt_input" and "chat_node".
"""

import threading
import json
import asyncio
import sqlite3
from typing import Any, AsyncIterator, Optional, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import RemoveMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.base import (
    Checkpoint, CheckpointMetadata, CheckpointTuple, ChannelVersions,
)
from langchain_core.runnables import RunnableConfig
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
)
from langchain_core.tools import BaseTool
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langsmith import traceable
from langchain_mcp_adapters.client import MultiServerMCPClient
from crypto_subgraph import encrypt_message, decrypt_message

load_dotenv()

# ===============================
# LLM
# ===============================

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

MCP_CONFIG = {
    "expense-tracker": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run",
            "--project",
            "D:\\shamik\\projects\\expencess_tracker_mcp",
            "expense-tracker-mcp"
        ]
    }
}

# ===============================
# MEMORY CONFIG
# ===============================

# Token budget for the conversation history (system prompt excluded).
# When total tokens exceed this, oldest messages are pruned and
# summarised into state["summary"].
TOKEN_LIMIT = 5000


def _count_tokens(text: str) -> int:
    """
    Lightweight token estimator (~4 chars ≈ 1 token for English text).
    Avoids a network round-trip to tiktoken's BPE file while staying
    accurate enough for the 5 000-token trim decision.
    Each message also carries ~4 tokens of structural overhead
    (role, separators) — callers add that separately.
    """
    return max(1, len(text) // 4)


def _messages_token_count(messages: list[BaseMessage]) -> int:
    """Return total estimated tokens for a list of messages."""
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += _count_tokens(content) + 4   # 4-token per-message overhead
    return total


# ===============================
# STATE
# ===============================

class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # Accumulated summary of pruned messages.  None until first prune.
    summary: Optional[str]


# ===============================
# SYSTEM PROMPT
# ===============================

SYSTEM_PROMPT = SystemMessage(content="""You are a helpful general-purpose assistant with access to an expense tracker tool.

STRICT TOOL USAGE RULES:
- ONLY use expense tracker tools when the user explicitly asks to add, view,
  update or delete expenses, check budgets, spending summaries, or manage categories.
- For ALL other requests respond directly. Do NOT call any tools.

Summarize expense results in plain English. Never show raw JSON.""")

# ===============================
# SANITIZER
# ===============================

def sanitize_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    responded_ids = {
        msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage)
    }
    clean = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if not all(tc["id"] in responded_ids for tc in msg.tool_calls):
                print(f"[Sanitizer] Dropping orphaned tool_call: {msg.tool_calls}")
                continue
        clean.append(msg)
    return clean

# ===============================
# ENCRYPTION
# ===============================

def encrypt_content(content) -> str:
    if isinstance(content, list):
        content = json.dumps(content)
    if not content:
        return content
    return encrypt_message(str(content))


def decrypt_content(content: str) -> str:
    if not content:
        return content
    return decrypt_message(content)


def decrypt_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    out = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            out.append(HumanMessage(content=decrypt_content(msg.content), id=msg.id))
        elif isinstance(msg, AIMessage):
            out.append(AIMessage(
                content=decrypt_content(msg.content) if msg.content else "",
                tool_calls=msg.tool_calls, id=msg.id))
        elif isinstance(msg, ToolMessage):
            out.append(ToolMessage(
                content=decrypt_content(msg.content),
                tool_call_id=msg.tool_call_id, id=msg.id))
        else:
            out.append(msg)
    return out


# ===============================
# MEMORY SUBGRAPH
# ===============================
# The subgraph operates on the same ChatbotState so it can be compiled
# as an inner graph and called as a node in the main graph.
#
# Nodes:
#   check_tokens  → decides whether trimming is needed
#   summarize     → called only when tokens > TOKEN_LIMIT
#
# Edges:
#   START → check_tokens
#   check_tokens --[need_summary]--> summarize → END
#   check_tokens --[ok]-----------> END

def _build_memory_subgraph(summarizer_llm: ChatOpenAI) -> Any:
    """
    Build and compile a LangGraph subgraph that trims messages exceeding
    TOKEN_LIMIT and generates / merges a running summary.

    Parameters
    ----------
    summarizer_llm : ChatOpenAI
        The LLM instance to use for summarisation (same as main llm).

    Returns
    -------
    Compiled subgraph (Runnable) that can be used as a node.
    """

    async def check_tokens_node(state: ChatbotState) -> dict:
        """
        Count tokens.  If within budget, return unchanged.
        If over budget, trim oldest messages using RemoveMessage (the
        correct LangGraph deletion pattern) and stash pruned messages
        in the module-level scratch dict so summarize_node can read them
        within the same subgraph invocation.
        """
        messages = state["messages"]
        total_tokens = _messages_token_count(messages)

        if total_tokens <= TOKEN_LIMIT:
            # Nothing to do — pass through unchanged
            return {}

        # ---- trim oldest messages until we are back under budget ----
        to_summarise: list[BaseMessage] = []
        remaining: list[BaseMessage] = list(messages)

        while remaining and _messages_token_count(remaining) > TOKEN_LIMIT:
            to_summarise.append(remaining.pop(0))

        # Guard: never pop the very last message (the new user turn)
        if not remaining:
            remaining = [to_summarise.pop()]

        # Stash pruned messages so summarize_node can read them
        _memory_subgraph_scratch["to_summarise"]  = to_summarise
        _memory_subgraph_scratch["needs_summary"] = True

        # RemoveMessage is the correct LangGraph API to delete state entries.
        # Each RemoveMessage(id=x) instructs add_messages to drop that id.
        removals = [RemoveMessage(id=m.id) for m in to_summarise if m.id]
        return {"messages": removals}

    async def summarize_node(state: ChatbotState) -> dict:
        """
        Summarise the pruned messages, merge with any prior summary,
        and persist back into state["summary"].
        Only called when _memory_subgraph_scratch["needs_summary"] is True.
        """
        to_summarise: list[BaseMessage] = _memory_subgraph_scratch.pop(
            "to_summarise", []
        )
        _memory_subgraph_scratch.pop("needs_summary", None)
        _memory_subgraph_scratch.pop("trimmed_messages", None)

        if not to_summarise:
            return {}

        # Decrypt before summarising (messages are stored encrypted)
        decrypted = decrypt_messages(to_summarise)

        # Build a concise transcript for the summariser
        transcript_parts = []
        for msg in decrypted:
            role = (
                "User" if isinstance(msg, HumanMessage)
                else "Assistant" if isinstance(msg, AIMessage)
                else "Tool"
            )
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.strip():
                transcript_parts.append(f"{role}: {content.strip()}")

        transcript = "\n".join(transcript_parts)

        prior_summary: Optional[str] = state.get("summary")

        if prior_summary:
            prompt = (
                f"You are maintaining a running summary of a conversation.\n\n"
                f"EXISTING SUMMARY:\n{prior_summary}\n\n"
                f"NEW CONVERSATION SEGMENT TO INCORPORATE:\n{transcript}\n\n"
                f"Write a single concise paragraph (≤120 words) that merges "
                f"the existing summary with the new segment. Capture all "
                f"important context, decisions, and facts. Be terse."
            )
        else:
            prompt = (
                f"Summarise the following conversation segment into a single "
                f"concise paragraph (≤120 words). Capture the key topics, "
                f"decisions, and facts discussed. Be terse.\n\n"
                f"CONVERSATION:\n{transcript}"
            )

        response = await summarizer_llm.ainvoke([HumanMessage(content=prompt)])
        new_summary = response.content.strip()

        print(f"[MemorySubgraph] Summary updated ({len(to_summarise)} msgs pruned). "
              f"Summary: {new_summary[:80]}…")

        return {"summary": new_summary}

    def _needs_summary(state: ChatbotState) -> str:
        """Routing function: did check_tokens_node flag a summary is needed?"""
        return "summarize" if _memory_subgraph_scratch.get("needs_summary") else END

    # Build subgraph
    subgraph = StateGraph(ChatbotState)
    subgraph.add_node("check_tokens", check_tokens_node)
    subgraph.add_node("summarize",    summarize_node)

    subgraph.add_edge(START, "check_tokens")
    subgraph.add_conditional_edges(
        "check_tokens",
        _needs_summary,
        {"summarize": "summarize", END: END},
    )
    subgraph.add_edge("summarize", END)

    return subgraph.compile()


# Module-level scratch space for inter-node communication within the
# memory subgraph (same Python process, single-threaded event loop).
_memory_subgraph_scratch: dict = {}


# ===============================
# GRAPH BUILDER
# ===============================

def _build_graph(checkpointer, llm_with_tools, tools):

    @traceable(name="encrypt_input_node")
    async def encrypt_input_node(state: ChatbotState):
        messages = state["messages"]
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return {"messages": messages[:-1] + [
                HumanMessage(content=encrypt_content(last.content), id=last.id)
            ]}
        return {"messages": messages}

    # Build the memory subgraph once (uses same llm for summarisation)
    memory_subgraph = _build_memory_subgraph(llm)

    @traceable(name="chat_node")
    async def chat_node(state: ChatbotState):
        messages = decrypt_messages(sanitize_messages(state["messages"]))

        # ---- Prepend summary as context (if present) ----
        # Inserted BEFORE the main SYSTEM_PROMPT so the LLM reads:
        #   [summary context] → [main system prompt] → [conversation]
        preamble: list[BaseMessage] = []
        summary: Optional[str] = state.get("summary")
        if summary:
            preamble.append(SystemMessage(
                content=f"[CONVERSATION HISTORY SUMMARY]\n{summary}"
            ))

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = preamble + [SYSTEM_PROMPT] + messages
        else:
            # System prompt already present (edge case) — insert summary before it
            messages = preamble + messages

        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [AIMessage(
            content=encrypt_content(response.content) if response.content else "",
            tool_calls=response.tool_calls,
            id=response.id
        )]}

    @traceable(name="encrypt_tool_result_node")
    async def encrypt_tool_result_node(state: ChatbotState):
        messages = state["messages"]
        last = messages[-1]
        if isinstance(last, ToolMessage):
            return {"messages": messages[:-1] + [
                ToolMessage(
                    content=encrypt_content(last.content),
                    tool_call_id=last.tool_call_id,
                    id=last.id
                )
            ]}
        return {"messages": messages}

    graph = StateGraph(ChatbotState)
    graph.add_node("encrypt_input",       encrypt_input_node)
    graph.add_node("memory",              memory_subgraph)      # ← NEW SUBGRAPH NODE
    graph.add_node("chat_node",           chat_node)
    graph.add_node("tools",               ToolNode(tools))
    graph.add_node("encrypt_tool_result", encrypt_tool_result_node)

    graph.add_edge(START,                 "encrypt_input")
    graph.add_edge("encrypt_input",       "memory")             # ← route through memory
    graph.add_edge("memory",              "chat_node")          # ← then to chat
    graph.add_conditional_edges("chat_node", tools_condition,
                                {"tools": "tools", END: END})
    graph.add_edge("tools",               "encrypt_tool_result")
    graph.add_edge("encrypt_tool_result", "chat_node")
    return graph.compile(checkpointer=checkpointer)


# ===============================
# HYBRID CHECKPOINTER
# Sync SqliteSaver + async methods that delegate via asyncio.to_thread.
# Protected by an asyncio.Lock to serialise concurrent checkpoint writes
# on the same sqlite3.Connection (SQLite is fine with one connection
# across threads when check_same_thread=False, but we still want to
# avoid overlapping transactions from concurrent pregel writes).
# ===============================

class HybridSqliteSaver(SqliteSaver):
    """
    SqliteSaver with working async methods.

    LangGraph's async graph calls (`astream`, `aget_state`, `ainvoke`)
    use `aget_tuple` / `alist` / `aput` / `aput_writes`. The stock
    `SqliteSaver` raises NotImplementedError for those. We override
    them to dispatch to the sync implementations on a worker thread
    via `asyncio.to_thread`, which preserves async semantics and keeps
    pregel's own SYNC `get_tuple()` calls (the ones that crashed
    AsyncSqliteSaver) on the fast path — direct sync, no loop trap.
    """

    def __init__(self, conn: sqlite3.Connection, *args, **kwargs):
        super().__init__(conn, *args, **kwargs)
        # Lock to serialise writes to the single shared sqlite connection
        self._write_lock = threading.Lock()

    # ---- reads ----
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        # Materialise in a worker thread, then yield results on the loop.
        # pregel typically calls alist with small result sets.
        def _collect() -> list[CheckpointTuple]:
            return list(self.list(config, filter=filter, before=before, limit=limit))

        items = await asyncio.to_thread(_collect)
        for item in items:
            yield item

    # ---- writes ----
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        def _do():
            with self._write_lock:
                return self.put(config, checkpoint, metadata, new_versions)
        return await asyncio.to_thread(_do)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        def _do():
            with self._write_lock:
                return self.put_writes(config, writes, task_id, task_path)
        return await asyncio.to_thread(_do)

    async def adelete_thread(self, thread_id: str) -> None:
        def _do():
            with self._write_lock:
                return self.delete_thread(thread_id)
        return await asyncio.to_thread(_do)


# ===============================
# ONE-TIME ASYNC INIT
# Called exactly once from chatbot_v5.py's st.cache_resource block.
# ===============================

async def init_chatbot():
    """
    Create ALL resources. Called once from the background loop.
    Returns (graph, checkpointer, db_conn, mcp_client).

    Uses HybridSqliteSaver: sync SqliteSaver methods (no loop-affinity
    trap) + async methods (so graph.astream / graph.aget_state work).
    """
    # --- MCP ---
    mcp_client = MultiServerMCPClient(MCP_CONFIG)
    try:
        mcp_tools = await mcp_client.get_tools()
        print(f"[MCP] Loaded {len(mcp_tools)} tools: {[t.name for t in mcp_tools]}")
    except Exception as e:
        print(f"[Warning] MCP unavailable: {e}")
        mcp_tools = []

    llm_with_tools = llm.bind_tools(mcp_tools) if mcp_tools else llm

    # --- Checkpointer ---
    # check_same_thread=False → connection usable from ANY thread.
    # isolation_level=None → autocommit (matches aiosqlite default).
    db_conn      = sqlite3.connect(
        "chatbot.db",
        check_same_thread=False,
        isolation_level=None,
    )
    checkpointer = HybridSqliteSaver(db_conn)
    checkpointer.setup()

    graph = _build_graph(checkpointer, llm_with_tools, mcp_tools)

    return graph, checkpointer, db_conn, mcp_client