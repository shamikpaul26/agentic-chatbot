"""
chatbot_v5.py  —  PERMANENT FIX

See chat_model_v5.py for the full root-cause write-up.

SUMMARY
-------
Previously used `AsyncSqliteSaver`, whose sync methods raise
`InvalidStateError` when invoked from inside its own captured loop.
LangGraph's pregel engine has internal sync `checkpointer.get_tuple(...)`
calls that run on that same loop during streaming → error on every turn.

Fix: switched to sync `SqliteSaver` in chat_model_v5.py. That saver has
no loop-affinity trap and exposes both sync and async APIs.

Consequences for this file:
  * `get_all_threads()` uses `_checkpointer.list(None)` directly (sync,
    safe from any thread — no need to hop onto the background loop).
  * `load_conversation()` keeps using `_graph.aget_state(...)` via
    `run_in_loop(...)` — still async, still works, just now backed by a
    checkpointer that doesn't blow up.
  * The streaming path is unchanged (it was never the bug — the bug was
    an internal sync-get_tuple call inside pregel that AsyncSqliteSaver
    rejected).

The @st.cache_resource singleton for (loop, graph, checkpointer) remains.
Loop and graph are created once per process and never recreated.
"""

import streamlit as st
from chat_model_v5 import init_chatbot
from crypto_subgraph import decrypt_message
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import uuid
import asyncio
import threading
from queue import Queue


# ================================================================
# PERMANENT SINGLETON — loop + chatbot, created once, cached forever
# ================================================================

@st.cache_resource
def get_resources():
    """
    Runs ONCE per process. st.cache_resource guarantees this.
    Creates the event loop and initialises ALL resources on it.
    Every subsequent Streamlit rerun gets this cached result.
    """
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True, name="chatbot-loop")
    t.start()

    # Run async init ON that loop — blocks here until complete
    future = asyncio.run_coroutine_threadsafe(init_chatbot(), loop)
    graph, checkpointer, db_conn, mcp_client = future.result(timeout=30)

    print(f"[Init] Resources created on loop {id(loop)} thread {t.name}")
    return loop, graph, checkpointer


_loop, _graph, _checkpointer = get_resources()


def run_in_loop(coro):
    """Submit a coroutine to the persistent background loop. Thread-safe."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result()


# ================================================================
# Decrypt helper
# ================================================================

def safe_decrypt(content: str) -> str:
    try:
        return decrypt_message(content)
    except Exception:
        return content


# ================================================================
# Utility Functions
# ================================================================

def generate_thread_id() -> str:
    return str(uuid.uuid4())


def add_thread(thread_id: str):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state.thread_id = thread_id
    st.session_state.message_history = []
    add_thread(thread_id)


def load_conversation(thread_id: str) -> list[dict]:
    async def _load():
        state = await _graph.aget_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        return state.values.get("messages", []) if state and state.values else []

    try:
        messages = run_in_loop(_load())
    except Exception as e:
        print(f"[load_conversation] thread {thread_id}: {e}")
        return []

    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({
                "role": "user",
                "content": safe_decrypt(msg.content)
            })
        elif (
            isinstance(msg, AIMessage)
            and msg.content
            and not getattr(msg, "tool_calls", None)
        ):
            result.append({
                "role": "assistant",
                "content": safe_decrypt(msg.content)
            })
    return result


def get_chat_title(thread_id: str) -> str:
    for msg in load_conversation(thread_id):
        if msg["role"] == "user":
            return msg["content"][:30] + "..."
    return "New Chat"


def get_all_threads() -> list[str]:
    """
    Sync iteration — SqliteSaver.list() is thread-safe and works from
    the main Streamlit thread without hopping onto the background loop.
    """
    try:
        threads = set()
        for checkpoint in _checkpointer.list(None):
            tid = checkpoint.config.get("configurable", {}).get("thread_id")
            if tid:
                threads.add(tid)
        return list(threads)
    except Exception as e:
        print(f"[get_all_threads] {e}")
        return []


# ================================================================
# Session State Init
# ================================================================

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = get_all_threads()
    add_thread(st.session_state.thread_id)


# ================================================================
# Sidebar
# ================================================================

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.header("Conversations")

for thread_id in st.session_state.chat_threads:
    title = get_chat_title(thread_id)
    if st.sidebar.button(title, key=thread_id):
        st.session_state.thread_id = thread_id
        st.session_state.message_history = load_conversation(thread_id)


# ================================================================
# Chat History
# ================================================================

for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ================================================================
# Chat Input & Config
# ================================================================

user_input = st.chat_input("Type your message")

CONFIG = {
    "configurable": {"thread_id": st.session_state.thread_id},
    "metadata":     {"thread_id": st.session_state.thread_id, "session": "streamlit"},
    "tags":         ["streamlit", "langgraph", "chatbot"],
    "run_name":     "chatBot-analysis"
}


# ================================================================
# Streaming — collect full encrypted token, decrypt once, stream words
# ================================================================

def stream_response():
    chunk_queue: Queue = Queue()

    async def _stream():
        try:
            async for chunk, metadata in _graph.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if (
                    chunk.content
                    and not isinstance(chunk, ToolMessage)
                    and not getattr(chunk, "tool_calls", None)
                    and metadata.get("langgraph_node") == "chat_node"
                ):
                    chunk_queue.put(chunk.content)
        except Exception as e:
            chunk_queue.put(e)
        finally:
            chunk_queue.put(None)

    future = asyncio.run_coroutine_threadsafe(_stream(), _loop)

    full_encrypted = ""
    while True:
        item = chunk_queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        full_encrypted += item

    # Surface any late exception from the background task
    try:
        future.result(timeout=5)
    except Exception as e:
        print(f"[stream_response] background task error: {e}")

    full_plain = safe_decrypt(full_encrypted) if full_encrypted else ""
    if not full_plain:
        yield ""
        return

    words = full_plain.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")


# ================================================================
# Handle User Message
# ================================================================

if user_input:
    st.session_state.message_history.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            ai_message = st.write_stream(stream_response)
            st.session_state.message_history.append(
                {"role": "assistant", "content": ai_message}
            )
        except Exception as e:
            st.error(f"Error: {e}")