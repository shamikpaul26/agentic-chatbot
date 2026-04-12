import streamlit as st
from chat_model_v4 import get_chatbot, retrive_all_thread
from langchain_core.messages import HumanMessage
import uuid
import asyncio
import threading
from queue import Queue, Empty


# ================================================================
# Persistent background event loop
#
# A single asyncio loop runs in a dedicated daemon thread for the
# lifetime of the Streamlit process.  Every async operation is
# submitted to this loop via run_in_loop() / stream_in_loop().
#
# Why not asyncio.run() per call?
#   asyncio.run() creates a NEW loop each time.  AsyncSqliteSaver
#   and its internal asyncio.Lock are bound to whichever loop first
#   called setup().  Calling them from a different loop raises:
#     "RuntimeError: <Lock> is bound to a different event loop"
#
# By routing everything through one persistent loop we guarantee
# the checkpointer, the Lock, and every coroutine all share the
# same loop.
# ================================================================

_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
_loop_thread = threading.Thread(target=_loop.run_forever, daemon=True)
_loop_thread.start()


def run_in_loop(coro):
    """Submit a coroutine to the background loop and block until it returns."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result()


# ================================================================
# Utility Functions
# ================================================================

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state.chat_thread:
        st.session_state.chat_thread.append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state.thread_id = thread_id
    add_thread(thread_id)
    st.session_state.message_history = []


def load_conversation(thread_id):
    async def _load():
        chatbot = await get_chatbot()
        state = await chatbot.aget_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        return state.values.get("messages", [])

    return run_in_loop(_load())


def get_chat_title(thread_id):
    messages = load_conversation(thread_id)
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content[:30] + "..."
    return "New Chat"


# ================================================================
# Session State Initialization
# ================================================================

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_thread" not in st.session_state:
    # retrive_all_thread is async — submit to the background loop
    st.session_state.chat_thread = run_in_loop(retrive_all_thread())
    add_thread(st.session_state.thread_id)


# ================================================================
# Sidebar
# ================================================================

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.header("Conversations")

for thread_id in st.session_state.chat_thread:
    title = get_chat_title(thread_id)
    if st.sidebar.button(title, key=thread_id):
        st.session_state.thread_id = thread_id
        messages = load_conversation(thread_id)

        temp_message = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_message.append({"role": role, "content": msg.content})

        st.session_state.message_history = temp_message


# ================================================================
# Chat History
# ================================================================

for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ================================================================
# Chat Input
# ================================================================

user_input = st.chat_input("Type your message")

CONFIG = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    },
    "metadata": {
        "thread_id": st.session_state.thread_id,
        "session": "streamlit_chat",
    },
    "tags": ["streamlit", "langgraph", "chatbot"],
    "run_name": "chatBot-analysis"
}


# ================================================================
# Streaming Response
#
# The async generator runs inside the background loop and pushes
# each text chunk into a Queue.  The sync generator below pulls
# from that Queue so st.write_stream receives a normal iterator.
# A sentinel value of None signals end-of-stream.
# ================================================================

def stream_response():
    chunk_queue: Queue = Queue()

    async def _stream():
        try:
            chatbot = await get_chatbot()
            async for message_chunk, metadata in chatbot.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if message_chunk.content:
                    chunk_queue.put(message_chunk.content)
        except Exception as e:
            chunk_queue.put(f"\n\n[Error: {e}]")
        finally:
            chunk_queue.put(None)   # sentinel

    asyncio.run_coroutine_threadsafe(_stream(), _loop)

    while True:
        chunk = chunk_queue.get()
        if chunk is None:
            break
        yield chunk


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
        ai_message = st.write_stream(stream_response)

    st.session_state.message_history.append(
        {"role": "assistant", "content": ai_message}
    )