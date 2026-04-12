import streamlit as st
from chat_model_v3 import chatbot, retrive_all_thread
from langchain_core.messages import HumanMessage
import uuid


# -----------------------------
# Utility Functions
# -----------------------------

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
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


def get_chat_title(thread_id):
    """
    Extract first user message to use as conversation title
    """
    messages = load_conversation(thread_id)

    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content[:30] + "..."

    return "New Chat"


# -----------------------------
# Session State Initialization
# -----------------------------

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_thread" not in st.session_state:
    st.session_state.chat_thread = retrive_all_thread()
    add_thread(st.session_state.thread_id)


# -----------------------------
# Sidebar
# -----------------------------

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

            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            temp_message.append({
                "role": role,
                "content": msg.content
            })

        st.session_state.message_history = temp_message


# -----------------------------
# Chat History
# -----------------------------

for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Chat Input
# -----------------------------

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
    "run_name":"chatBot-analysis"
}


# -----------------------------
# Streaming Response
# -----------------------------

def stream_response():
    for message_chunk, metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=CONFIG,
        stream_mode="messages"
    ):
        if message_chunk.content:
            yield message_chunk.content


# -----------------------------
# Handle User Message
# -----------------------------

if user_input:

    # Save user message
    st.session_state.message_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        ai_message = st.write_stream(stream_response)

    # Save assistant response
    st.session_state.message_history.append(
        {"role": "assistant", "content": ai_message}
    )