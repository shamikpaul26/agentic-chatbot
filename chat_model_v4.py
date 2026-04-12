import asyncio
import threading
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langsmith import traceable
from langchain_mcp_adapters.client import MultiServerMCPClient
import aiosqlite

load_dotenv()

# ===============================
# LLM
# ===============================

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


client = MultiServerMCPClient(
    {
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
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []


mcp_tools = load_mcp_tools()

tools = [*mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm



# ===============================
# STATE
# ===============================

class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ===============================
# CHAT NODE
# ===============================

@traceable(name="chat_node")
async def chat_node(state: ChatbotState):
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

# ===============================
# GRAPH BUILDER
# ===============================

def build_graph(checkpointer):
    graph = StateGraph(ChatbotState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)
    return graph.compile(checkpointer=checkpointer)

# ===============================
# DATABASE & GRAPH — lazy singleton
#
# All async resources (aiosqlite connection, AsyncSqliteSaver and
# its internal asyncio.Lock) are created inside get_chatbot(), which
# is always called via run_in_loop() in chatbot_v5.py.  That means
# they are created on — and permanently bound to — the single
# persistent background loop, never Streamlit's main-thread loop.
#
# _init_lock uses threading.Lock, NOT asyncio.Lock.
# asyncio.Lock() captures the running loop at creation time.
# At module-import time Streamlit's loop is running, so an
# asyncio.Lock created here would be bound to the wrong loop and
# raise "bound to a different event loop" the first time
# get_chatbot() tries to acquire it on the background loop.
# threading.Lock has no event-loop binding whatsoever.
# ===============================

_chatbot      = None
_checkpointer = None
_init_lock    = threading.Lock()        # ← threading, NOT asyncio


async def get_chatbot():
    """
    Return the singleton compiled graph, initialising it on first call.
    threading.Lock is acquired synchronously (non-blocking for the loop)
    to prevent a double-init race between concurrent coroutines.
    """
    global _chatbot, _checkpointer

    if _chatbot is not None:            # fast path — no lock needed
        return _chatbot

    with _init_lock:                    # slow path — only runs once
        if _chatbot is None:
            db_conn       = await aiosqlite.connect("chatbot.db")
            _checkpointer = AsyncSqliteSaver(db_conn)
            _chatbot      = build_graph(_checkpointer)

    return _chatbot


# ===============================
# THREAD HISTORY
# ===============================

async def retrive_all_thread():
    await get_chatbot()                 # ensures _checkpointer is ready
    all_threads = set()
    async for checkpoint in _checkpointer.alist(None):
        all_threads.add(
            checkpoint.config["configurable"]["thread_id"]
        )
    return list(all_threads)