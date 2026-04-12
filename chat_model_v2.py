from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from langsmith import traceable

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)

class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@traceable(name="chat_node")
def chat_node(state: ChatbotState):
    messages = state["messages"]
    response = llm.invoke(messages)

    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatbotState)

# Nodes
graph.add_node("chat_node", chat_node)

# Edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrive_all_thread():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
        
    return list(all_threads)
