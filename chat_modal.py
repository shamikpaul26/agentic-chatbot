from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)

class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatbotState):
    messages = state["messages"]
    response = llm.invoke(messages)

    return {"messages": [response]}


checkpointer = MemorySaver()

graph = StateGraph(ChatbotState)

# Nodes
graph.add_node("chat_node", chat_node)

# Edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
