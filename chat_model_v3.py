from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from langsmith import traceable
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
import yfinance as yf
from textblob import TextBlob
import pandas as pd

load_dotenv()

# ===============================
# LLM
# ===============================

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# ===============================
# STATE
# ===============================

class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ===============================
# SEARCH TOOL
# ===============================

search_tool = TavilySearchResults()

# ===============================
# CALCULATOR TOOL
# ===============================

@traceable(name="calculator_tool")
def calculator_func(operation: str, numbers: List[float]) -> float:
    """
    A custom calculator that supports multiple operations.

    operation options:
    add, subtract, multiply, divide, power, mod, percentage
    """


    if operation == "add":
        return sum(numbers)

    if operation == "subtract":
        result = numbers[0]
        for n in numbers[1:]:
            result -= n
        return result

    if operation == "multiply":
        result = 1
        for n in numbers:
            result *= n
        return result

    if operation == "divide":
        result = numbers[0]
        for n in numbers[1:]:
            result /= n
        return result

    if operation == "power":
        return numbers[0] ** numbers[1]

    if operation == "mod":
        return numbers[0] % numbers[1]

    raise ValueError("Unsupported operation")

calculator = tool(calculator_func)

# ===============================
# STOCK PRICE TOOL
# ===============================

@traceable(name="stock_price_tool")
def stock_price_func(symbol: str):
    """Get latest stock price for NSE symbol"""
    stock = yf.Ticker(symbol + ".NS")
    data = stock.history(period="1d")

    return float(data["Close"].iloc[-1])

get_stock_price = tool(stock_price_func)

# ===============================
# MARKET SENTIMENT TOOL
# ===============================

@traceable(name="market_sentiment_tool")
def market_sentiment_func(symbol: str):
    """
    Analyze market sentiment for a stock using news headlines.
    """
    stock = yf.Ticker(symbol + ".NS")
    news = stock.news

    sentiments = []

    for article in news[:5]:
        headline = article["title"]
        polarity = TextBlob(headline).sentiment.polarity
        sentiments.append(polarity)

    if not sentiments:
        return "No news available"

    avg = sum(sentiments) / len(sentiments)

    if avg > 0:
        return "Bullish sentiment"
    elif avg < 0:
        return "Bearish sentiment"
    else:
        return "Neutral sentiment"

market_sentiment = tool(market_sentiment_func)

# ===============================
# TECHNICAL ANALYSIS TOOL
# ===============================

@traceable(name="technical_analysis_tool")
def technical_analysis_func(symbol: str):
    """
    Perform basic technical analysis for NSE stock.
    """

    stock = yf.download(symbol + ".NS", period="3mo")

    stock["SMA20"] = stock["Close"].rolling(20).mean()
    stock["SMA50"] = stock["Close"].rolling(50).mean()

    latest = stock.iloc[-1]

    signal = "Neutral"

    if latest["SMA20"] > latest["SMA50"]:
        signal = "Bullish trend"
    elif latest["SMA20"] < latest["SMA50"]:
        signal = "Bearish trend"

    return {
        "close_price": float(latest["Close"]),
        "SMA20": float(latest["SMA20"]),
        "SMA50": float(latest["SMA50"]),
        "signal": signal
    }

technical_analysis = tool(technical_analysis_func)

# ===============================
# PORTFOLIO ANALYZER TOOL
# ===============================

@traceable(name="portfolio_analyzer_tool")
def portfolio_analyzer_func(portfolio: dict):
    """
    Analyze portfolio value.
    Example input:
    {"RELIANCE":10,"TCS":5,"INFY":8}
    """

    total_value = 0
    stock_values = {}

    for symbol, qty in portfolio.items():

        price = yf.Ticker(symbol + ".NS").history(period="1d")["Close"].iloc[-1]

        value = price * qty

        stock_values[symbol] = value
        total_value += value

    weights = {
        k: round(v / total_value * 100, 2)
        for k, v in stock_values.items()
    }

    return {
        "total_portfolio_value": round(total_value, 2),
        "allocation_percent": weights
    }

portfolio_analyzer = tool(portfolio_analyzer_func)

# ===============================
# TOOL LIST
# ===============================

tools = [
    search_tool,
    calculator,
    get_stock_price,
    market_sentiment,
    technical_analysis,
    portfolio_analyzer
]

llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)

# ===============================
# CHAT NODE
# ===============================

@traceable(name="chat_node")
def chat_node(state: ChatbotState):

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

# ===============================
# TOOL ROUTER
# ===============================

@traceable(name="tool_router")
def tool_condition(state: ChatbotState):

    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END

# ===============================
# DATABASE
# ===============================

conn = sqlite3.connect("chatbot.db", check_same_thread=False)

checkpointer = SqliteSaver(conn)

# ===============================
# GRAPH
# ===============================

graph = StateGraph(ChatbotState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tool_condition,
)

graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# ===============================
# THREAD HISTORY
# ===============================

def retrive_all_thread():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        all_threads.add(
            checkpoint.config["configurable"]["thread_id"]
        )

    return list(all_threads)