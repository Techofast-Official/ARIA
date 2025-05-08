import os
import operator
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Dict
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ChatMessage, ToolMessage, AnyMessage
)
from langgraph.graph import StateGraph, END, MessagesState,START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode

from Insight_tools import (
    issue_categorization, product_categorization,
    issue_resolution_identification, sentiment_analysis_summarization,
    conversation_summary
)

load_dotenv()

# Instantiate Groq LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

# Tools
tools = [
    issue_categorization,
    product_categorization,
    issue_resolution_identification,
    sentiment_analysis_summarization,
    conversation_summary,
]

# Bind tools to the model
llm = llm.bind_tools(tools)

class AgentState(Dict):
    task: str
    conv: str
    critique: str
    insights: Dict[str, str]
    messages: Annotated[list[AnyMessage], operator.add]

EXECUTOR_PROMPT = """You are a smart Conversation Analysis assistant expert in extracting and compiling conversation insights. \
You will be provided with conversation between a customer and a customer support representative delimited by ###.
You will be provided the insight attributes that are desired to be extracted from the conversations delimited by ***.
You can use the tools you have to extract the desired insights about the conversations.
Compile and reply with all the desired Insights.
"""

class Insights(TypedDict):
    insights: Dict[str, str]

def insights_manager(state: AgentState):
    insights = {}
    messages = state['messages']

    if len(messages) == 0:
        human = f"""###\n{state['conv']}\n###\n***\n{state['task']}\n***"""
        messages = [
            SystemMessage(content=EXECUTOR_PROMPT),
            HumanMessage(content=human)
        ]

    response = llm.invoke(messages)
    state['tool_calls'] = response.tool_calls
    tool_ctr = len(state['tool_calls'])

    if tool_ctr == 0:
        insights = llm.with_structured_output(Insights).invoke([
            SystemMessage(content=state['task'] + " Format it if already available"),
            HumanMessage(content="|".join([str(item.content) for item in messages]))
        ])
        state['insights'] = insights

    return {"messages": [response], "insights": insights}

# Define Graph
builder = StateGraph(AgentState)
builder.add_node("insights_manager", insights_manager)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "insights_manager")
builder.add_conditional_edges("insights_manager", tools_condition)
builder.add_edge("tools", "insights_manager")

react_graph = builder.compile()
