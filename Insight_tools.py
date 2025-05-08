import os
from dotenv import load_dotenv

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ChatMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()

# Instantiate Groq model
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

# Issue Categorization Tool
@tool
def issue_categorization(conv: str) -> str:
    """
    This tool categorizes the conversation context.

    Parameters:
    - conv (str): The conversation context.

    Returns:
    - String containing the following :
      - issue_area: The area of the issue (Order, Login and Account, Shopping, Cancellations and returns, Warranty, Shipping).
      - issue_category: The category of the issue.
      - issue_sub_category: The sub-category of the issue.
      - issue_complexity: The complexity of the issue (medium, high, less).
    """
    prompt = """
    Categorize the following issue based on the conversation context.
    Provide the following details in a string:
    - issue_area: {Order, Login and Account, Shopping, Cancellations and returns, Warranty, Shipping,}
    - issue_category
    - issue_sub_category
    - issue_complexity: {medium, high, less}
    """
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=conv)])
    # print("issue Categorization Details: ",response.content)
    return str(response.content)


# Product Categorization Tool
@tool
def product_categorization(conv: str) -> str:
    """
    This tool categorizes the product based on the conversation context.

    Parameters:
    - conv (str): The conversation context.

    Returns:
    - String containing the following:
      - product_category: The category of the product (Electronics, Men/Women/Kids, Appliances).
      - product_sub_category: The sub-category of the product (e.g., Wet Grinder, Diaper, Printer, Ceiling Fan, Dishwasher, Headphone, Backpack).
    """
    prompt = """
    System: Categorize the product based on the conversation context.
    Provide the following details:
    - product_category: {Electronics, Men/Women/Kids, Appliances}
    - product_sub_category: {product type: like Wet Grinder, Diaper, Printer, Ceiling Fan, Dishwasher, Headphone, Backpack etc}
    """
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=conv)])
    # print("Product Categorization details:",response.content)
    return str(response.content)


# Issue and Resolution Identification Tool
@tool
def issue_resolution_identification(conv: str) -> str:
    """
    This tool identifies issues and resolutions based on the conversation context.

    Parameters:
    - conv (str): The conversation context.

    Returns:
    - String containing the following:
      - customer_pain_points: Specific issues or challenges raised by the customer.
      - solutions_proposed: Solutions or suggestions provided during the conversation.
      - followup_required: Any follow-up actions needed.
      - action_items: Tasks or actions decided upon during the conversation.
    """
    prompt = """
    System: Identify issues and resolutions based on the conversation context.
    Provide the following details in a String:
    - customer_pain_points: Specific issues or challenges raised by the customer.
    - solutions_proposed: Solutions or suggestions provided during the conversation.
    - followup_required: Any follow-up actions needed.
    - action_items: Tasks or actions decided upon during the conversation.
    """
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=conv)])
    # print("Issue details:",response.content)

    return str(response.content)


# Sentiment Analysis and Summarization Tool
@tool
def sentiment_analysis_summarization(conv: str) -> str:
    """
    This tool analyzes the sentiment and summarizes the conversation.

    Parameters:
    - conv (str): The conversation context.

    Returns:
    - String containing the following details:
      - customer_satisfaction: Whether the customer is satisfied or not (Yes/No).
      - satisfaction_level: A rating scale (e.g., 1-5 or 1-10) to quantify the customer's satisfaction level.
      - customer_sentiment_summary: A brief summary of the customer's overall sentiment throughout the conversation.
      - representative_tone: Representative's tone during the conversation.
    """
    prompt = """
    System: Analyze the sentiment and summarize the conversation.
    Provide the following details in a String:
    - customer_satisfaction: Whether the customer is satisfied or not (Yes/No).
    - satisfaction_level: A rating scale (e.g., 1-5 or 1-10) to quantify the customer's satisfaction level.
    - customer_sentiment_summary: A brief summary of the customer's overall sentiment throughout the conversation.
    - representative_tone: Representative's tone during the conversation.
    """
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=conv)])
    # print("Sentiment tool",response.content)

    return str(response.content)


# Conversation Summary Tool
@tool
def conversation_summary(conv: str) -> str:
    """
    This tool summarizes the conversation.

    Parameters:
    - conv (str): The conversation context.

    Returns:
    - String containing the following keys:
      - intent: Identification of the customer's intent behind the conversation (e.g., complaint, inquiry, feedback).
      - conversation_summary: Summary of the conversation in 150 words.
    """
    prompt = """
    System: Summarize the conversation.
    Provide the following details in a String:
    - intent: Identification of the customer's intent behind the conversation (e.g., complaint, inquiry, feedback).
    - conversation_summary: Summary of the conversation in 150 words.
    """
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=conv)])
    # print("Conv Summary:",response.content)

    return str(response.content)
