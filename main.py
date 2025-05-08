import os
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import re

import rag  # <-- Imported your rag module

# Configuration
load_dotenv()
API_KEY = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

# RAG Setup
EMBED_MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", "all-MiniLM-L6-v2")
emb_model = SentenceTransformer(EMBED_MODEL_PATH)
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index_test.bin")
index = faiss.read_index(FAISS_INDEX_PATH)

# Load conversation insights CSV for retrieval
DATA_CSV = os.environ.get("DATA_CSV_PATH", "output.csv")
df = pd.read_csv(DATA_CSV)
COLS_MERGE = ["issue_category", "issue_sub_category", "issue_complexity", "product_category", "product_sub_category"]
if "combined_text" not in df.columns:
    df["combined_text"] = df.apply(lambda row: "\n".join(f"{col}: {row[col]}" for col in COLS_MERGE), axis=1)
stored_embeddings = emb_model.encode(df["combined_text"].tolist())

# RAG Functions
def retrieve_similar(conv: str):
    q_emb = emb_model.encode(conv)
    D, I = index.search(np.array([q_emb]), 1)
    idx = int(I[0][0])
    return df.iloc[idx]["combined_text"], idx

def extract_json(text: str) -> str:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def generate_insights(conv_text: str, required_insights: list) -> dict:
    similar_text, sim_idx = retrieve_similar(conv_text)
    similar_row = df.iloc[sim_idx]
    similar_insights = {k: str(similar_row[k]) for k in required_insights if k in similar_row}
    similar_insights_str = json.dumps(similar_insights, indent=2)

    prompt = f"""
Analyze the following conversation and extract the structured insights as specified below. Your response must be a valid JSON object with the exact keys listed, and nothing else (no additional text or explanations).

Conversation: '''{conv_text}'''

Extract the following insights:
- issue_area: One of [Order, Login and Account, Shopping, Cancellations and returns, Warranty, Shipping]
- issue_category: The category of the issue (e.g., Product recall, Delayed delivery).
- issue_sub_category: The sub-category of the issue (e.g., Return process, Product not received).
- issue_complexity: One of [medium, high, small]
- product_category: One of [Electronics, Men/Women/Kids, Appliances]
- product_sub_category: Specific product type (e.g., Computer Monitor, Headphone, Wet Grinder).
- customer_pain_points: Specific issues or challenges raised by the customer.
- solutions_proposed: Solutions or suggestions provided during the conversation.
- followup_required: Any follow-up actions needed (Yes/No).
- action_items: Tasks or actions decided upon during the conversation.
- customer_satisfaction: Whether the customer is satisfied or not (Yes/No).
- satisfaction_level: An integer between 0 to 5.
- customer_sentiment_summary: A brief summary of the customer's overall sentiment.
- representative_tone: Representative's tone during the conversation (e.g., Helpful, Apologetic).
- intent: Customer's intent (e.g., complaint, inquiry, feedback).
- conversation_summary: Summary of the conversation in approximately 150 words.

Also, consider the insights from a similar conversation for reference:
Similar Insights: '''{similar_insights_str}'''

**Important**: Return only the JSON object with the specified keys. Do not include any additional text, such as "Here is the extracted JSON object:" or any explanations. Ensure the response is valid JSON.

Example of expected output:
{{
  "issue_area": "Order",
  "issue_category": "Delayed delivery",
  "issue_sub_category": "Product not received",
  "issue_complexity": "medium",
  "product_category": "Electronics",
  "product_sub_category": "Headphone",
  "customer_pain_points": "Customer has not received the product even after the expected delivery date.",
  "solutions_proposed": "Representative suggested to check the tracking status of the order and assured to escalate the issue to the concerned team.",
  "followup_required": "Yes",
  "action_items": "Representative to escalate the issue to the logistic team and follow-up with the customer with an update.",
  "customer_satisfaction": "No",
  "satisfaction_level": 3,
  "customer_sentiment_summary": "The customer is upset and frustrated due to the delayed delivery.",
  "representative_tone": "Empathetic and understanding",
  "intent": "Complaint",
  "conversation_summary": "The customer contacted regarding a delayed delivery of a headset ordered. The product has not been received even after the expected delivery date. The representative suggested the customer to check the tracking status and assured that they will escalate the issue to the logistic team. The customer is upset and frustrated due to the delayed delivery."
}}
"""
    response = llm.invoke([SystemMessage(content=prompt)])
    raw_response = response.content.strip()

    json_str = extract_json(raw_response)

    try:
        structured_data = json.loads(json_str)
        for key in required_insights:
            if key not in structured_data:
                structured_data[key] = "N/A"
        if "satisfaction_level" in structured_data:
            try:
                structured_data["satisfaction_level"] = int(float(structured_data["satisfaction_level"]))
            except (ValueError, TypeError):
                structured_data["satisfaction_level"] = None
        return structured_data
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse JSON: {e}",
            "raw_response": raw_response,
            "conversation": conv_text
        }

def main():
    st.set_page_config(page_title="Next-Gen Conversation Insight Dashboard", layout="wide")
    st.title("🤖 ARIA: Augmented Retrieval & Insight Agent")

    uploaded_files = st.file_uploader("Upload conversation TXT or CSV of insights", type=["txt", "csv"], accept_multiple_files=True)
    if not uploaded_files:
        st.info("Upload one or more .txt conversation files or a .csv insights file to proceed.")
        return

    required_insights = [
        "issue_area",
        "issue_category",
        "issue_sub_category",
        "issue_complexity",
        "product_category",
        "product_sub_category",
        "customer_pain_points",
        "solutions_proposed",
        "followup_required",
        "action_items",
        "customer_satisfaction",
        "satisfaction_level",
        "customer_sentiment_summary",
        "representative_tone",
        "intent",
        "conversation_summary"
    ]

    insights_df = None
    if any(f.name.endswith(".txt") for f in uploaded_files):
        st.markdown("### Processing Conversations")
        insights_list = []
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith(".txt"):
                conv_text = uploaded_file.read().decode("utf-8")
                st.markdown(f"#### Conversation: {uploaded_file.name}")
                st.write(conv_text)

                with st.spinner(f"Generating insights for {uploaded_file.name}…"):
                    structured_data = generate_insights(conv_text, required_insights)

                if "error" in structured_data:
                    st.error(f"❌ Failed to process {uploaded_file.name}: {structured_data['error']}")
                    st.markdown("**Raw Response:**")
                    st.code(structured_data["raw_response"])
                    continue

                st.markdown(f"**Generated Insights for {uploaded_file.name}:**")
                st.json(structured_data)

                insights_list.append({
                    "conversation": conv_text,
                    **structured_data
                })

        if insights_list:
            insights_df = pd.DataFrame(insights_list)
            st.markdown("### All Generated Insights")
            st.dataframe(insights_df)
    else:
        if len(uploaded_files) > 1:
            st.warning("Only one CSV file can be processed at a time. Using the first uploaded CSV.")
        uploaded_file = uploaded_files[0]
        insights_df = pd.read_csv(uploaded_file)
        st.markdown("### Uploaded Insights")
        st.dataframe(insights_df)

    if insights_df is None or insights_df.empty:
        st.error("No valid data to display. Please check the uploaded files.")
        return

    st.sidebar.header("Filters")
    if "issue_area" in insights_df:
        areas = st.sidebar.multiselect("Issue Areas", insights_df["issue_area"].unique(), default=None)
        if areas:
            insights_df = insights_df[insights_df["issue_area"].isin(areas)]
    top_n = st.sidebar.slider("Top N", 3, 10, 5)

    # 💬 New Query Input Section
    st.header("💬 Check for Solution to a New Query")
    user_query = st.text_input("Enter your query here (e.g., a customer's issue or message):")
    if st.button("Submit Query"):
        with st.spinner("Searching for similar issue..."):
            try:
                retrieved_index, solution = rag.rag(user_query)
                if solution.strip():
                    st.success("✅ Solution Found:")
                    st.markdown(solution)
                else:
                    st.warning("❗ We will let you know the process soon.")
            except Exception as e:
                st.error(f"An error occurred during solution retrieval: {e}")

    st.header("📊 Insights Visualizations")
    if "satisfaction_level" in insights_df:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Conversations", len(insights_df))
        if not insights_df["satisfaction_level"].isna().all():
            col2.metric("Average Satisfaction", round(insights_df["satisfaction_level"].mean(), 2))
        if "issue_complexity" in insights_df:
            pct_high = (insights_df["issue_complexity"].str.lower() == "high").mean() * 100
            col3.metric("High Complexity %", f"{pct_high:.1f}%")

    if len(insights_df) > 1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"Top {top_n} Issue Areas")
            if "issue_area" in insights_df:
                issue_area_df = insights_df["issue_area"].value_counts().nlargest(top_n).reset_index()
                issue_area_df.columns = ["issue_area", "count"]
                fig1 = px.bar(issue_area_df, x="issue_area", y="count", labels={"count": "Frequency"})
                st.plotly_chart(fig1, use_container_width=True)
        with c2:
            st.subheader(f"Top {top_n} Pain Points")
            if "issue_sub_category" in insights_df:
                pain_point_df = insights_df["issue_sub_category"].value_counts().nlargest(top_n).reset_index()
                pain_point_df.columns = ["issue_sub_category", "count"]
                fig2 = px.bar(pain_point_df, x="issue_sub_category", y="count", labels={"count": "Frequency"})
                st.plotly_chart(fig2, use_container_width=True)

        if "satisfaction_level" in insights_df:
            with st.expander("Distribution Charts"):
                st.plotly_chart(px.pie(insights_df, names="satisfaction_level", title="Satisfaction Distribution"), use_container_width=True)
        if "issue_complexity" in insights_df:
            with st.expander("Complexity Distribution"):
                st.plotly_chart(px.pie(insights_df, names="issue_complexity", title="Complexity Distribution"), use_container_width=True)
    else:
        st.info("Visualizations are limited for a single conversation. Here are the insights:")
        st.dataframe(insights_df)

    csv_buf = insights_df.to_csv(index=False).encode()
    st.download_button("Download Insights Data", data=csv_buf, file_name="insights.csv", mime="text/csv")

if __name__ == "__main__":
    main()
