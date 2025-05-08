import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

# -- Page config --
st.set_page_config(page_title="Customer Feedback Dashboard", layout="wide")

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # If you have a timestamp column, parse dates:
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def download_df(df):
    towrite = BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    return towrite

def main():
    st.title("📊 Customer Feedback Dashboard")

    # — Sidebar: file upload & filters —
    st.sidebar.header("1. Upload & Filters")
    uploaded = st.sidebar.file_uploader("Upload feedback CSV", type=["csv"])
    if not uploaded:
        st.sidebar.info("Please upload a CSV file to continue.")
        return

    df = load_data(uploaded)

    # Example date filter (uncomment if timestamp exists)
    # min_date, max_date = df['timestamp'].min(), df['timestamp'].max()
    # date_range = st.sidebar.date_input("Date range", [min_date, max_date])
    # df = df[(df['timestamp'] >= date_range[0]) & (df['timestamp'] <= date_range[1])]

    # Multiselect filters
    issue_areas = st.sidebar.multiselect("Issue Areas", df["issue_area"].unique(), default=df["issue_area"].unique())
    product_cats = st.sidebar.multiselect("Product Categories", df["product_category"].unique(), default=df["product_category"].unique())
    sat_levels = st.sidebar.multiselect("Satisfaction Level", df["satisfaction_level"].unique(), default=df["satisfaction_level"].unique())

    # Apply filters
    filt = (
        df["issue_area"].isin(issue_areas)
        & df["product_category"].isin(product_cats)
        & df["satisfaction_level"].isin(sat_levels)
    )
    filtered = df[filt]

    if filtered.empty:
        st.warning("No data matching current filters. Adjust your selections.")
        return

    # — Download filtered —
    st.sidebar.download_button(
        label="Download filtered data",
        data=download_df(filtered),
        file_name="filtered_feedback.csv",
        mime="text/csv",
    )

    # — KPIs —
    st.header("📈 Summary Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Conversations", len(filtered))
    k2.metric("Avg. Satisfaction", round(filtered["satisfaction_level"].mean(), 2))
    # Percent high-complexity
    pct_high = (filtered["issue_complexity"].str.lower() == "high").mean() * 100
    k3.metric("High Complexity (%)", f"{pct_high:.1f}%")
    # Unique customers or sessions (if you have an ID column)
    # k4.metric("Unique Customers", filtered["customer_id"].nunique())
    k4.metric("Filtered Rows", len(filtered))

    # — Top-N selector for charts —
    top_n = st.sidebar.slider("Top N", min_value=3, max_value=10, value=5)

    # — Charts layout —
    st.markdown("### 🔍 Key Insights")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader(f"Top {top_n} Issue Areas")
        top_issues = filtered["issue_area"].value_counts().nlargest(top_n).reset_index()
        top_issues.columns = ["issue_area", "count"]
        fig1 = px.bar(top_issues, x="issue_area", y="count", labels={"count": "Frequency"})
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader(f"Top {top_n} Pain Points")
        top_pains = filtered["issue_sub_category"].value_counts().nlargest(top_n).reset_index()
        top_pains.columns = ["issue_sub_category", "count"]
        fig2 = px.bar(top_pains, x="issue_sub_category", y="count", labels={"count": "Frequency"})
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📊 Distribution Insights")
    d1, d2, d3 = st.columns(3)

    with d1:
        st.subheader("Satisfaction Distribution")
        fig3 = px.pie(filtered, names="satisfaction_level", title="")
        st.plotly_chart(fig3, use_container_width=True)

    with d2:
        st.subheader("Issue Area Breakdown")
        fig4 = px.bar(filtered, x="issue_area", title="")
        st.plotly_chart(fig4, use_container_width=True)

    with d3:
        st.subheader("Customer Satisfaction Histogram")
        fig5 = px.histogram(filtered, x="customer_satisfaction", title="")
        st.plotly_chart(fig5, use_container_width=True)

    # — Optional: Sentiment word-cloud or summary text —
    if "customer_sentiment_summary" in filtered.columns:
        st.markdown("### 📝 Conversation Summaries")
        for i, row in filtered.head(3).iterrows():
            st.markdown(f"**Conversation {i+1}:** {row['conversation_summary']}")

if __name__ == "__main__":
    main()
