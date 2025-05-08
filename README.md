# 🤖 ARIA: Augmented Retrieval & Insight Agent

Welcome to the **ARIA: Augmented Retrieval & Insight Agent** – an AI-powered analytics platform that transforms raw customer support conversations into actionable business insights, interactive dashboards, and instant solutions.

---

## 🚀 Features

- **Automated Insight Extraction:**  
  Upload chat logs or CSVs and instantly extract structured insights like issue area, pain points, satisfaction, sentiment, and more using advanced LLMs.

- **Semantic RAG Search:**  
  Retrieve and rephrase relevant solutions to new customer queries using FAISS vector search and LLM-powered rephrasing.

- **Interactive Dashboards:**  
  Explore trends, filter by issue area, satisfaction, product category, and visualize key metrics with Streamlit and Plotly.

- **Flexible Data Input:**  
  Supports both plain text conversations and pre-structured CSVs for maximum flexibility.

- **Production-Ready Stack:**  
  Built with Python, Streamlit, Plotly, FAISS, Sentence Transformers, and Groq LLMs.

---

## 🏗️ Project Structure

```
.
├── main.py                # Main Streamlit app for insight extraction & dashboard
├── Dashboard.py           # Standalone dashboard for CSV analytics
├── Conv_insights.py       # Conversation insights extraction logic
├── Insight_tools.py       # Modular tools for LLM-based insight extraction
├── Json_To_Csv.py         # Utility to convert JSON insights to CSV
├── rag.py                 # RAG (Retrieval-Augmented Generation) semantic search
├── output.csv             # Default insights output file
├── faiss_index_test.bin   # FAISS vector index for semantic search
├── Conversation/          # Folder with sample conversation .txt files
├── Row_datasets/          # Additional datasets
├── .env                   # Environment variables (API keys, model paths, etc.)
└── ...
```

---

## ⚡ Quick Start

1. **Clone the repo & install dependencies:**
    ```sh
    git clone https://github.com/Techofast-Official/ARIA
    pip install -r requirements.txt
    ```

2. **Set up your `.env` file:**
    ```
    GROQ_API_KEY=your_groq_api_key
    EMBED_MODEL_PATH=all-MiniLM-L6-v2
    DATA_CSV_PATH=output.csv
    FAISS_INDEX_PATH=faiss_index_test.bin
    ```

3. **Run the main dashboard:**
    ```sh
    streamlit run main.py
    ```

4. **(Optional) Use the standalone dashboard:**
    ```sh
    streamlit run Dashboard.py
    ```

---

## 🧠 How It Works

- **Insight Extraction:**  
  Uses LLM prompts (via Groq) to parse conversations and extract structured fields (issue area, pain points, sentiment, etc.).

- **Semantic Search (RAG):**  
  Embeds queries and historical issues using Sentence Transformers, retrieves similar cases with FAISS, and uses LLMs to judge and rephrase solutions.

- **Visualization:**  
  Interactive dashboards built with Streamlit and Plotly for real-time analytics and filtering.

---

## 📊 Example Use Cases

- Customer support analytics & pain point discovery
- Automated satisfaction and sentiment tracking
- Instant solution retrieval for new queries
- Business intelligence for support teams

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** (UI & dashboards)
- **Plotly** (visualizations)
- **FAISS** (vector search)
- **Sentence Transformers** (embeddings)
- **Groq LLMs** (insight extraction & rephrasing)
- **Pandas** (data wrangling)

---

## 📂 Sample Data

- Place your `.txt` conversation files in the `Conversation/` folder.
- Upload CSVs with insights for dashboard analytics.

---

▶️YouTube🔴

[![Watch the demo](https://img.youtube.com/vi/lIzwvYZBkVM/0.jpg)](https://youtu.be/lIzwvYZBkVM)


## 🤝 Contributing

Pull requests and suggestions are welcome!  
Feel free to open an issue or reach out for collaboration.

---

## 📄 License

MIT License

---

> Built with ❤️ by Mohd Ahmad Ansari
