import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# ------------------------------
# Configuration
# ------------------------------
embedding_model_path = os.environ.get("EMBED_MODEL_PATH", "all-MiniLM-L6-v2")
output_csv_path = os.environ.get("DATA_CSV_PATH", "output.csv")
index_path = os.environ.get("FAISS_INDEX_PATH", "faiss_index_test.bin")

# ------------------------------
# Load Embedding Model and Data
# ------------------------------
emb_model = SentenceTransformer(embedding_model_path)
df = pd.read_csv(output_csv_path)

cols_merge = [
    "issue_category",
    "issue_sub_category",
    "issue_complexity",
    "product_category",
    "product_sub_category",
]

def merge_cols(row, cols):
    return "\n".join(f"{col}: {row[col]}" for col in cols)

if "combined_text" not in df.columns:
    df["combined_text"] = df.apply(lambda x: merge_cols(x, cols_merge), axis=1)

# ------------------------------
# Load FAISS Index
# ------------------------------
index = faiss.read_index(index_path)

# ------------------------------
# Load Groq LLM
# ------------------------------
llm = ChatGroq(model="llama3-8b-8192", temperature=0)

# ------------------------------
# Main Semantic RAG Function
# ------------------------------
def rag(conv: str):
    # Step 1: Embed the user query
    query_embedding = emb_model.encode(conv)

    # Step 2: Top-k retrieval from FAISS
    top_k = 5
    distances, indices = index.search(np.array([query_embedding]), top_k)

    # Step 3: Use LLM to judge similarity
    for i in range(top_k):
        candidate_idx = indices[0][i]
        retrieved_entry = df.iloc[candidate_idx]["combined_text"]
        solution = df.iloc[candidate_idx]["solutions_proposed"]

        # Semantic similarity check
        similarity_prompt = f"""You are a semantic expert.

Compare the following user query and a documented issue. Decide if they are referring to the **same issue**, even if the wording is different.

Only answer "YES" or "NO".

User Query:
{conv}

Documented Issue:
{retrieved_entry}

Answer:"""

        decision = llm.invoke(similarity_prompt).content.strip().lower()

        if "yes" in decision:
            # Rephrase solution to be customer-ready
            rephrase_prompt = f"""You are a support assistant.

A user has asked for help. You found a relevant solution. Rephrase this solution clearly and briefly (2–3 lines), giving direct instructions.

User Query:
{conv}

Original Solution:
{solution}

Rephrased Solution:"""
            rephrased = llm.invoke(rephrase_prompt).content.strip()
            return candidate_idx, rephrased

    # If none matched
    return -1, ""
