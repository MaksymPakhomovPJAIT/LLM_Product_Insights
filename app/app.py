"""
Product Insights Q&A (RAG with OpenAI + FAISS + Streamlit)

This app allows users to ask natural language questions about consumer products
(shampoos) and get answers grounded in product descriptions and customer reviews.
It demonstrates a practical application of GenAI in consumer insights and retail.
"""

import os, json, numpy as np, faiss
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# Setup
# Load API key from .env file (never hardcode keys in code!)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IDX_DIR = PROJECT_ROOT / "src/index"   # Folder with FAISS index + metadata
META_PATH = IDX_DIR / "meta.jsonl"
INDEX_PATH = IDX_DIR / "faiss.index"

# Models
EMBED_MODEL = "text-embedding-3-small"   # cheap + fast embeddings
CHAT_MODEL = "gpt-4o-mini"               # efficient chat model for Q&A

# -----------------------------
# Load index + docs (cached)
# -----------------------------
@st.cache_resource
def load_index_and_docs():
    """Load FAISS index and metadata once and cache them in Streamlit."""
    docs = [json.loads(l) for l in META_PATH.read_text(encoding="utf-8").splitlines()]
    index = faiss.read_index(str(INDEX_PATH))
    return index, docs

index, docs = load_index_and_docs()

# Prepare brand list for dropdown filter
brands = sorted({d.get("brand", "?") for d in docs if d.get("brand")})
brands = [b for b in brands if b and b != "?"]

# -----------------------------
# Retrieval helpers
# -----------------------------
def embed_query(q: str):
    """Convert a query string into an embedding vector (normalized)."""
    e = client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
    v = np.array(e, dtype="float32")
    return v / np.linalg.norm(v)

def search(query: str, k=6, allowed_brands=None, min_score=0.35, max_per_brand=3):
    """
    Retrieve top-k relevant chunks from FAISS index.
    - allowed_brands: optional filter (only return chunks from these brands)
    - min_score: minimum similarity threshold
    - max_per_brand: limit how many chunks per brand to avoid dominance
    """
    v = embed_query(query)
    # Over-retrieve (k*4) to allow filtering/diversification
    D, I = index.search(np.array([v]), k * 4)
    hits, per_brand = [], {}
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        d = docs[idx].copy()
        brand = d.get("brand", "?")
        # Apply filters
        if allowed_brands and brand not in allowed_brands:
            continue
        if float(score) < float(min_score):
            continue
        if per_brand.get(brand, 0) >= max_per_brand:
            continue
        # Save hit with score
        d["score"] = float(score)
        hits.append(d)
        per_brand[brand] = per_brand.get(brand, 0) + 1
        if len(hits) >= k:
            break
    return hits

# -----------------------------
# RAG pipeline
# -----------------------------
SYSTEM = (
    "You are a careful assistant answering questions about consumer shampoos. "
    "ONLY use the provided context; if the answer isn't covered, say you have the information needed to answer. "
    "Be concise, compare products when relevant, and include caveats (e.g., sensitive scalp). "
    "Reference evidence as [1], [2], etc."
)

def build_context(docs):
    """Format retrieved docs into a context block for the LLM prompt."""
    lines = []
    for i, d in enumerate(docs, start=1):
        brand = d.get("brand", "?")
        src = d.get("source", "?").upper()
        lines.append(f"[{i}] {src} | Brand: {brand} | score={d['score']:.3f}\n{d['content']}")
    return "\n\n".join(lines)

def answer(query: str, k=6, allowed_brands=None, min_score=0.35):
    """Retrieve supporting chunks and ask the LLM for a grounded answer."""
    ctx_docs = search(query, k=k, allowed_brands=allowed_brands, min_score=min_score)
    if not ctx_docs:
        return "I don’t know from the provided data. Try lowering the min score or broadening brands.", []
    context = build_context(ctx_docs)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nIf you cite, reference chunks by [number]."}
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=350
    )
    return resp.choices[0].message.content, ctx_docs

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Product Insights Q&A", layout="centered")
st.title("Product Insights Q&A (RAG)")
st.caption("Answers are grounded in product descriptions and real reviews. Citations included.")

# User input
query = st.text_input("Your question", placeholder="e.g., Best option for dandruff + sensitive scalp?")
col1, col2 = st.columns(2)
with col1:
    k = st.slider("Context chunks", 2, 10, 6, 1)
with col2:
    min_score = st.slider("Min score", 0.0, 0.9, 0.35, 0.05)

brand_sel = st.multiselect("Filter by brand (optional)", options=brands)

# Answer
if st.button("Ask") and query.strip():
    with st.spinner("Thinking..."):
        ans, ctx = answer(query, k=k, allowed_brands=brand_sel or None, min_score=min_score)

    st.markdown("### Answer")
    st.write(ans)

    # Show retrieved evidence
    with st.expander("Show retrieved context"):
        for i, d in enumerate(ctx, start=1):
            st.markdown(f"**[{i}] {d['source'].upper()} | {d.get('brand','?')} | score={d['score']:.3f}**")
            st.write(d["content"])