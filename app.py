import streamlit as st
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# ------------------ Groq Setup ------------------
# Load Groq API key from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# ------------------ Embedding Setup ------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ Functions ------------------
# Read PDF
def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# Chunk text into smaller pieces
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Build FAISS index
def build_faiss(chunks):
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

# Retrieve top-k relevant chunks
def retrieve(query, chunks, index, k=3):
    query_emb = embedder.encode([query])
    distances, indices = index.search(np.array(query_emb), k)
    return [chunks[i] for i in indices[0]]

# Ask Groq LLaMA with context
def ask_llm(query, context):
    try:
        prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="PDF Q&A with Groq LLaMA", layout="wide")
st.title("📄 PDF Q&A Chatbot (FAISS + Sentence Transformers + Groq LLaMA)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading and indexing PDF..."):
        text = read_pdf(uploaded_file)
        chunks = chunk_text(text)
        index, embeddings = build_faiss(chunks)
        st.success("PDF processed and indexed ✅")

    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Retrieving answer..."):
            top_chunks = retrieve(query, chunks, index)
            context = " ".join(top_chunks)
            answer = ask_llm(query, context)
            st.write("### Answer:")
            st.write(answer)
