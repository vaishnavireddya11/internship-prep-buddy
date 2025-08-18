import streamlit as st
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# Load secrets
groq.api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Function: Read PDF
def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# Function: Chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Function: Build FAISS index
def build_faiss(chunks):
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

# Function: Retrieve top-k chunks
def retrieve(query, chunks, index, k=3):
    query_emb = embedder.encode([query])
    distances, indices = index.search(np.array(query_emb), k)
    return [chunks[i] for i in indices[0]]

# Function: Ask OpenAI with context
def ask_llm(query, context):
    prompt = f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2
    )
    return response["choices"][0]["message"]["content"]

# Streamlit App
st.set_page_config(page_title="PDF Q&A with Embeddings", layout="wide")
st.title("📄 PDF Q&A with FAISS + Sentence Transformers")

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
