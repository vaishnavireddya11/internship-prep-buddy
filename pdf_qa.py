import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os
import json

# ------------------ Groq Setup ------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ------------------ PDF Functions ------------------
def extract_text_from_pdf(pdf_file):
    """Reads PDF and extracts full text."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ------------------ Text Chunking ------------------
def chunk_text(text, chunk_size=500):
    """Split text into smaller chunks (default 500 words)."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ------------------ Embedding + FAISS ------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss(chunks):
    """Build FAISS index from text chunks."""
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

# Retrieve relevant chunks
def retrieve(query, chunks, index, k=3):
    """Retrieve top-k relevant chunks for a query."""
    query_emb = embedder.encode([query])
    distances, indices = index.search(np.array(query_emb), k)
    return [chunks[i] for i in indices[0]]

# ------------------ LLM Answering ------------------
def ask_llm(prompt, temperature=0.3, model="llama-3.3-70b-versatile"):
    """Ask Groq LLM with a prompt."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"
