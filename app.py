import os
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# ------------------ Groq Setup ------------------
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

# Chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Build FAISS
def build_faiss(chunks):
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

# Retrieve relevant chunks
def retrieve(query, chunks, index, k=3):
    query_emb = embedder.encode([query])
    distances, indices = index.search(np.array(query_emb), k)
    return [chunks[i] for i in indices[0]]

# Ask Groq
def ask_llm(prompt, temperature=0.3, model="llama-3.3-70b-versatile"):
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

# ------------------ Streamlit Layout ------------------
st.set_page_config(page_title="Smart PDF Assistant", layout="wide")

st.sidebar.title("📂 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

# Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Q&A", "Study Plan", "Quiz", "History"])

# Initialize session state
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if uploaded_file:
    with st.spinner("Reading and indexing PDF..."):
        text = read_pdf(uploaded_file)
        chunks = chunk_text(text)
        index, embeddings = build_faiss(chunks)
        st.sidebar.success("✅ PDF processed")

    # ------------------ Q&A ------------------
    if page == "Q&A":
        st.title("📄 Q&A with PDF")
        query = st.text_input("Ask a question:")
        if query:
            top_chunks = retrieve(query, chunks, index)
            context = " ".join(top_chunks)
            answer = ask_llm(f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:")
            st.write("### Answer:")
            st.write(answer)

            # Save to history
            st.session_state.qa_history.append({"q": query, "a": answer})

    # ------------------ Study Plan ------------------
    elif page == "Study Plan":
        st.title("📘 Personalized Study Plan")
        time_input = st.text_input("Available time (e.g., '2 hours', '3 days')")
        if st.button("Generate Plan") and time_input:
            plan = ask_llm(f"Create a detailed study plan based on this content:\n\n{text[:3000]}\n\nTime available: {time_input}\n\nStudy Plan:")
            st.write("### Study Plan:")
            st.write(plan)

    # ------------------ Quiz ------------------
    elif page == "Quiz":
        st.title("📝 Quiz Generator")
        num_qs = st.number_input("Number of questions", min_value=1, max_value=20, value=5, step=1)
        if st.button("Create Quiz"):
            quiz = ask_llm(f"Generate {num_qs} quiz questions (with answers) based on this content:\n\n{text[:3000]}\n\nQuiz:")
            st.write("### Quiz:")
            st.write(quiz)

    # ------------------ History ------------------
    elif page == "History":
        st.title("📜 Q&A History")
        if st.session_state.qa_history:
            for item in st.session_state.qa_history:
                st.write(f"**Q:** {item['q']}")
                st.write(f"**A:** {item['a']}")
                st.markdown("---")
        else:
            st.info("No history yet. Ask some questions first!")
else:
    st.warning("⚠️ Please upload a PDF to get started.")
