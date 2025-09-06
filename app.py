import os
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import json

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

# Build FAISS (cached for speed)
@st.cache_resource
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
if "study_plan" not in st.session_state:
    st.session_state.study_plan = None
if "quiz" not in st.session_state:
    st.session_state.quiz = []

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
        if st.button("Get Answer") and query:
            top_chunks = retrieve(query, chunks, index)
            context = " ".join(top_chunks)
            answer = ask_llm(f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:")
            st.session_state.qa_history.append({"q": query, "a": answer})

        # Show full history
        if st.session_state.qa_history:
            for item in reversed(st.session_state.qa_history):
                st.write(f"**Q:** {item['q']}")
                st.write(f"**A:** {item['a']}")
                st.markdown("---")

    # ------------------ Study Plan ------------------
    elif page == "Study Plan":
        st.title("📘 Personalized Study Plan")
        time_input = st.text_input("Available time (e.g., '2 hours', '3 days')")
        if st.button("Generate Plan") and time_input:
            plan = ask_llm(f"Create a detailed study plan based on this content:\n\n{text[:3000]}\n\nTime available: {time_input}\n\nStudy Plan:")
            st.session_state.study_plan = plan

        # Display saved plan
        if st.session_state.study_plan:
            st.write("### Study Plan:")
            st.write(st.session_state.study_plan)

    # ------------------ Quiz ------------------
    elif page == "Quiz":
        st.title("📝 Quiz Generator")
        num_qs = st.number_input("Number of questions", min_value=1, max_value=10, value=3, step=1)
        if st.button("Create Quiz"):
            quiz_prompt = f"""
            Generate {num_qs} multiple-choice quiz questions based ONLY on the PDF content below.
            Each question must have 4 options (A, B, C, D) and exactly one correct answer.
            Return valid JSON as:
            [
              {{"question": "...", "options": ["A) ...","B) ...","C) ...","D) ..."], "answer": "B"}}
            ]

            Content: {text[:3000]}
            """
            quiz_text = ask_llm(quiz_prompt, temperature=0.5)

            try:
                quiz = json.loads(quiz_text)
                if isinstance(quiz, list):
                    st.session_state.quiz = quiz
                else:
                    st.session_state.quiz = []
            except:
                st.session_state.quiz = []

        # Display quiz with MCQs
        if st.session_state.quiz:
            for i, q in enumerate(st.session_state.quiz):
                st.subheader(f"Q{i+1}: {q['question']}")
                choice = st.radio("Choose an option:", q["options"], key=f"q{i}")
                if st.button(f"Check Answer {i+1}", key=f"ans{i}"):
                    if choice.strip().startswith(q["answer"]):
                        st.success("✅ Correct!")
                    else:
                        st.error(f"❌ Wrong. Correct Answer: {q['answer']}")

    # ------------------ History ------------------
    elif page == "History":
        st.title("📜 Q&A History")
        if st.session_state.qa_history:
            for item in reversed(st.session_state.qa_history):
                st.write(f"**Q:** {item['q']}")
                st.write(f"**A:** {item['a']}")
                st.markdown("---")
        else:
            st.info("No history yet. Ask some questions first!")
else:
    st.warning("⚠️ Please upload a PDF to get started.")
