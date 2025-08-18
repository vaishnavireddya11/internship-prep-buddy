import streamlit as st
from PyPDF2 import PdfReader
from groq import Groq
import os

# Load Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Function to query LLM
def ask_llm(query, context):
    try:
       response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # Updated to supported model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ],
    temperature=0.2
)

        return response.choices[0].message["content"]
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="PDF Q&A with Groq", layout="wide")

st.title("📄 PDF Q&A Chatbot (Groq-powered)")
st.write("Upload a PDF, then ask questions about it!")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    pdf_text = extract_pdf_text(uploaded_file)

    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Thinking..."):
            answer = ask_llm(query, pdf_text[:4000])  # limit context size
            st.subheader("Answer:")
            st.write(answer)
