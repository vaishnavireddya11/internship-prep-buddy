import PyPDF2
from groq import Groq  
import streamlit as st

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def ask_about_pdf(text, question):
    prompt = f"""You are an AI assistant. A user has uploaded a PDF document. Use the content below to answer the question.

PDF Content:
{text[:3000]}  # Only the first 3000 characters are used

Question:
{question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # ✅ Correct model name
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on a PDF."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content
