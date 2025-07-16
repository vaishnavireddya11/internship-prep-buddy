import PyPDF2
from groq import Groq  # use openai if you're using OpenAI instead
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
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content
