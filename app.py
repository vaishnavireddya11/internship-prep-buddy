import streamlit as st
from pdf_qa import extract_text_from_pdf, ask_about_pdf

st.set_page_config(page_title="QueryCrack - PDF Q&A", layout="wide")

st.title("📄 QueryCrack - PDF Q&A (Groq)")
st.write("Upload a PDF and ask questions based on its content.")


# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF uploaded and processed!")

    # User question
    question = st.text_input("Ask a question about the document:")

    if st.button("Get Answer"):
        if question.strip():
            with st.spinner("Thinking..."):
                answer = ask_about_pdf(pdf_text, question)
                st.markdown(f"**Answer:** {answer}")
        else:
            st.warning("⚠️ Please enter a question.")
