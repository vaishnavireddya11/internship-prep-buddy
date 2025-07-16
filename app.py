import streamlit as st
import random
from hr_interview import get_hr_feedback
from prompts import HR_QUESTIONS
import pdf_qa

st.set_page_config(page_title="QueryCrack", page_icon="🧠")

st.title("🧠 QueryCrack")

# --- HR Interview Bot Section ---
st.header("🧑‍💼 HR Interview Practice")

# Initialize session state
if "question_index" not in st.session_state:
    st.session_state.question_index = random.randint(0, len(HR_QUESTIONS) - 1)
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "feedback" not in st.session_state:
    st.session_state.feedback = ""
if "user_input_value" not in st.session_state:
    st.session_state.user_input_value = ""

# Display current question
question = HR_QUESTIONS[st.session_state.question_index]
st.markdown(f"**🗣️ HR Question:** {question}")

# Text input
user_input = st.text_area("💬 Your Answer:", key="answer_input", height=200)

# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("📝 Get Feedback"):
        if user_input.strip():
            st.session_state.feedback = get_hr_feedback(user_input.strip())
            st.session_state.show_feedback = True
        else:
            st.warning("Please enter your answer before submitting.")

with col2:
    if st.button("➡️ Next Question"):
        st.session_state.question_index = random.randint(0, len(HR_QUESTIONS) - 1)
        st.session_state.feedback = ""
        st.session_state.show_feedback = False
        st.session_state.answer_input = ""  # Clear previous answer
        st.rerun()

if st.session_state.show_feedback:
    st.markdown("### ✅ AI Feedback")
    st.write(st.session_state.feedback)

st.markdown("---")

# --- PDF Q&A Section ---
st.header("📄 Ask Questions from a PDF")

pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

if pdf_file:
    with st.spinner("Reading PDF..."):
        pdf_text = pdf_qa.extract_text_from_pdf(pdf_file)
        st.success("PDF uploaded successfully!")

    question = st.text_input("Ask a question about the PDF:")
    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                answer = pdf_qa.ask_about_pdf(pdf_text, question)
                st.markdown(f"**Answer:** {answer}")
        else:
            st.warning("Please enter a question.")
