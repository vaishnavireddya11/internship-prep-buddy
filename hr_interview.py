import requests
from prompts import HR_PROMPT_TEMPLATE
import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]  # 🔑 paste your real key
GROQ_MODEL = "llama3-8b-8192"  # Or use llama3-70b-8192 for higher quality

def get_hr_feedback(user_answer: str) -> str:
    prompt = HR_PROMPT_TEMPLATE.format(answer=user_answer)

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful AI HR expert."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Failed to connect to Groq API: {e}"


