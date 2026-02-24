import os
import streamlit as st
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = "You are Gregory's personal assistant. Provide clear, concise next-step guidance."


def call_groq(messages):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )
    return response.choices[0].message.content


st.set_page_config(
    page_title="Gregory's Personal Assistant (Cloud Version)",
    layout="centered",
)

st.title("Gregory's Personal Assistant (Cloud Version)")
st.caption("Powered by Groq • Clean • Fast • Predictable")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

if "display_history" not in st.session_state:
    st.session_state.display_history = []

for user_msg, assistant_msg in st.session_state.display_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(assistant_msg)

user_input = st.chat_input("Type your message...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    messages = st.session_state.messages.copy()
    messages.append({"role": "user", "content": user_input})

    assistant_reply = call_groq(messages)

    st.session_state.messages = messages + [
        {"role": "assistant", "content": assistant_reply}
    ]
    st.session_state.display_history.append((user_input, assistant_reply))

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
