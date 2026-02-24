import os
import streamlit as st
from groq import Groq

# -----------------------------
# Environment + Client
# -----------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# System Prompt
# -----------------------------
SYSTEM_PROMPT = """
You are Gregory’s personal assistant.
Your job is to provide clear, structured, next-step guidance with zero fluff.
Always:
- Be concise and confident.
- Give actionable steps.
- Maintain a professional, binder-ready tone.
- Avoid filler language.
"""

# -----------------------------
# Groq Chat Function
# -----------------------------
def call_groq(messages):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )
    return response.choices[0].message.content


# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(
    page_title="Gregory's Personal Assistant (Cloud Version)",
    layout="centered",
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Gregory’s Assistant")
    st.write("Cloud Version")
    st.write("Status: **Online**")

    if st.button("Reset Conversation"):
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Online and ready. What would you like to do?"}
        ]
        st.session_state.display_history = [
            ("", "Online and ready. What would you like to do?")
        ]
        st.experimental_rerun()

# -----------------------------
# Title + Greeting
# -----------------------------
st.title("Gregory's Personal Assistant (Cloud Version)")
st.caption("Powered by Groq • Clean • Fast • Predictable")

# -----------------------------
# Session State Initialization
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Online and ready. What would you like to do?"}
    ]

if "display_history" not in st.session_state:
    st.session_state.display_history = [
        ("", "Online and ready. What would you like to do?")
    ]

# -----------------------------
# Display Chat History
# -----------------------------
for user_msg, assistant_msg in st.session_state.display_history:
    if user_msg:
        with st.chat_message("user"):
            st.markdown(user_msg)
    if assistant_msg:
        with st.chat_message("assistant"):
            st.markdown(assistant_msg)

# -----------------------------
# Chat Input
# -----------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare messages for Groq
    messages = st.session_state.messages.copy()
    messages.append({"role": "user", "content": user_input})

    # Get assistant reply
    assistant_reply = call_groq(messages)

    # Update session state
    st.session_state.messages = messages + [
        {"role": "assistant", "content": assistant_reply}
    ]
    st.session_state.display_history.append((user_input, assistant_reply))

    # Display assistant reply
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Gregory’s Personal Assistant • Cloud Version")
