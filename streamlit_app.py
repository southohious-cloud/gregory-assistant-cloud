import os
import streamlit as st
from groq import Groq
from PIL import Image
import pdfplumber

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
# Groq Summarization Function
# -----------------------------
def summarize_text_with_groq(text: str) -> str:
    if not text or not text.strip():
        return "I couldn't extract readable text from this file."

    messages = [
        {"role": "system", "content": "Summarize the following text concisely and clearly."},
        {"role": "user", "content": text}
    ]

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
        st.rerun()

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
# FILE UPLOAD (INSIDE CHAT AREA)
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a file for instant summary (PDF, TXT, PNG, JPG)",
    type=["pdf", "txt", "png", "jpg", "jpeg"],
    label_visibility="visible"
)

if uploaded_file is not None:
    # Show file receipt (ONE TIME ONLY)
    file_notice = f"📄 File received: **{uploaded_file.name}**"
    st.session_state.display_history.append(("", file_notice))

    file_type = uploaded_file.type
    extracted_text = ""

    # --- TXT ---
    if file_type == "text/plain":
        extracted_text = uploaded_file.read().decode("utf-8", errors="ignore")

    # --- PDF ---
    elif file_type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            extracted_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )

    # --- IMAGES (OCR removed for cloud stability) ---
    elif file_type.startswith("image/"):
        extracted_text = "(Image text extraction is not available in this cloud version.)"

    # Summarize with Groq
    summary = summarize_text_with_groq(extracted_text)

    # Add summary to history (NO immediate display → prevents duplication)
    st.session_state.messages.append({"role": "assistant", "content": summary})
    st.session_state.display_history.append(("", summary))


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
