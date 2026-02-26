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

### DOCUMENT INTERPRETATION LAYER
When the user uploads a file or provides text, you must interpret the content with clarity, accuracy, and zero hallucination. Always stay grounded in the provided material.

You support four primary modes:
1. SUMMARY — A concise, neutral overview of the content.
2. EXPLANATION — A plain-language breakdown of what the content means, why it matters, and how to understand it without jargon.
3. KEY POINTS — A distilled list of the most important facts, decisions, or ideas.
4. NEXT STEPS — Practical, reasonable actions a typical person might take based on the content. Do not give medical, legal, financial, or safety-critical advice. Keep suggestions general and informational.

When the user selects “Everything,” produce all four sections in this order:
- Summary
- Explanation
- Key Points
- Next Steps

Formatting rules:
- Use clear section headers.
- Keep paragraphs short and readable.
- Use bullet points for Key Points and Next Steps.
- Never invent details not present in the document.
- If the document is incomplete, unclear, or missing context, state this explicitly.

After producing the main output, allow optional transformations ONLY when the user asks for them:
- Rewrite simpler
- Rewrite more formal
- Rewrite as an email
- Rewrite for a child
- Rewrite as a checklist
- Rewrite as a step-by-step plan

When performing a transformation:
- Do not add new information.
- Preserve the meaning of the original content.
- Keep the tone aligned with the requested style.

If the user asks for something outside these modes, follow normal assistant behavior while staying grounded in the document.
"""

# -----------------------------
# Formatting Wrapper
# -----------------------------
def format_output_with_headers(raw_output: str, mode: str) -> str:
    """
    Ensures clean, consistent section headers for all modes.
    Also enforces bullet formatting for Key Points and Next Steps.
    """

    text = raw_output.strip()

    headers = {
        "Summary": ["Summary"],
        "Explanation": ["Explanation"],
        "Key Points": ["Key Points"],
        "Next Steps": ["Next Steps"],
        "Everything": ["Summary", "Explanation", "Key Points", "Next Steps"]
    }

    expected = headers.get(mode, [])

    # If model already includes headers, return as-is
    if any(h.lower() in text.lower() for h in expected):
        return text

    # Single-mode formatting
    if mode != "Everything":
        formatted = f"### {mode}\n\n{text}"

        # Enforce bullets for specific modes
        if mode in ["Key Points", "Next Steps"]:
            body = enforce_bullet_points(text)
            formatted = f"### {mode}\n\n{body}"

        return formatted

    # Everything Mode → split into four sections
    parts = text.split("\n\n")
    cleaned = []

    for header, content in zip(expected, parts):
        content = content.strip()

        # Enforce bullets for Key Points + Next Steps
        if header in ["Key Points", "Next Steps"]:
            content = enforce_bullet_points(content)

        cleaned.append(f"### {header}\n\n{content}")

    return "\n\n".join(cleaned)

def normalize_spacing(text: str) -> str:
    """
    Cleans up spacing:
    - Ensures exactly one blank line between sections
    - Removes double or triple blank lines
    - Removes leading/trailing whitespace
    """

    # Split into lines and strip whitespace
    lines = [line.rstrip() for line in text.split("\n")]

    cleaned = []
    blank = False

    for line in lines:
        if line.strip() == "":
            # Only allow ONE blank line in a row
            if not blank:
                cleaned.append("")
            blank = True
        else:
            cleaned.append(line)
            blank = False

    # Remove leading/trailing blank lines
    while cleaned and cleaned[0] == "":
        cleaned.pop(0)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    return "\n".join(cleaned)

def add_transformation_buttons():
    """
    Renders post-output transformation buttons and returns the selected action.
    Returns None if no button was pressed.
    """
    st.markdown("### Additional Options")

    col1, col2 = st.columns(2)

    with col1:
        simpler = st.button("Rewrite simpler")
        formal = st.button("Rewrite more formal")
        email = st.button("Rewrite as email")

    with col2:
        checklist = st.button("Rewrite as checklist")
        steps = st.button("Rewrite as step-by-step plan")
        deeper = st.button("Explain deeper")

    if simpler: return "Rewrite simpler"
    if formal: return "Rewrite more formal"
    if email: return "Rewrite as email"
    if checklist: return "Rewrite as checklist"
    if steps: return "Rewrite as step-by-step plan"
    if deeper: return "Explain deeper"

    return None

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
    st.sidebar.title("Gregory’s Personal Assistant")
    st.sidebar.caption("Cloud Version")
    
    st.write("Status: **Online**")

    # ⭐ NEW: Mode Selector
    st.markdown("### Document Processing Mode")
    processing_mode = st.radio(
        "Choose how I should process uploaded documents:",
        [
            "Summary",
            "Explanation",
            "Key Points",
            "Next Steps",
            "Everything"
        ],
        index=0
    )

    if st.button("Reset Conversation"):
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Online and ready. What would you like to do?"}
        ]
        st.session_state.display_history = [
            ("", "Online and ready. What would you like to do?")
        ]
        st.session_state.last_document_text = None
        st.session_state.last_document_name = None
        st.session_state.last_document_summary = None
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

if "last_document_text" not in st.session_state:
    st.session_state.last_document_text = None

if "last_document_name" not in st.session_state:
    st.session_state.last_document_name = None

if "last_document_summary" not in st.session_state:
    st.session_state.last_document_summary = None

# -----------------------------
# FILE UPLOAD (INSIDE CHAT AREA)
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a file for instant processing (PDF, TXT, PNG, JPG)",
    type=["pdf", "txt", "png", "jpg", "jpeg"],
    label_visibility="visible"
)

if uploaded_file is not None:
    # File notice
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

    # --- IMAGES (no OCR in cloud version) ---
    elif file_type.startswith("image/"):
        extracted_text = "(Image text extraction is not available in this cloud version.)"

    # ⭐ NEW: Mode Instruction
    mode_instruction = {
        "Summary": "Provide a concise, neutral summary of the document.",
        "Explanation": "Explain the document in plain language, focusing on meaning and clarity.",
        "Key Points": "Extract the most important key points from the document.",
        "Next Steps": "Suggest reasonable next steps based on the document, without giving medical, legal, or financial advice.",
        "Everything": (
            "Provide all four sections in this order:\n"
            "1. Summary\n"
            "2. Explanation\n"
            "3. Key Points\n"
            "4. Next Steps"
        )
    }[processing_mode]

    # ⭐ NEW: Groq call using selected mode
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": mode_instruction},
        {"role": "user", "content": extracted_text}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )

    raw_output = response.choices[0].message.content

    # ⭐ NEW: Clean, consistent section headers
    output = format_output_with_headers(raw_output, processing_mode)

    # Store document context
    st.session_state.last_document_text = extracted_text
    st.session_state.last_document_name = uploaded_file.name
    st.session_state.last_document_summary = output

    # Display output
    doc_header = f"### {processing_mode}: {uploaded_file.name}"
    st.session_state.display_history.append(("", doc_header))
    st.session_state.display_history.append(("", output))
    st.session_state.messages.append({"role": "assistant", "content": output})

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
    with st.chat_message("user"):
        st.markdown(user_input)

    messages = st.session_state.messages.copy()
    messages.append({"role": "user", "content": user_input})

    # Add document context if available
    if st.session_state.last_document_text and st.session_state.last_document_name:
        doc_context = (
            f"The user has uploaded a document named '{st.session_state.last_document_name}'. "
            f"Here is the document content (possibly truncated):\n\n"
            f"{st.session_state.last_document_text[:6000]}"
        )
        messages.insert(
            1,
            {
                "role": "system",
                "content": (
                    "You have access to the following document context. Use it to answer questions, "
                    "extract tasks, create checklists, or explain sections when relevant.\n\n"
                    + doc_context
                ),
            },
        )

    assistant_reply = call_groq(messages)

    st.session_state.messages = messages + [
        {"role": "assistant", "content": assistant_reply}
    ]
    st.session_state.display_history.append((user_input, assistant_reply))

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Gregory's Personal Assistant Cloud Version")
