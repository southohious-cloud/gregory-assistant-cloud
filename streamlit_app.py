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

You must ALWAYS obey the selected processing mode.  
Only produce the output for the selected mode.  
Do NOT include any other sections unless the mode is “Everything”.

Modes:
1. SUMMARY — A concise, neutral overview of the content.
2. EXPLANATION — A plain-language breakdown of meaning and clarity.
3. KEY POINTS — A distilled list of the most important facts or ideas.
4. NEXT STEPS — Practical, reasonable actions a typical person might take.
5. EVERYTHING — Produce all four sections in this order:
   - Summary
   - Explanation
   - Key Points
   - Next Steps

Formatting rules:
- Use clear section headers.
- Keep paragraphs short and readable.
- Use bullet points for Key Points and Next Steps.
- Never invent details not present in the document.
- If the document is incomplete or unclear, state this explicitly.

Transformations (only when requested):
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
"""

# -----------------------------
# Bullet Enforcement Helper
# -----------------------------
def enforce_bullet_points(text: str) -> str:
    """
    Ensures each line in Key Points or Next Steps is formatted as a bullet.
    Removes empty lines and normalizes spacing.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    bullet_lines = []
    for line in lines:
        # Avoid double bullets
        if line.startswith("- "):
            bullet_lines.append(line)
        else:
            bullet_lines.append(f"- {line}")

    return "\n".join(bullet_lines)

# -----------------------------
# Formatting Wrapper
# -----------------------------
def format_output_with_headers(raw_output: str, mode: str) -> str:
    """
    Ensures clean, consistent section headers for all modes.
    Also enforces bullet formatting for Key Points and Next Steps.
    For non-Everything modes, it aggressively trims extra sections.
    """

    text = raw_output.strip()

    headers = {
        "Summary": ["Summary"],
        "Explanation": ["Explanation"],
        "Key Points": ["Key Points"],
        "Next Steps": ["Next Steps"],
        "Everything": ["Summary", "Explanation", "Key Points", "Next Steps"],
    }

    expected = headers.get(mode, [])

    # 🔹 Single-mode behavior: force ONLY that section
    if mode != "Everything":
        # If the model returned multiple sections, keep only the first chunk
        # before any other known header like "Explanation", "Key Points", etc.
        split_markers = ["\n### Explanation", "\n### Key Points", "\n### Next Steps"]
        cut_index = len(text)
        for marker in split_markers:
            idx = text.lower().find(marker.lower())
            if idx != -1:
                cut_index = min(cut_index, idx)

        text = text[:cut_index].strip()

        # Ensure we have a header for this mode
        if not any(h.lower() in text.lower() for h in expected):
            text = f"### {mode}\n\n{text}"

        # Enforce bullets for specific modes
        if mode in ["Key Points", "Next Steps"]:
            body = enforce_bullet_points(text)
            return body

        return text

    # 🔹 Everything mode → split into four sections
    parts = text.split("\n\n")
    cleaned = []

    for header, content in zip(expected, parts):
        content = content.strip()

        # Enforce bullets for Key Points + Next Steps
        if header in ["Key Points", "Next Steps"]:
            content = enforce_bullet_points(content)

        cleaned.append(f"### {header}\n\n{content}")

    return "\n\n".join(cleaned)
    
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
# Streamlit Page Setup
# -----------------------------
st.set_page_config(
    page_title="Gregory's Personal Assistant (Cloud Version)",
    layout="centered",
)

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

if "processing_mode" not in st.session_state:
    st.session_state.processing_mode = "Summary"


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.sidebar.title("Gregory’s Personal Assistant")
    st.sidebar.caption("Cloud Version")
    
    st.write("Status: **Online**")

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
        key="processing_mode"
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
# Groq Chat Function
# -----------------------------
def call_groq(messages):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )
    return response.choices[0].message.content


# -----------------------------
# Collapsible History Renderer
# -----------------------------
def render_collapsible_history(history):
    """
    Renders chat history using collapsible sections to prevent clutter.
    The most recent assistant message is shown fully.
    Older messages are collapsed.
    """
    if not history:
        return

    # Show the last message fully
    last_user, last_assistant = history[-1]
    if last_user:
        with st.chat_message("user"):
            st.markdown(last_user)
    if last_assistant:
        with st.chat_message("assistant"):
            st.markdown(last_assistant)

    # Render older messages collapsed
    for user_msg, assistant_msg in history[:-1]:
        with st.expander("Previous Output", expanded=False):
            if user_msg:
                st.markdown(f"**User:**\n\n{user_msg}")
            if assistant_msg:
                st.markdown(f"**Assistant:**\n\n{assistant_msg}")
# -----------------------------
# FILE UPLOAD + AUTO-REPROCESS
# -----------------------------

# Detect mode change
if "last_mode" not in st.session_state:
    st.session_state.last_mode = processing_mode

mode_changed = processing_mode != st.session_state.last_mode
st.session_state.last_mode = processing_mode

uploaded_file = st.file_uploader(
    "Upload a file for instant processing (PDF, TXT, PNG, JPG)",
    type=["pdf", "txt", "png", "jpg", "jpeg"],
    label_visibility="visible"
)

# Trigger processing if:
# 1. A new file is uploaded
# 2. OR the mode changed AND we have a previous document
if uploaded_file is not None or (mode_changed and st.session_state.last_document_text):

    # -----------------------------
    # CASE 1 — New file uploaded
    # -----------------------------
    if uploaded_file is not None:
        file_notice = f"📄 File received: **{uploaded_file.name}**"
        st.session_state.display_history.append(("", file_notice))

        file_type = uploaded_file.type

        # Extract text normally
        if file_type == "text/plain":
            extracted_text = uploaded_file.read().decode("utf-8", errors="ignore")

        elif file_type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                extracted_text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )

        elif file_type.startswith("image/"):
            extracted_text = "(Image text extraction is not available in this cloud version.)"

        # Save for auto-reprocessing
        st.session_state.last_document_text = extracted_text
        st.session_state.last_document_name = uploaded_file.name

    # -----------------------------
    # CASE 2 — Mode changed, no new file
    # -----------------------------
    else:
        extracted_text = st.session_state.last_document_text
        
    # ⭐ NEW: Mode Instruction
    MODE_INSTRUCTIONS = {
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
    }

    mode_instruction = MODE_INSTRUCTIONS[st.session_state.processing_mode]

    # ⭐ Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": mode_instruction},
        {"role": "user", "content": extracted_text}
    ]

    # ⭐ Groq call
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )

    raw_output = response.choices[0].message.content

    # ⭐ Clean, consistent section headers
    output = format_output_with_headers(raw_output, st.session_state.processing_mode)

    # Store document context
    st.session_state.last_document_text = extracted_text
    st.session_state.last_document_name = uploaded_file.name
    st.session_state.last_document_summary = output

    # Display output
    doc_header = f"### {st.session_state.processing_mode}: {uploaded_file.name}"
    st.session_state.display_history.append(("", doc_header))
    st.session_state.display_history.append(("", output))
    
# ⭐ NEW: Post-output transformation buttons
action = add_transformation_buttons()

if action:
    transform_instruction = (
        f"Transform the previous output using this instruction: {action}. "
        f"Do not add new information."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": transform_instruction},
        {"role": "user", "content": st.session_state.last_document_summary},
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )

    transformed = response.choices[0].message.content

    # Display transformed output
    st.session_state.display_history.append(("", f"### {action}"))
    st.session_state.display_history.append(("", transformed))
    st.session_state.messages.append({"role": "assistant", "content": transformed})

    st.rerun()

# -----------------------------
# Display Chat History (Collapsible)
# -----------------------------
render_collapsible_history(st.session_state.display_history)

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
