import os
import streamlit as st
from groq import Groq
from PIL import Image
import pdfplumber

# -----------------------------
# Initialize session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "display_history" not in st.session_state:
    st.session_state.display_history = []

if "last_document_text" not in st.session_state:
    st.session_state.last_document_text = None

if "last_document_name" not in st.session_state:
    st.session_state.last_document_name = None

if "last_document_summary" not in st.session_state:
    st.session_state.last_document_summary = None

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
Follow the processing instructions provided in the next system message exactly.
Only produce the content required by that instruction.
Do not add summaries, explanations, key points, next steps, or multiple sections unless the instruction explicitly requires them.
Keep your writing clear, factual, and concise.
Never invent details that are not present in the document.
"""

# -----------------------------
# Bullet Normalizer
# -----------------------------
def enforce_bullet_points(text: str) -> str:
    """
    Normalizes all bullet styles to '- ' and ensures each line is a clean bullet.
    Removes empty lines and strips any existing bullet symbols.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    bullet_symbols = ["•", "○", "●", "*", "-", "▪", "·", "‣", "–"]

    bullet_lines = []
    for line in lines:
        cleaned = line
        for sym in bullet_symbols:
            if cleaned.startswith(sym):
                cleaned = cleaned[len(sym):].strip()
        bullet_lines.append(f"- {cleaned}")

    return "\n".join(bullet_lines)


# -----------------------------
# Formatting Wrapper (Strict Single-Mode)
# -----------------------------
def format_output_with_headers(raw_output: str, mode: str) -> str:
    """
    Ensures clean, consistent section headers for all modes.
    Strips ALL model-generated headers.
    Removes ALL extra sections in single-mode.
    Enforces bullet formatting for Key Points and Next Steps.
    """

    text = raw_output.strip()

    # All possible section headers the model might generate
    all_headers = ["Summary", "Explanation", "Key Points", "Next Steps"]

    # Expected header for the selected mode
    expected = [mode]

    # Normalize helper for header detection
    def normalize_header(line: str) -> str:
        return " ".join(line.strip().lower().replace(":", "").replace("#", "").split())

    # -----------------------------------------
    # SINGLE-MODE BEHAVIOR
    # -----------------------------------------
    if mode != "Everything":

        # Split into lines
        lines = text.split("\n")

        # Remove ALL model-generated headers AND stop when a new section begins
        cleaned = []
        for line in lines:
            norm = normalize_header(line)

            # If this line is ANY section header (Summary, Explanation, Key Points, Next Steps)
            # AND it is NOT the selected mode → STOP (cut off extra sections)
            if norm in [h.lower() for h in all_headers] and norm != mode.lower():
                break

            # If this line is the header for the selected mode → skip it
            if norm == mode.lower():
                continue

            cleaned.append(line)

        text = "\n".join(cleaned).strip()

        # Add our clean header
        text = f"### {mode}\n\n{text}"

        # Bullet enforcement for Key Points + Next Steps
        if mode in ["Key Points", "Next Steps"]:
            return enforce_bullet_points(text)

        return text

    # -----------------------------------------
    # EVERYTHING MODE
    # -----------------------------------------
    parts = text.split("\n\n")
    cleaned_sections = []

    for header, content in zip(all_headers, parts):
        content = content.strip()

        # Remove ALL model-generated header lines inside each section
        lines = []
        for line in content.split("\n"):
            norm = normalize_header(line)
            if norm in [h.lower() for h in all_headers]:
                continue
            lines.append(line)

        content = "\n".join(lines).strip()

        # Bullet enforcement for Key Points + Next Steps
        if header in ["Key Points", "Next Steps"]:
            content = enforce_bullet_points(content)

        cleaned_sections.append(f"### {header}\n\n{content}")

    return "\n\n".join(cleaned_sections)


# -----------------------------
# Post-Output Transformation Buttons
# -----------------------------
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

mode_instruction = {
    "Summary": """
You are extracting the SUMMARY section from the document.

Your task:
- Extract only the essential summary of the document.
- Do NOT include explanation, key points, or next steps.
- Keep the summary concise, factual, and directly tied to the document.
- Do NOT invent content that is not present or implied.

Output only the summary text with no header.
""",

    "Explanation": """
You are extracting the EXPLANATION section from the document.

Your task:
- Explain the document’s meaning, purpose, or intent.
- Clarify the ideas in the document using plain language.
- Do NOT include summary, key points, or next steps.
- Base the explanation strictly on the document’s content.

Output only the explanation text with no header.
""",

    "Key Points": """
You are extracting the KEY POINTS from the document.

Your task:
- Identify the most important ideas, facts, or statements.
- Convert them into a clean bullet list.
- Do NOT include summary, explanation, or next steps.
- Do NOT invent points that are not present or implied.

Output only the bullet list with no header.
""",

    "Next Steps": """
You are extracting the NEXT STEPS from the document.

Your task:
- Extract explicit next steps if they exist.
- If the document does NOT explicitly list next steps, infer them using reasonable, non‑speculative logic based on the document’s goals, problems, or implied future actions.
- Always produce a clear, actionable list of next steps written in Gregory’s workflow style: direct, professional, predictable, and free of speculation.
- Use concrete action verbs such as define, confirm, implement, review, document, and validate.
- Keep steps specific, ordered, and execution‑ready.
- Do NOT include summary, explanation, or key points.
- Do NOT invent facts, names, dates, or details not supported by the document.

Output only the bullet list with no header.
""",

    "Everything": """
You are extracting ALL FOUR SECTIONS from the document:

1. Summary
2. Explanation
3. Key Points
4. Next Steps (infer if missing)

Your task:
- Produce all four sections in that exact order.
- Each section must contain only its own content.
- Key Points and Next Steps must be bullet lists.
- Next Steps must be inferred if the document does not explicitly contain them.
- Do NOT mix content between sections.
- Do NOT add extra sections.

Output format (no extra text):

Summary:
<summary>

Explanation:
<explanation>

Key Points:
<bullet list>

Next Steps:
<bullet list>
"""
}

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

    # -----------------------------
    # Use the UPDATED global mode_instruction dictionary
    # -----------------------------
    instruction = mode_instruction[st.session_state.processing_mode]

    # -----------------------------
    # Build messages correctly
    # -----------------------------
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": instruction},
        {"role": "user", "content": extracted_text}
    ]

    # -----------------------------
    # Groq call
    # -----------------------------
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )

    raw_output = response.choices[0].message.content

    output = format_output_with_headers(raw_output, st.session_state.processing_mode)

    # Store document context
    st.session_state.last_document_text = extracted_text
    st.session_state.last_document_summary = output

    # Display output
    doc_header = f"### {st.session_state.processing_mode}: {st.session_state.last_document_name}"
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
