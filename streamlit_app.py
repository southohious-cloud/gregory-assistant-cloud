import os
import streamlit as st
from groq import Groq
from PIL import Image
import pdfplumber

# -----------------------------
# Fixed-height scrollable output panel (CSS)
# -----------------------------
st.markdown("""
<style>
.output-panel {
    height: 450px;
    overflow-y: auto;
    padding: 1rem;
    border: 1px solid #ccc;
    border-radius: 6px;
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

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

Follow the mode-specific processing instructions provided in the next system message exactly.
Only produce the content required by the selected mode.
Do not add summaries, explanations, key points, next steps, or multiple sections unless the mode explicitly requires them.

General rules:
- Obey the mode_instruction text exactly.
- Never mix content between modes.
- Never invent details that are not present or implied in the document.
- Keep all writing clear, factual, and concise.
- For Summary, Explanation, Key Points, and Next Steps modes, output only the requested content with no headers.
- For Everything mode, output all four sections in the exact order and format specified.
- When inferring (Next Steps mode only), use reasonable, non-speculative logic grounded in the document.

Transformations (only when explicitly requested):
- Rewrite simpler
- Rewrite more formal
- Rewrite as an email
- Rewrite as a checklist
- Rewrite as a step-by-step plan
- Explain deeper

Transformation rules:
- Do not add new information.
- Preserve the meaning of the original content.
- Match the tone and structure of the requested transformation.
- Output only the transformed content with no extra commentary.

Your behavior must remain stable, predictable, and strictly mode-driven.
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
# Formatting Wrapper (Strict Mode-Wide Enforcement)
# -----------------------------
def format_output_with_headers(raw_output: str, mode: str) -> str:
    """
    Strict formatting enforcement for ALL modes:
    - Strips ALL model-generated headers
    - Removes ALL extra sections in single-mode
    - Normalizes whitespace
    - Normalizes bullet formatting
    - Enforces paragraph-only output for Summary + Explanation
    - Enforces bullet-only output for Key Points + Next Steps
    - Enforces correct section order for Everything mode
    """

    text = raw_output.strip()

    # All possible section headers the model might generate
    all_headers = ["Summary", "Explanation", "Key Points", "Next Steps"]

    # Normalize helper for header detection
    def normalize_header(line: str) -> str:
        return " ".join(line.strip().lower().replace(":", "").replace("#", "").split())

    # -----------------------------
    # GLOBAL CLEANUP FOR ALL MODES
    # -----------------------------
    # Remove accidental leading headers
    for h in all_headers:
        if text.lower().startswith(h.lower()):
            text = text[len(h):].strip()

    # Normalize whitespace
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    # Normalize bullets
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("* ", "• ", "‣ ", "- ")):
            cleaned_lines.append("- " + stripped[2:].strip())
        else:
            cleaned_lines.append(stripped)
    text = "\n".join(cleaned_lines).strip()

    # -----------------------------
    # SINGLE-MODE BEHAVIOR
    # -----------------------------
    if mode != "Everything":

        # Split into lines
        lines = text.split("\n")
        cleaned = []

        for line in lines:
            norm = normalize_header(line)

            # If this line is ANY section header that is NOT the selected mode → STOP
            if norm in [h.lower() for h in all_headers] and norm != mode.lower():
                break

            # Skip header for the selected mode
            if norm == mode.lower():
                continue

            cleaned.append(line)

        text = "\n".join(cleaned).strip()

        # -----------------------------
        # Mode-specific enforcement
        # -----------------------------
        if mode in ["Summary", "Explanation"]:
            # Must be a single paragraph
            text = text.replace("\n", " ").strip()
            return text

        if mode in ["Key Points", "Next Steps"]:
            # Must be bullets only
            bullets = [l for l in text.split("\n") if l.startswith("- ")]
            return "\n".join(bullets).strip()

        return text

    # -----------------------------
    # EVERYTHING MODE
    # -----------------------------
    # Split into sections (model usually separates with blank lines)
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
            bullets = [l for l in content.split("\n") if l.startswith("- ")]
            content = "\n".join(bullets).strip()

        cleaned_sections.append(f"### {header}\n\n{content}")

    return "\n\n".join(cleaned_sections).strip()



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
        ["Summary", "Explanation", "Key Points", "Next Steps", "Everything"],
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
# File uploader (MAIN PAGE)
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a file for instant processing (PDF, TXT, PNG, JPG)",
    type=["pdf", "txt", "png", "jpg", "jpeg"],
    label_visibility="visible"
)

# -----------------------------
# OUTPUT CONTAINER (main page)
# -----------------------------
output_container = st.container()

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
# REMOVE HISTORY COMPLETELY
# -----------------------------
def render_collapsible_history(history):
    return  # History disabled

# -----------------------------
# FILE UPLOAD + AUTO-REPROCESS
# -----------------------------

# Detect mode change
if "last_mode" not in st.session_state:
    st.session_state.last_mode = processing_mode

mode_changed = processing_mode != st.session_state.last_mode
st.session_state.last_mode = processing_mode

# Trigger processing if:
# 1. A new file is uploaded
# 2. OR we already have a previous document stored
if uploaded_file is not None or st.session_state.get("last_document_text"):

    # -----------------------------
    # CASE 1 — New file uploaded
    # -----------------------------
    if uploaded_file is not None:
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

    # -----------------------------
    # DISPLAY ONLY THE CURRENT OUTPUT (INSTANT REPLACEMENT)
    # -----------------------------
    with output_container:
        st.markdown(
            f"""
            <div class='output-panel'>
                <h3>{st.session_state.processing_mode}: {st.session_state.last_document_name}</h3>
                {output}
            </div>
            """,
            unsafe_allow_html=True
        )
# -----------------------------
# ⭐ NEW: Post-output transformation buttons
# -----------------------------
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

    # Replace output directly
    with output_container:
        st.markdown(
            f"""
            <div class='output-panel'>
                <h3>{action}</h3>
                {transformed}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.session_state.last_document_summary = transformed
    st.rerun()

# -----------------------------
# Chat Input (kept functional)
# -----------------------------
user_input = st.chat_input("Type your message...")

if user_input:
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

    st.markdown(assistant_reply)

# ----------------------------------------
# Scroll reset on every rerun (content-aware)
# ----------------------------------------
st.markdown("""
<script>
function resetWhenReady() {
    const panel = document.querySelector('.output-panel');
    if (!panel) {
        setTimeout(resetWhenReady, 30);
        return;
    }

    // Wait until content is actually inside the panel
    if (panel.innerText.trim().length < 10) {
        setTimeout(resetWhenReady, 30);
        return;
    }

    // Now safe to reset scroll
    panel.scrollTop = 0;
}

resetWhenReady();
</script>
""", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Gregory's Personal Assistant Cloud Version")
