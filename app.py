from typing import List, Tuple
import base64

import streamlit as st
import streamlit.components.v1 as components

from pdf_qa import (
    extract_text_by_page,
    chunk_texts,
    SentenceEmbedder,
    build_nn_index,
    retrieve_top_k,
    get_qa_pipeline,
    answer_with_context,
)
from transcription import record_audio, transcribe_audio_array


st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")
st.title("📄 Chat with PDF")
st.caption("Upload a PDF, ask a question by voice or text, and get answers grounded in the PDF.")

# Inline CSS to visually place the clear (✕) buttons inside the input field area
st.markdown(
    """
    <style>
      /* Compact round clear buttons */
      button[title="clear-text"], button[title="clear-voice"] {
        padding: 0 !important;
        width: 30px !important;
        height: 30px !important;
        border-radius: 15px !important;
        line-height: 1 !important;
      }
      /* Nudge the clear buttons left so they appear inside the right edge of the input */
      div[data-testid="column"]:has(button[title="clear-text"]) button[title="clear-text"],
      div[data-testid="column"]:has(button[title="clear-voice"]) button[title="clear-voice"] {
        margin-left: -46px !important; /* pull into input */
        margin-top: 6px !important;    /* vertical center approx */
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _clear_voice_input_state() -> None:
    # Clears voice-mode input and any displayed answer
    st.session_state['last_transcript'] = ""
    st.session_state['transcribed_question_input'] = ""
    st.session_state['answer'] = None


def _clear_text_input_state() -> None:
    # Clears text-mode input and any displayed answer
    st.session_state['text_question'] = ""
    st.session_state['answer'] = None
    st.session_state['pending_question'] = None


@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceEmbedder:
    return SentenceEmbedder()


@st.cache_resource(show_spinner=False)
def get_qa() -> object:
    return get_qa_pipeline()


def build_index_from_pdf(pdf_bytes: bytes):
    pages: List[str] = extract_text_by_page(pdf_bytes)
    chunks_with_page: List[Tuple[str, int]] = chunk_texts(pages)
    texts = [c for c, _ in chunks_with_page]
    embedder = get_embedder()
    embs = embedder.encode(texts)
    nn = build_nn_index(embs)
    return chunks_with_page, nn, embedder


with st.sidebar:
    st.header("PDF")
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    st.divider()
    st.header("Question input")
    mode = st.radio("Mode", ["Text", "Voice"], index=0, horizontal=True)
    record_secs = None
    if mode == "Voice":
        record_secs = st.number_input("Record seconds", min_value=2, max_value=30, value=5)


if pdf_file is None:
    st.info("Upload a PDF to begin.")
    st.stop()


if "index" not in st.session_state or st.session_state.get("pdf_name") != pdf_file.name:
    try:
        with st.spinner("Building index..."):
            pdf_bytes = pdf_file.read()
            chunks_with_page, nn, embedder = build_index_from_pdf(pdf_bytes)
            st.session_state["index"] = {
                "chunks": chunks_with_page,
                "nn": nn,
                "embedder": embedder,
            }
            st.session_state["pdf_name"] = pdf_file.name
            st.session_state["pdf_bytes"] = pdf_bytes
            # Reset chat history when switching PDFs
            st.session_state["history"] = []
    except Exception as e:
        st.error(str(e))
        st.stop()


col_pdf, col_chat = st.columns([1, 2])

with col_pdf:
    st.subheader("PDF preview")
    pdf_bytes = st.session_state.get("pdf_bytes")
    if pdf_bytes:
        # Render first pages as images wrapped in an anchor to open the full PDF in a new tab
        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        opened = False
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_to_show = min(len(doc), 3)
            for i in range(pages_to_show):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                png_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                elem_id = f"pdf_link_{i}"
                html = f"""
                    <a id=\"{elem_id}\" target=\"_blank\" rel=\"noopener noreferrer\">
                        <img src=\"data:image/png;base64,{png_b64}\" style=\"width:100%;display:block;border:1px solid #eee;border-radius:8px\" />
                    </a>
                    <div style=\"text-align:center;color:#888;font-size:12px;margin-top:4px;\">Click preview to open full PDF</div>
                    <script>
                      (function() {{
                        const base64 = "{pdf_b64}";
                        const byteChars = atob(base64);
                        const byteNumbers = new Array(byteChars.length);
                        for (let j = 0; j < byteChars.length; j++) {{ byteNumbers[j] = byteChars.charCodeAt(j); }}
                        const byteArray = new Uint8Array(byteNumbers);
                        const blob = new Blob([byteArray], {{ type: 'application/pdf' }});
                        const url = URL.createObjectURL(blob);
                        const a = document.getElementById('{elem_id}');
                        if (a) a.href = url;
                      }})();
                    </script>
                """
                components.html(html, height=min(900, getattr(pix, 'height', 600) + 50))
                opened = True
        except ModuleNotFoundError:
            st.info("Optional dependency 'pymupdf' not installed.")
        except Exception:
            pass
        if not opened:
            # Fallback: simple link created via Blob URL
            components.html(
                f"""
                <a id=\"pdf_fallback_link\" target=\"_blank\" rel=\"noopener noreferrer\">Open PDF</a>
                <script>
                  (function() {{
                    const base64 = "{pdf_b64}";
                    const byteChars = atob(base64);
                    const byteNumbers = new Array(byteChars.length);
                    for (let i = 0; i < byteChars.length; i++) {{ byteNumbers[i] = byteChars.charCodeAt(i); }}
                    const byteArray = new Uint8Array(byteNumbers);
                    const blob = new Blob([byteArray], {{ type: 'application/pdf' }});
                    const url = URL.createObjectURL(blob);
                    const a = document.getElementById('pdf_fallback_link');
                    if (a) a.href = url;
                  }})();
                </script>
                """,
                height=24,
            )
    else:
        st.info("PDF not available for preview.")

with col_chat:
    st.subheader("Ask a question")
    question = ""
    if 'last_transcript' not in st.session_state:
        st.session_state['last_transcript'] = ""
    if 'answer' not in st.session_state:
        st.session_state['answer'] = None
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'pending_question' not in st.session_state:
        st.session_state['pending_question'] = None

    do_ask = False
    if mode == "Text":
        c1, c2, c3 = st.columns([6, 0.2, 1])
        with c1:
            question = st.text_input(
                "Type your question",
                value="",
                key="text_question",
                label_visibility="collapsed",
                placeholder="Type your question",
            )
        with c2:
            st.button("✕", use_container_width=True, key="discard_text_btn", on_click=_clear_text_input_state, help="Clear", type="secondary",)
        with c3:
            do_ask = st.button("Ask", use_container_width=True, key="ask_text")
    else:
        if st.button("Ask", key="record_btn"):
            # Start a new recording: clear any previously shown answer
            st.session_state['answer'] = None
            with st.spinner("Recording and transcribing..."):
                audio = record_audio(duration=int(record_secs))
                question_text = transcribe_audio_array(audio)
                if question_text:
                    st.session_state['last_transcript'] = question_text
                    # Keep the editable input in sync with the latest transcript
                    st.session_state['transcribed_question_input'] = question_text
                    question = question_text
                else:
                    st.warning("Could not transcribe audio.")
        # Always show editable input row in voice mode
        if 'transcribed_question_input' not in st.session_state:
            st.session_state['transcribed_question_input'] = st.session_state.get('last_transcript', "")

        c1, c2, c4 = st.columns([6, 0.2, 1])
        with c1:
            question = st.text_input(
                "Transcribed question (editable)",
                key="transcribed_question_input",
                label_visibility="collapsed",
                placeholder="Edit transcribed question",
            )
        with c2:
            st.button("✕", use_container_width=True, key="discard_btn", on_click=_clear_voice_input_state, help="Clear", type="secondary")
        with c4:
            do_ask = st.button("Ask", use_container_width=True, key="ask_voice")

            # Discard handled via on_click callback above

            # No record-again flow; user can discard and press Ask to re-record

    # If user pressed Ask, clear any shown answer immediately and capture pending question
    if do_ask:
        st.session_state['answer'] = None
        st.session_state['pending_question'] = question

    if do_ask and question:
        with st.spinner("Retrieving and answering..."):
            chunks = st.session_state["index"]["chunks"]
            nn = st.session_state["index"]["nn"]
            embedder = st.session_state["index"]["embedder"]
            topk = retrieve_top_k(question, embedder.encode, nn, chunks, k=5)
            qa = get_qa()
            result = answer_with_context(question, topk, qa)
        st.session_state['answer'] = result.get("answer", "")
        # Append to history exactly once per Ask
        asked_text = st.session_state.get('pending_question') or question
        if asked_text:
            st.session_state['history'].append({"q": asked_text, "a": st.session_state['answer']})
            st.session_state['pending_question'] = None

    if st.session_state.get('answer'):
        st.markdown("**Answer**")
        st.write(st.session_state['answer']) 

    # History section (shown only when there is at least one entry)
    if st.session_state['history']:
        st.divider()
        hcol1, hcol2 = st.columns([3, 1])
        with hcol1:
            st.subheader("History")
        with hcol2:
            if st.button("Clear history", use_container_width=True, key="clear_history_btn"):
                st.session_state['history'] = []
                st.session_state['answer'] = None
                st.rerun()

        for item in reversed(st.session_state['history']):
            st.markdown("**Q:** " + item.get("q", ""))
            st.markdown("**A:** " + item.get("a", ""))
            st.divider()


