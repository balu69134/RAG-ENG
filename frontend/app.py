import os

import streamlit as st

from api_client import APIClientError, RagApiClient


API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
api = RagApiClient(API_URL)

st.set_page_config(
    page_title="RAG-ENG",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --rag-red: #c62828;
        --rag-red-hover: #a91f1f;
        --rag-ink: #202124;
        --rag-muted: #6b7280;
        --rag-border: #e5e7eb;
        --rag-sidebar: #f3f4f6;
        --rag-success: #16804a;
        --rag-error: #b42318;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #ffffff;
        color: var(--rag-ink);
    }

    [data-testid="stSidebar"] {
        background: var(--rag-sidebar);
        border-right: 1px solid var(--rag-border);
    }

    [data-testid="stSidebar"] > div:first-child {
        background: var(--rag-sidebar);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1120px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--rag-ink) !important;
        letter-spacing: -0.02em;
    }

    h1 {
        font-size: 1.8rem !important;
        line-height: 1.25 !important;
        font-weight: 650 !important;
        margin-bottom: 0.25rem !important;
    }

    h2 {
        font-size: 1.3rem !important;
        line-height: 1.35 !important;
        font-weight: 600 !important;
    }

    h3 {
        font-size: 1.05rem !important;
        line-height: 1.4 !important;
        font-weight: 600 !important;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--rag-muted);
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid var(--rag-border);
    }

    [data-testid="stTabs"] [role="tab"] {
        color: var(--rag-muted);
        border-bottom: 2px solid transparent;
        padding: 0.7rem 0.05rem;
        font-size: 0.95rem;
        font-weight: 500;
        transition: none;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--rag-red);
        border-bottom-color: var(--rag-red);
    }

    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background: var(--rag-red);
    }

    [data-testid="stBaseButton-primary"] {
        background: var(--rag-red) !important;
        border: 1px solid var(--rag-red) !important;
        border-radius: 4px !important;
        color: #ffffff !important;
        box-shadow: none !important;
        transition: none !important;
    }

    [data-testid="stBaseButton-primary"]:hover {
        background: var(--rag-red-hover) !important;
        border-color: var(--rag-red-hover) !important;
    }

    [data-testid="stBaseButton-secondary"] {
        border: 1px solid #d1d5db !important;
        border-radius: 4px !important;
        color: #374151 !important;
        background: #ffffff !important;
        box-shadow: none !important;
    }

    [data-baseweb="textarea"] {
        border-radius: 4px;
        border-color: #d1d5db;
        box-shadow: none;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #fafafa;
        border: 1px dashed #cbd0d7;
        border-radius: 4px;
    }

    [data-testid="stAlert"] {
        border-radius: 4px;
        box-shadow: none;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--rag-border);
        border-radius: 4px;
        box-shadow: none;
    }

    .rag-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #374151;
        font-size: 0.9rem;
        margin: 0.35rem 0 0.55rem;
    }

    .rag-status-dot {
        display: inline-block;
        width: 0.5rem;
        height: 0.5rem;
        border-radius: 50%;
        flex: 0 0 0.5rem;
    }

    .rag-status-dot.connected { background: var(--rag-success); }
    .rag-status-dot.disconnected { background: var(--rag-error); }

    .rag-document-detail {
        color: var(--rag-muted);
        font-size: 0.88rem;
        margin-top: 0.15rem;
    }

    [data-testid="stDeployButton"],
    #MainMenu,
    footer {
        display: none !important;
    }

    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.2rem 1rem 2rem;
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.85rem;
        }
        [data-testid="stTabs"] [role="tab"] {
            font-size: 0.88rem;
            padding: 0.65rem 0.02rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def show_backend_status() -> None:
    with st.sidebar:
        st.subheader("Backend status")
        try:
            health = api.health()
            connected = health.get("status") == "ok" and health.get("qdrant") == "connected"
        except APIClientError:
            connected = False

        if connected:
            status_text = "API and Qdrant connected"
            status_class = "connected"
        else:
            status_text = "Backend unavailable"
            status_class = "disconnected"

        st.markdown(
            f'<div class="rag-status"><span class="rag-status-dot {status_class}"></span>'
            f"<span>{status_text}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(f"API: {API_URL}")


def render_upload_screen() -> None:
    st.subheader("Add documents")
    st.caption("Upload one or more text-based PDF files. Scanned PDFs may need OCR first.")

    files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDFs are limited to 25 MB each by default.",
    )

    if files:
        st.caption("Selected files")
        for file in files:
            st.text(f"{file.name}  |  {format_size(file.size)}")

    if st.button("Process PDFs", type="primary", disabled=not files, key="process_pdfs"):
        progress = st.progress(0, text="Preparing uploads...")
        for index, uploaded_file in enumerate(files, start=1):
            progress.progress((index - 1) / len(files), text=f"Processing {uploaded_file.name}...")
            try:
                result = api.upload_pdf(uploaded_file.name, uploaded_file.getvalue())
                st.success(
                    f"{result['filename']} indexed: {result['page_count']} pages, "
                    f"{result['chunk_count']} passages."
                )
                if result.get("embedding_backend") == "hashed-lexical":
                    st.caption(
                        "Retrieval is using lexical embeddings on this machine; "
                        "related wording may be missed."
                    )
            except APIClientError as exc:
                st.error(f"{uploaded_file.name}: {exc}")
            progress.progress(index / len(files), text=f"Processed {index} of {len(files)}")
        progress.empty()


def render_ask_screen() -> None:
    st.subheader("Ask your documents")
    query = st.text_area(
        "Question",
        placeholder="Ask a question about your uploaded documents...",
        max_chars=4000,
        height=132,
    )
    top_k = st.slider("Passages to retrieve", min_value=1, max_value=10, value=5)
    st.caption(f"{top_k} passages selected")
    submitted = st.button("Get answer", type="primary", key="get_answer")

    if submitted:
        st.session_state.pop("last_answer", None)
        st.session_state.pop("ask_error", None)
        if not query.strip():
            st.session_state["ask_error"] = "Enter a question first."
        else:
            with st.spinner("Searching your documents and preparing an answer..."):
                try:
                    st.session_state["last_answer"] = api.ask(query.strip(), top_k)
                except APIClientError as exc:
                    st.session_state["ask_error"] = str(exc)

    if error := st.session_state.get("ask_error"):
        st.error(error)

    result = st.session_state.get("last_answer")
    if not result:
        return

    st.divider()
    st.subheader("Answer")
    st.markdown(result["answer"])
    st.caption(f"Retrieval mode: {result.get('embedding_backend', 'unknown')}")

    sources = result.get("sources", [])
    if not sources:
        st.info("No matching document passages were found.")
        return

    st.markdown("##### Retrieved sources")
    for source in sources:
        source_label = (
            f"[{source['source_id']}] {source['filename']}  |  "
            f"Page {source['page_number']}  |  Similarity {source['score']:.3f}"
        )
        with st.expander(source_label):
            st.text(source["excerpt"])


def render_documents_screen() -> None:
    st.subheader("Indexed documents")

    if notice := st.session_state.pop("document_notice", None):
        st.success(notice)

    try:
        documents = api.list_documents()
    except APIClientError as exc:
        st.error(str(exc))
        return

    if not documents:
        st.info("No PDFs have been uploaded yet.")
        return

    for index, document in enumerate(documents):
        details, action = st.columns([6, 1], vertical_alignment="center")
        with details:
            st.write(document["filename"])
            st.markdown(
                f'<div class="rag-document-detail">{document["page_count"]} pages'
                f'  |  {document["chunk_count"]} passages</div>',
                unsafe_allow_html=True,
            )
        with action:
            if st.button("Delete", key=f"delete_{document['document_id']}"):
                try:
                    api.delete_document(document["document_id"])
                    st.session_state["document_notice"] = f"Deleted {document['filename']}."
                    st.rerun()
                except APIClientError as exc:
                    st.error(str(exc))
        if index < len(documents) - 1:
            st.divider()


show_backend_status()

st.title("RAG-ENG")
st.caption("Ask questions and get answers grounded in your uploaded PDFs.")

upload_tab, ask_tab, documents_tab = st.tabs(
    ["Upload PDFs", "Ask a question", "Documents"]
)

with upload_tab:
    render_upload_screen()

with ask_tab:
    render_ask_screen()

with documents_tab:
    render_documents_screen()
