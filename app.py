import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.document_loader import extract_text
from utils.text_processor import clean_text, split_text
from utils.embedding import EmbeddingModel
from utils.vector_store import VectorStore
from utils.kimi_client import KimiClient

load_dotenv()

BASE_DIR = Path(__file__).parent
DOC_DIR = BASE_DIR / "data" / "documents"
INDEX_DIR = BASE_DIR / "data" / "index"
DOC_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Nexa — Document AI Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# UI theme
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
    --accent: #7c5cff;
    --accent-2: #4f8cff;
    --card: rgba(255,255,255,.045);
    --border: rgba(255,255,255,.10);
    --muted: rgba(255,255,255,.62);
}

.block-container { padding: 1.4rem 2rem 3rem; max-width: 1450px; }
[data-testid="stSidebar"] { min-width: 300px; max-width: 320px; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }

.hero {
    padding: 28px 30px;
    border: 1px solid var(--border);
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(124,92,255,.16), rgba(79,140,255,.08) 55%, rgba(255,255,255,.025));
    margin-bottom: 18px;
}
.hero-kicker {
    color: #a995ff;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    margin-bottom: 7px;
}
.hero h1 { margin: 0; font-size: 2.35rem; letter-spacing: -.045em; }
.hero p { margin: 8px 0 0; color: var(--muted); font-size: 1rem; }

.stat-card {
    padding: 17px 18px;
    border: 1px solid var(--border);
    border-radius: 17px;
    background: var(--card);
    min-height: 92px;
}
.stat-label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
.stat-value { font-size: 1.45rem; font-weight: 700; margin-top: 5px; }

.feature-card {
    padding: 18px;
    border: 1px solid var(--border);
    border-radius: 18px;
    background: var(--card);
    height: 100%;
}
.feature-icon { font-size: 1.3rem; }
.feature-title { font-weight: 700; margin-top: 8px; }
.feature-copy { color: var(--muted); font-size: .88rem; margin-top: 4px; }

.sidebar-brand {
    padding: 8px 4px 18px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 18px;
}
.sidebar-brand .name { font-size: 1.35rem; font-weight: 800; letter-spacing: -.03em; }
.sidebar-brand .tag { color: var(--muted); font-size: .8rem; margin-top: 2px; }

.section-label {
    color: var(--muted);
    font-size: .73rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin: 18px 0 8px;
}

.empty-state {
    text-align: center;
    padding: 55px 25px;
    border: 1px dashed rgba(255,255,255,.16);
    border-radius: 22px;
    background: rgba(255,255,255,.02);
}
.empty-state .icon { font-size: 2.5rem; }
.empty-state h3 { margin: 10px 0 5px; }
.empty-state p { color: var(--muted); margin: 0 auto; max-width: 520px; }

div[data-testid="stChatMessage"] { border-radius: 18px; }
div[data-testid="stChatInput"] { margin-top: 8px; }

.source-pill {
    display: inline-block;
    padding: 4px 9px;
    border-radius: 999px;
    border: 1px solid var(--border);
    font-size: .74rem;
    color: var(--muted);
}

footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_embedding_model():
    return EmbeddingModel()


@st.cache_resource
def get_vector_store():
    return VectorStore(INDEX_DIR)


def supported_documents():
    return sorted(
        [
            path
            for path in DOC_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt"}
        ],
        key=lambda p: p.name.lower(),
    )


def save_uploaded_files(uploaded_files):
    saved = []
    for file in uploaded_files:
        target = DOC_DIR / Path(file.name).name
        target.write_bytes(file.getbuffer())
        saved.append(target)
    return saved


def rebuild_index():
    documents = []
    errors = []

    for path in supported_documents():
        try:
            raw = extract_text(path)
            cleaned = clean_text(raw)
            chunks = split_text(cleaned)
            for i, chunk in enumerate(chunks):
                documents.append(
                    {
                        "text": chunk,
                        "source": path.name,
                        "chunk_id": i,
                    }
                )
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if not documents:
        return 0, errors

    model = get_embedding_model()
    store = get_vector_store()
    texts = [d["text"] for d in documents]
    embeddings = model.encode(texts)
    store.build(embeddings, documents)
    return len(documents), errors


def answer_question(question, top_k=5):
    store = get_vector_store()
    if not store.exists():
        return None, []

    model = get_embedding_model()
    q_embedding = model.encode([question])[0]
    results = store.search(q_embedding, top_k=top_k)

    context_blocks = []
    for result in results:
        context_blocks.append(
            f"[Source: {result['source']} | Chunk: {result['chunk_id']}]\n{result['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    kimi = KimiClient(
        api_key=os.getenv("KIMI_API_KEY", ""),
        base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1"),
        model=os.getenv("KIMI_MODEL", "kimi-k2.5"),
    )
    answer = kimi.answer(question, context)
    return answer, results


def render_sources(results):
    if not results:
        return
    with st.expander(f"📚 Sources used · {len(results)} retrieved chunks"):
        for source in results:
            st.markdown(
                f"**{source['source']}** · chunk {source['chunk_id']} · "
                f"similarity `{source['score']:.3f}`"
            )


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="name">✦ Nexa AI</div>
            <div class="tag">Your private document workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Knowledge base</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Add documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX and TXT.",
    )
    if st.button("＋ Save documents", use_container_width=True, type="primary"):
        if uploaded:
            saved = save_uploaded_files(uploaded)
            st.success(f"Saved {len(saved)} document(s).")
        else:
            st.info("Select at least one document first.")

    if st.button("⟳ Build / Rebuild index", use_container_width=True):
        with st.spinner("Building your knowledge base..."):
            count, errors = rebuild_index()
        if count:
            st.success(f"Index ready · {count} chunks")
        else:
            st.warning("No supported documents found.")
        for error in errors:
            st.warning(f"Could not process {error}")

    st.markdown('<div class="section-label">Workspace</div>', unsafe_allow_html=True)
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑 Clear local data", use_container_width=True):
        for path in DOC_DIR.iterdir():
            if path.is_file():
                path.unlink()
        get_vector_store().clear()
        st.session_state.messages = []
        st.success("Documents and index cleared.")
        st.rerun()

    st.markdown('<div class="section-label">System status</div>', unsafe_allow_html=True)
    documents = supported_documents()
    index_ready = get_vector_store().exists()
    status_text = "Ready" if index_ready else "Not built"
    st.caption(f"📄 Documents: **{len(documents)}**")
    st.caption(f"🧠 Index: **{status_text}**")
    st.caption(f"⚡ Model: **{os.getenv('KIMI_MODEL', 'kimi-k2.5')}**")


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">NLP · RAG · Kimi</div>
        <h1>Ask your documents anything.</h1>
        <p>Upload your files, build a searchable knowledge base, and get grounded answers with source references.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">Documents</div><div class="stat-value">{len(supported_documents())}</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">Knowledge base</div><div class="stat-value">{"Ready" if get_vector_store().exists() else "Not built"}</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">Conversation</div><div class="stat-value">{len(st.session_state.messages)} messages</div></div>',
        unsafe_allow_html=True,
    )

if not st.session_state.messages:
    st.markdown("### ✨ Start a conversation")
    f1, f2, f3 = st.columns(3)
    features = [
        ("📄", "Upload", "Add PDF, DOCX or TXT files from the sidebar."),
        ("🔎", "Retrieve", "Relevant document chunks are found with semantic search."),
        ("💬", "Ask", "Kimi generates an answer grounded in the retrieved context."),
    ]
    for column, (icon, title, copy) in zip((f1, f2, f3), features):
        with column:
            st.markdown(
                f'<div class="feature-card"><div class="feature-icon">{icon}</div>'
                f'<div class="feature-title">{title}</div><div class="feature-copy">{copy}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("")
    st.markdown(
        '<div class="empty-state"><div class="icon">💡</div><h3>Your workspace is ready</h3>'
        '<p>Upload your study material or project documents, build the index, then use the chat box below to ask questions.</p></div>',
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

question = st.chat_input("Ask a question about your documents…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your knowledge base…"):
            try:
                answer, results = answer_question(question)
                if answer is None:
                    answer = (
                        "Your knowledge base is empty. Upload PDF, DOCX or TXT files "
                        "from the sidebar and click **Build / Rebuild index** first."
                    )
                    results = []
                st.markdown(answer)
                render_sources(results)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": results,
                    }
                )
            except Exception as exc:
                st.error(str(exc))
                st.info(
                    "Check your KIMI_API_KEY in .env and confirm the Kimi endpoint/model "
                    "configured in your environment."
                )
