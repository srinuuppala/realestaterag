"""Real Estate AI Assistant — Streamlit front end.

Login → chat. Answers stream token by token, every answer carries the documents
it was built from, and the assistant refuses rather than guesses when the
knowledge base has nothing to say.
"""

from __future__ import annotations

import streamlit as st

from src import auth
from src.config import get_settings
from src.loader import discover
from src.memory import ConversationMemory
from src.rag_chain import RagChain
from src.retriever import HybridRetriever
from src.utils import get_logger
from src.vectorstore import get_vectorstore

log = get_logger(__name__)
settings = get_settings()

st.set_page_config(
    page_title="Real Estate AI Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

STARTERS = [
    "What is the payment plan for Skyline Horizon Towers?",
    "Which projects are under ₹1 crore, and where are they?",
    "What is the cancellation and refund policy at Urban Nest?",
    "Compare the possession dates of the two Meridian projects.",
]

PALETTES = {
    "light": {
        "bg": "#F4F5F2", "surface": "#FFFFFF", "raised": "#F8F9F6",
        "ink": "#12151A", "muted": "#5C6672", "line": "#DFE1DC", "accent": "#1F3AE0",
        "accent_soft": "#E6EAFD",
    },
    "dark": {
        "bg": "#0B0D11", "surface": "#14171D", "raised": "#1A1E26",
        "ink": "#E9EBEE", "muted": "#8A93A0", "line": "#262B34", "accent": "#8098FF",
        "accent_soft": "#1A213E",
    },
}


def inject_theme(theme: str) -> None:
    c = PALETTES[theme]
    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Space+Grotesk:wght@600;700&display=swap');

          .stApp {{ background: {c["bg"]}; }}
          .stApp, .stApp p, .stApp li, .stApp span, .stApp label {{
            color: {c["ink"]}; font-family: 'Inter', system-ui, sans-serif;
          }}
          .stApp h1, .stApp h2, .stApp h3 {{
            font-family: 'Space Grotesk', system-ui, sans-serif;
            color: {c["ink"]}; letter-spacing: -0.01em;
          }}
          section[data-testid="stSidebar"] {{
            background: {c["surface"]}; border-right: 1px solid {c["line"]};
          }}
          [data-testid="stChatMessage"] {{
            background: {c["surface"]}; border: 1px solid {c["line"]};
            border-radius: 6px; padding: 1rem 1.15rem; margin-bottom: 0.6rem;
          }}
          .stChatInput textarea, .stTextInput input {{
            background: {c["surface"]} !important; color: {c["ink"]} !important;
            border: 1px solid {c["line"]} !important; border-radius: 6px !important;
          }}
          .stButton button {{
            background: {c["surface"]}; color: {c["ink"]};
            border: 1px solid {c["line"]}; border-radius: 6px;
            font-weight: 500; transition: border-color .15s ease;
          }}
          .stButton button:hover {{ border-color: {c["accent"]}; color: {c["accent"]}; }}
          .stButton button[kind="primary"] {{
            background: {c["accent"]}; color: #FFFFFF; border-color: {c["accent"]};
          }}
          .stButton button[kind="primary"]:hover {{ opacity: .9; color: #FFFFFF; }}
          [data-testid="stExpander"] {{
            background: {c["surface"]}; border: 1px solid {c["line"]}; border-radius: 6px;
          }}
          .annotation {{
            font-family: 'JetBrains Mono', monospace; font-size: 0.69rem;
            text-transform: uppercase; letter-spacing: .08em; color: {c["muted"]};
          }}
          .cite-card {{
            border: 1px solid {c["line"]}; border-left: 3px solid {c["accent"]};
            background: {c["raised"]}; border-radius: 4px;
            padding: .7rem .9rem; margin-bottom: .5rem;
          }}
          .cite-head {{
            font-family: 'JetBrains Mono', monospace; font-size: .72rem;
            color: {c["accent"]}; letter-spacing: .04em;
          }}
          .cite-title {{ font-weight: 600; font-size: .9rem; color: {c["ink"]}; }}
          .cite-snippet {{ font-size: .82rem; color: {c["muted"]}; margin-top: .3rem; line-height: 1.5; }}
          .kb-grid {{
            display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; margin-top: .4rem;
          }}
          .kb-cell {{
            background: {c["raised"]}; border: 1px solid {c["line"]};
            border-radius: 4px; padding: .5rem .6rem;
          }}
          .kb-value {{
            font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem;
            font-weight: 700; color: {c["ink"]};
          }}
          .hero-rule {{ height: 1px; background: {c["line"]}; margin: 1rem 0 1.4rem; }}
          .badge {{
            display: inline-block; background: {c["accent_soft"]}; color: {c["accent"]};
            font-family: 'JetBrains Mono', monospace; font-size: .66rem;
            text-transform: uppercase; letter-spacing: .07em;
            padding: .2rem .45rem; border-radius: 3px; margin-right: .3rem;
          }}
          #MainMenu, footer {{ visibility: hidden; }}
          .block-container {{ padding-top: 2.2rem; max-width: 62rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached resources — built once per process, not once per rerun
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_chain() -> tuple[RagChain, dict]:
    store, chunks = get_vectorstore()
    retriever = HybridRetriever(store, chunks)

    files = discover()
    stats = {
        "documents": len(files),
        "chunks": len(chunks),
        "formats": sorted({path.suffix.lstrip(".").upper() for path in files}),
        "projects": sorted({c.metadata.get("project", "") for c in chunks} - {"", "All projects"}),
    }
    return RagChain(retriever), stats


def ensure_state() -> None:
    st.session_state.setdefault("theme", "light")
    st.session_state.setdefault("memory", ConversationMemory())
    st.session_state.setdefault("pending_question", None)


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------
def render_login() -> None:
    _, centre, _ = st.columns([1, 1.5, 1])

    with centre:
        st.markdown("<div style='height:5vh'></div>", unsafe_allow_html=True)
        st.markdown('<p class="annotation">Internal tool · Sales & support</p>', unsafe_allow_html=True)
        st.markdown("# Real Estate AI Assistant")
        st.markdown(
            "Ask anything about our projects, payment plans, RERA filings and policies. "
            "Every answer is drawn from the document repository and cites its sources."
        )
        st.markdown('<div class="hero-rule"></div>', unsafe_allow_html=True)

        with st.form("login", clear_on_submit=False):
            username = st.text_input("Username", placeholder="demo")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

        if submitted:
            if auth.attempts_remaining() == 0:
                st.error("Too many failed attempts. Reload the page to try again.")
            elif auth.login(username, password):
                st.rerun()
            else:
                st.error(f"Incorrect username or password. {auth.attempts_remaining()} attempts left.")

        st.caption(f"Demo credentials — `{settings.auth_username}` / `{settings.auth_password}`")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(stats: dict) -> None:
    memory: ConversationMemory = st.session_state.memory

    badges = "".join(f'<span class="badge">{fmt}</span>' for fmt in stats["formats"])

    with st.sidebar:
        st.markdown("### Real Estate Assistant")
        st.markdown(
            f'<p class="annotation">Signed in as {auth.current_user()}</p>',
            unsafe_allow_html=True,
        )

        left, right = st.columns(2)
        with left:
            if st.button("New chat", use_container_width=True):
                memory.clear()
                st.rerun()
        with right:
            next_theme = "dark" if st.session_state.theme == "light" else "light"
            if st.button(f"{'🌙' if next_theme == 'dark' else '☀️'} Theme", use_container_width=True):
                st.session_state.theme = next_theme
                st.rerun()

        st.divider()

        st.markdown('<p class="annotation">Knowledge base</p>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="kb-grid">
              <div class="kb-cell"><div class="kb-value">{stats["documents"]}</div>
                <div class="annotation">documents</div></div>
              <div class="kb-cell"><div class="kb-value">{stats["chunks"]}</div>
                <div class="annotation">chunks</div></div>
            </div>
            <p style="margin-top:.6rem">{badges}</p>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown('<p class="annotation">This conversation</p>', unsafe_allow_html=True)
        questions = [turn.content for turn in memory.turns if turn.role == "user"]
        if not questions:
            st.caption("No questions yet.")
        else:
            for index, question in enumerate(reversed(questions[-8:]), start=1):
                if st.button(
                    f"{question[:46]}{'…' if len(question) > 46 else ''}",
                    key=f"history-{index}",
                    use_container_width=True,
                ):
                    st.session_state.pending_question = question
                    st.rerun()

            if st.button("Clear conversation", use_container_width=True):
                memory.clear()
                st.rerun()

        st.divider()

        with st.expander("Configuration"):
            st.markdown(
                f"""
                | | |
                |---|---|
                | Embeddings | `{settings.embedding_model.split("/")[-1]}` |
                | Retrieval | Hybrid — FAISS + BM25 (RRF) |
                | Chunks returned | `{settings.top_k}` of `{settings.fetch_k}` |
                | Memory window | `{settings.memory_window}` turns |
                | LLM | `{settings.llm_model if settings.llm_configured else "not configured"}` |
                | Provider | `{settings.provider if settings.llm_configured else "—"}` |
                """
            )
            if not settings.llm_configured:
                st.warning(
                    "No `GROQ_API_KEY`, so the assistant runs in **extractive mode**: retrieval "
                    "and citations work, but answers are the retrieved passages, not a summary.",
                    icon="⚠️",
                )

        if st.button("Sign out", use_container_width=True):
            auth.logout()
            st.rerun()


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
def render_citations(citations: list[dict]) -> None:
    if not citations:
        return

    label = f"Sources — {len(citations)} document{'s' if len(citations) > 1 else ''}"
    with st.expander(label):
        for citation in citations:
            blocks = ", ".join(f"[{block}]" for block in citation["blocks"])
            head = f'{blocks} · {citation["file_type"].upper()} · {citation["category"]}'
            st.markdown(
                f"""
                <div class="cite-card">
                  <div class="cite-head">{head}</div>
                  <div class="cite-title">{citation["title"]}</div>
                  <div class="annotation">{citation["source"]} · {citation["project"]}</div>
                  <div class="cite-snippet">{citation["snippet"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_transcript(memory: ConversationMemory) -> None:
    for turn in memory.turns:
        with st.chat_message(turn.role, avatar="🧑" if turn.role == "user" else "🏢"):
            st.markdown(turn.content)
            if turn.role == "assistant":
                render_citations(turn.citations)


def answer_question(chain: RagChain, question: str) -> None:
    memory: ConversationMemory = st.session_state.memory

    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🏢"):
        placeholder = st.empty()
        citation_slot = st.container()
        text = ""
        response = None

        with st.spinner("Searching the knowledge base…"):
            generator = chain.answer(question, memory)
            try:
                first_token, response = next(generator)
            except StopIteration:
                first_token, response = "", None

        if response is None:
            placeholder.error("Something went wrong while answering. Please try again.")
            return

        text += first_token
        # The trailing block is the typing cursor — it is what makes streaming read
        # as *typing* rather than as text appearing in jumps.
        placeholder.markdown(text + " ▌")

        for token, latest in generator:
            response = latest
            text += token
            placeholder.markdown(text + " ▌")

        placeholder.markdown(text)

        with citation_slot:
            render_citations(response.citations)

    memory.add("user", question)
    memory.add("assistant", response.answer, response.citations)


def render_chat(chain: RagChain) -> None:
    memory: ConversationMemory = st.session_state.memory

    st.markdown('<p class="annotation">Retrieval-augmented · grounded in the document repository</p>',
                unsafe_allow_html=True)
    st.markdown("# Real Estate AI Assistant")
    st.markdown('<div class="hero-rule"></div>', unsafe_allow_html=True)

    if memory.is_empty:
        st.markdown("Ask about a project, a payment plan, a policy — or start with one of these.")
        columns = st.columns(2)
        for index, starter in enumerate(STARTERS):
            with columns[index % 2]:
                if st.button(starter, key=f"starter-{index}", use_container_width=True):
                    st.session_state.pending_question = starter
                    st.rerun()

    render_transcript(memory)

    pending = st.session_state.pending_question
    if pending:
        st.session_state.pending_question = None
        answer_question(chain, pending)
        st.rerun()

    if question := st.chat_input("Ask about a project, payment plan, RERA filing or policy…"):
        answer_question(chain, question)
        st.rerun()


# ---------------------------------------------------------------------------
def main() -> None:
    ensure_state()
    inject_theme(st.session_state.theme)

    if not auth.is_authenticated():
        render_login()
        return

    try:
        with st.spinner("Preparing the knowledge base — this takes a moment on first run…"):
            chain, stats = load_chain()
    except Exception as error:  # noqa: BLE001 - show the operator what broke
        log.exception("Startup failed")
        st.error(f"The knowledge base could not be loaded: {error}")
        st.stop()

    render_sidebar(stats)
    render_chat(chain)


if __name__ == "__main__":
    main()
