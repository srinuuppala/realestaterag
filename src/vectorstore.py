"""FAISS index lifecycle: build, persist, load, invalidate.

The index is written to disk with a fingerprint of the corpus. On boot we
rebuild only if a document was added, removed or edited — otherwise a cold start
would re-embed 92 files every time.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import get_settings
from src.embedding import get_embeddings
from src.loader import build_corpus, discover
from src.utils import fingerprint, get_logger, timed

log = get_logger(__name__)

FINGERPRINT_FILE = "corpus.fingerprint"


def _fingerprint_path(index_dir: Path) -> Path:
    return index_dir / FINGERPRINT_FILE


def _is_stale(index_dir: Path, current: str) -> bool:
    marker = _fingerprint_path(index_dir)
    if not (index_dir / "index.faiss").exists() or not marker.exists():
        return True
    return marker.read_text(encoding="utf-8").strip() != current


@timed("build_index")
def build_index(chunks: list[Document]) -> FAISS:
    if not chunks:
        raise ValueError("Cannot build an index from an empty corpus")

    log.info("Embedding %d chunks…", len(chunks))
    return FAISS.from_documents(chunks, get_embeddings())


def save_index(store: FAISS, index_dir: Path, corpus_fingerprint: str) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    _fingerprint_path(index_dir).write_text(corpus_fingerprint, encoding="utf-8")
    log.info("Index saved to %s", index_dir)


@timed("load_index")
def load_index(index_dir: Path) -> FAISS:
    # The index is produced by this application, never by a third party, so
    # deserialisation is safe here.
    return FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def get_vectorstore(force_rebuild: bool = False) -> tuple[FAISS, list[Document]]:
    """Returns a ready index and the chunks behind it (BM25 needs the raw chunks).

    Rebuilds only when the corpus changed or a rebuild is explicitly requested.
    """
    settings = get_settings()
    index_dir = settings.index_dir
    current = fingerprint(discover(settings.knowledge_base_dir))

    chunks = build_corpus(settings.knowledge_base_dir)

    if force_rebuild or _is_stale(index_dir, current):
        reason = "rebuild requested" if force_rebuild else "corpus changed"
        log.info("Building FAISS index (%s)", reason)
        store = build_index(chunks)
        save_index(store, index_dir, current)
        return store, chunks

    log.info("Reusing cached FAISS index at %s", index_dir)
    try:
        return load_index(index_dir), chunks
    except Exception as error:  # noqa: BLE001 - a corrupt index must not be fatal
        log.error("Could not load the cached index (%s) — rebuilding", error)
        store = build_index(chunks)
        save_index(store, index_dir, current)
        return store, chunks
