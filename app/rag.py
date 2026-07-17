"""LabBot RAG — chunk, index, and query the policy documents.

Two chunking strategies are implemented on purpose:

  naive_chunk(text)       — splits every N characters, ignoring structure.
  structure_aware_chunk() — splits along markdown headings/sections first,
                             and only falls back to size-based splitting
                             within an over-long section.

Build the index with either strategy (see scripts/build_index.py) and
compare what a boundary-spanning question retrieves under each. That
comparison is the point of Part 1 — see coursework.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from . import config

_EMBED_FN = embedding_functions.DefaultEmbeddingFunction()  # local, no API key needed
_COLLECTION_NAME = "labbot_policies"


@dataclass
class Chunk:
    text: str
    source: str      # filename, e.g. "checkout_policy.md"
    heading: str      # nearest section heading, "" if none (naive strategy)
    chunk_id: str


# ---------------------------------------------------------------------------
# Strategy 1: naive fixed-size chunking (the baseline to compare against)
# ---------------------------------------------------------------------------
def naive_chunk(text: str, source: str, size: int = 400, overlap: int = 50) -> list[Chunk]:
    """Split every `size` characters with a little overlap. Ignores headings,
    paragraph breaks, everything. This is what you get if you don't think
    about chunking at all."""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        piece = text[start : start + size]
        chunks.append(Chunk(text=piece.strip(), source=source, heading="", chunk_id=f"{source}-naive-{idx}"))
        start += size - overlap
        idx += 1
    return [c for c in chunks if c.text]


# ---------------------------------------------------------------------------
# Strategy 2: structure-aware chunking
# ---------------------------------------------------------------------------
def structure_aware_chunk(text: str, source: str, max_size: int = 800) -> list[Chunk]:
    """Split on markdown `##` section headings first, so a chunk never mixes
    two sections. If a single section is still too long, fall back to
    splitting on paragraph breaks within that section (never mid-sentence
    if we can help it). The section heading is prepended to every chunk
    from that section, so the chunk is self-contained even out of context.
    """
    # Split into (heading, body) pairs on "## " headings. Keep the doc title
    # (first "# ") out of the loop, treat everything under it as the intro.
    sections: list[tuple[str, str]] = []
    current_heading = "Introduction"
    current_body: list[str] = []

    for line in text.splitlines():
        m = re.match(r"^##\s+(.*)", line)
        if m:
            if current_body:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = m.group(1).strip()
            current_body = []
        elif line.startswith("# "):
            continue  # doc title, not a section
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_heading, "\n".join(current_body).strip()))

    chunks: list[Chunk] = []
    for i, (heading, body) in enumerate(sections):
        body = body.strip()
        if not body:
            continue
        labeled = f"## {heading}\n{body}"
        if len(labeled) <= max_size:
            chunks.append(Chunk(text=labeled, source=source, heading=heading, chunk_id=f"{source}-{i}"))
            continue
        # Section too long: split on paragraph breaks, re-prepending the
        # heading to each piece so retrieval never loses the section context.
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        buf = ""
        part = 0
        for p in paragraphs:
            candidate = (buf + "\n\n" + p).strip() if buf else p
            if len(candidate) > max_size and buf:
                chunks.append(Chunk(text=f"## {heading}\n{buf}", source=source, heading=heading, chunk_id=f"{source}-{i}-{part}"))
                buf = p
                part += 1
            else:
                buf = candidate
        if buf:
            chunks.append(Chunk(text=f"## {heading}\n{buf}", source=source, heading=heading, chunk_id=f"{source}-{i}-{part}"))
    return chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def _load_docs() -> list[tuple[str, str]]:
    """Returns [(filename, text), ...] for every .md file in docs/policies/."""
    docs_dir = config.DOCS_DIR
    return [(p.name, p.read_text()) for p in sorted(docs_dir.glob("*.md"))]


def build_index(strategy: str = "structured", persist_dir: Path | None = None) -> chromadb.Collection:
    """(Re)build the Chroma collection from docs/policies/ using the given
    strategy ("structured" or "naive"). Wipes any existing collection of the
    same name first, so this is safe to re-run whenever docs change.
    """
    client = chromadb.PersistentClient(path=str(persist_dir or config.CHROMA_DIR))
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=_COLLECTION_NAME, embedding_function=_EMBED_FN)

    chunk_fn = naive_chunk if strategy == "naive" else structure_aware_chunk
    all_chunks: list[Chunk] = []
    for filename, text in _load_docs():
        all_chunks.extend(chunk_fn(text, filename))

    if not all_chunks:
        raise RuntimeError(f"No documents found in {config.DOCS_DIR} — add .md files to docs/policies/")

    collection.add(
        ids=[c.chunk_id for c in all_chunks],
        documents=[c.text for c in all_chunks],
        metadatas=[{"source": c.source, "heading": c.heading} for c in all_chunks],
    )
    return collection


def get_collection(persist_dir: Path | None = None) -> chromadb.Collection:
    """Get the existing collection, building it (structure-aware, the
    default we ship with) if it doesn't exist yet."""
    client = chromadb.PersistentClient(path=str(persist_dir or config.CHROMA_DIR))
    try:
        return client.get_collection(name=_COLLECTION_NAME, embedding_function=_EMBED_FN)
    except Exception:
        return build_index(strategy="structured", persist_dir=persist_dir)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
def query(question: str, k: int = 4, persist_dir: Path | None = None) -> list[dict]:
    """Return the top-k chunks most relevant to `question`, each as
    {"text": ..., "source": ..., "heading": ...}."""
    collection = get_collection(persist_dir)
    results = collection.query(query_texts=[question], n_results=k)
    out = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(docs, metas):
        out.append({"text": doc, "source": meta.get("source", ""), "heading": meta.get("heading", "")})
    return out