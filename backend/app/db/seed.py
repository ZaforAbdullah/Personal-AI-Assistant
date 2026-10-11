"""Load the markdown knowledge base, chunk it, embed it, store it.

Idempotent by content, not by count: the store is compared with the markdown (source, text and
embedding model), so an edited document or a switched embedding model is picked up and an
unchanged store costs one SELECT. The compare, re-embed and swap are one repository call
(`sync`), atomic and serialised across replicas.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.db.repositories import KnowledgeRepository
from app.llm.embeddings import Embedder

log = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"
_MAX_CHUNK_CHARS = 1200


@dataclass(slots=True)
class RawChunk:
    source: str
    content: str


def chunk_markdown(name: str, markdown: str) -> list[RawChunk]:
    """Split on `##` headings; split oversized sections on blank lines. Each chunk keeps its
    document title and heading so it makes sense on its own when retrieved."""
    title_match = re.match(r"^#\s+(.+)$", markdown.strip(), re.MULTILINE)
    title = title_match.group(1).strip() if title_match else name
    body = markdown[title_match.end() :] if title_match else markdown

    sections = re.split(r"^##\s+", body, flags=re.MULTILINE)
    chunks: list[RawChunk] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        heading, _, rest = section.partition("\n")
        rest = rest.strip()
        if not rest:  # preamble text without a heading
            heading, rest = "", heading
        source = f"{name}#{heading}" if heading else name
        prefix = f"{title} — {heading}\n" if heading else f"{title}\n"
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", rest) if p.strip()]
        buf = ""
        for para in paragraphs:
            if buf and len(buf) + len(para) > _MAX_CHUNK_CHARS:
                chunks.append(RawChunk(source, prefix + buf))
                buf = ""
            buf = f"{buf}\n\n{para}" if buf else para
        if buf:
            chunks.append(RawChunk(source, prefix + buf))
    return chunks


def load_knowledge(directory: Path = KNOWLEDGE_DIR) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_markdown(path.stem, path.read_text(encoding="utf-8")))
    return chunks


async def seed_knowledge(
    repo: KnowledgeRepository, embedder: Embedder, directory: Path = KNOWLEDGE_DIR
) -> int:
    """Returns the number of chunks embedded (0 when already up to date)."""
    raw = load_knowledge(directory)
    if not raw:
        log.warning("no knowledge documents found in %s", directory)
        return 0

    async def embed(texts: list[str]) -> list[list[float]]:
        return await embedder.embed(texts, kind="document")

    embedded = await repo.sync(embedder.model, [(c.source, c.content) for c in raw], embed)
    if embedded:
        log.info("seeded knowledge base: %d chunks with %s", embedded, embedder.model)
    else:
        log.info("knowledge base up to date (%d chunks, %s)", len(raw), embedder.model)
    return embedded
