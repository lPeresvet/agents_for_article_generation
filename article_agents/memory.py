"""ChromaDB-backed long-term memory for the manager agent.

Stores final articles per topic and retrieves the most relevant past
artifacts as additional context for new manager decisions.
"""

from __future__ import annotations

import math
import os
import time
import uuid
from itertools import zip_longest

from article_agents.trace import log as _trace

_DEFAULT_COLLECTION = "article_agents_manager"
_DEFAULT_EMBED_MODEL = "nomic-embed-text"


def normalize_topic_key(topic: str) -> str:
    """Stable key for matching topics across runs (case/spacing insensitive)."""
    return topic.strip().casefold()


def _strip_stored_document_body(doc: str) -> str:
    """Return article text from stored document format Topic: ...\\n\\n<body>."""
    doc = doc.strip()
    if doc.startswith("Topic:"):
        parts = doc.split("\n\n", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return doc


def _topic_from_stored_document(doc: str) -> str | None:
    """Parse 'Topic: ...' from the first line of a stored document."""
    if not doc or not doc.strip():
        return None
    first = doc.strip().split("\n", 1)[0]
    if not first.startswith("Topic:"):
        return None
    return first[len("Topic:") :].strip()


def _safe_len(obj) -> int:
    """Length without boolean coercion (numpy arrays raise on ``if arr``)."""
    if obj is None:
        return 0
    try:
        return len(obj)
    except TypeError:
        return 0


def _embedding_to_floats(emb) -> list[float] | None:
    """Chroma may return embeddings as lists or numpy arrays."""
    if emb is None:
        return None
    try:
        flat = emb.ravel() if hasattr(emb, "ravel") else emb
        return [float(x) for x in flat]
    except (TypeError, ValueError):
        return None


def _cosine_similarity(a, b) -> float:
    """Cosine similarity; accepts lists or numpy arrays (no boolean checks on arrays)."""
    try:
        la, lb = len(a), len(b)
    except TypeError:
        return 0.0
    if la == 0 or lb == 0 or la != lb:
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def topic_matches_query(stored_topic: str, query_key: str) -> bool:
    """True if stored topic string is the same run as the user query.

    Uses exact normalized equality, then substring match (query ≥3 chars) so
    ``ipv8`` matches ``IPv8 protocol overview`` but not unrelated short tokens.
    """
    if not stored_topic or not query_key:
        return False
    s = normalize_topic_key(stored_topic)
    k = query_key
    if s == k:
        return True
    if len(k) < 3:
        return False
    if k in s:
        return True
    if len(s) >= 3 and s in k:
        return True
    return False


class ManagerMemory:
    """Thin wrapper over a Chroma HTTP collection + Ollama embeddings.

    All operations degrade gracefully: if the DB or embedder is unreachable,
    methods log and return empty results instead of raising.
    """

    def __init__(self) -> None:
        self.collection_name = os.environ.get("CHROMA_COLLECTION", _DEFAULT_COLLECTION)
        self._k = int(os.environ.get("MANAGER_MEMORY_K", "3"))
        self._enabled = os.environ.get("MANAGER_MEMORY_ENABLED", "1") not in {"0", "false", "False"}
        self._collection = None
        self._embedder = None

        if not self._enabled:
            _trace("ManagerMemory disabled via MANAGER_MEMORY_ENABLED")
            return

        host = os.environ.get("CHROMA_HOST", "chromadb")
        port = int(os.environ.get("CHROMA_PORT", "8000"))
        retries = int(os.environ.get("CHROMA_CONNECT_RETRIES", "10"))
        retry_delay = float(os.environ.get("CHROMA_CONNECT_DELAY", "1.5"))

        try:
            import chromadb

            last_exc: Exception | None = None
            for attempt in range(1, retries + 1):
                try:
                    client = chromadb.HttpClient(host=host, port=port)
                    self._collection = client.get_or_create_collection(self.collection_name)
                    _trace(
                        f"ManagerMemory connected to Chroma at {host}:{port} "
                        f"(collection={self.collection_name}) on attempt {attempt}"
                    )
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    _trace(f"ManagerMemory: Chroma not ready (attempt {attempt}/{retries}): {exc}")
                    time.sleep(retry_delay)
            if last_exc is not None and self._collection is None:
                _trace(f"ManagerMemory: giving up on Chroma ({last_exc}); memory disabled")
        except Exception as exc:  # noqa: BLE001
            _trace(f"ManagerMemory: chromadb import failed ({exc}); memory disabled")
            self._collection = None

        try:
            from langchain_ollama import OllamaEmbeddings

            embed_model = os.environ.get("OLLAMA_EMBED_MODEL", _DEFAULT_EMBED_MODEL)
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            self._embedder = OllamaEmbeddings(model=embed_model, base_url=base_url)
        except Exception as exc:  # noqa: BLE001
            _trace(f"ManagerMemory: embedder unavailable ({exc}); memory disabled")
            self._embedder = None

    @property
    def is_ready(self) -> bool:
        return self._collection is not None and self._embedder is not None

    @property
    def has_collection(self) -> bool:
        return self._collection is not None

    def find_existing_article(self, topic: str) -> str | None:
        """Return a previously stored final article for this topic.

        Matching order:
        1) Metadata ``topic_key`` (current schema)
        2) Metadata ``topic`` exact string (legacy rows)
        3) Scan stored documents (bounded) and match normalized ``topic`` / ``Topic:`` line
        4) Semantic top-k then verify ``Topic:`` line matches (legacy rows without metadata)

        Step 4 needs embeddings; steps 1–3 only need Chroma.
        """
        if not self._enabled or self._collection is None or not topic.strip():
            return None
        key = normalize_topic_key(topic)
        stripped = topic.strip()

        def _hit_from_get(res) -> str | None:
            docs = (res or {}).get("documents") or []
            if _safe_len(docs) == 0:
                return None
            body = _strip_stored_document_body(docs[0])
            return body if body else None

        # 1) topic_key filter (preferred)
        try:
            res = self._collection.get(where={"topic_key": {"$eq": key}}, include=["documents"])
            body = _hit_from_get(res)
            if body:
                _trace(f"ManagerMemory.find_existing_article: hit via topic_key={key!r}")
                return body
        except Exception as exc:  # noqa: BLE001
            _trace(f"ManagerMemory.find_existing_article (topic_key): {exc}")

        # 2) legacy: metadata "topic" exact match as originally stored
        try:
            res = self._collection.get(where={"topic": {"$eq": stripped}}, include=["documents"])
            body = _hit_from_get(res)
            if body:
                _trace("ManagerMemory.find_existing_article: hit via metadata topic (exact)")
                return body
        except Exception as exc:  # noqa: BLE001
            _trace(f"ManagerMemory.find_existing_article (topic exact): {exc}")

        # 3) bounded scan: match normalized metadata.topic or Topic: line in document
        scan_limit = int(os.environ.get("MEMORY_SCAN_MAX", "5000"))
        try:
            res = self._collection.get(limit=scan_limit, include=["documents", "metadatas"])
            docs = (res or {}).get("documents") or []
            metas = (res or {}).get("metadatas") or []
            for doc, meta in zip_longest(docs, metas, fillvalue=None):
                if not doc:
                    continue
                meta = meta or {}
                tk = meta.get("topic_key")
                if tk is not None and topic_matches_query(str(tk), key):
                    body = _strip_stored_document_body(doc)
                    if body:
                        _trace("ManagerMemory.find_existing_article: hit via scan (topic_key meta)")
                        return body
                tm = meta.get("topic")
                if tm is not None and topic_matches_query(str(tm), key):
                    body = _strip_stored_document_body(doc)
                    if body:
                        _trace("ManagerMemory.find_existing_article: hit via scan (topic meta)")
                        return body
                tline = _topic_from_stored_document(doc)
                if tline is not None and topic_matches_query(tline, key):
                    body = _strip_stored_document_body(doc)
                    if body:
                        _trace("ManagerMemory.find_existing_article: hit via scan (Topic: line)")
                        return body
        except Exception as exc:  # noqa: BLE001
            _trace(f"ManagerMemory.find_existing_article (scan): {exc}")

        # 4) semantic near-duplicates: same normalized topic in stored doc header
        if self._embedder is None:
            _trace("ManagerMemory.find_existing_article: miss (no embedder for semantic verify)")
            return None
        try:
            hits = self.retrieve(
                stripped,
                k=int(os.environ.get("MEMORY_CACHE_SEMANTIC_K", "5")),
                min_similarity=0.0,
            )
            for h in hits:
                tline = _topic_from_stored_document(h)
                if tline is not None and topic_matches_query(tline, key):
                    body = _strip_stored_document_body(h)
                    if body:
                        _trace("ManagerMemory.find_existing_article: hit via semantic + Topic: verify")
                        return body
            # Optional: trust top-1 semantic result (same session / similar wording)
            if os.environ.get("MEMORY_CACHE_SEMANTIC_FIRST", "0") in {"1", "true", "True"} and _safe_len(hits) > 0:
                body = _strip_stored_document_body(hits[0])
                if body:
                    _trace("ManagerMemory.find_existing_article: hit via MEMORY_CACHE_SEMANTIC_FIRST (top-1)")
                    return body
        except Exception as exc:  # noqa: BLE001
            _trace(f"ManagerMemory.find_existing_article (semantic): {exc}")

        _trace(f"ManagerMemory.find_existing_article: miss for topic_key={key!r}")
        return None

    def retrieve(
        self,
        topic: str,
        k: int | None = None,
        *,
        min_similarity: float | None = None,
    ) -> list[str]:
        """Semantic neighbors for manager RAG, filtered by **cosine similarity** to the query.

        Only documents with similarity ≥ ``min_similarity`` are returned (default from
        ``MANAGER_MEMORY_MIN_SIMILARITY``, typically ~0.72). Pass ``min_similarity=0.0``
        to disable filtering (e.g. internal use in ``find_existing_article``).

        Chroma returns top-``n_results`` candidates; we over-fetch then filter so weak
        neighbors do not appear when nothing is truly close.
        """
        if not self.is_ready or not topic.strip():
            return []
        out_k = k if k is not None else self._k
        if min_similarity is None:
            raw = os.environ.get("MANAGER_MEMORY_MIN_SIMILARITY", "0.72").strip()
            try:
                min_sim = float(raw)
            except ValueError:
                min_sim = 0.72
        else:
            min_sim = float(min_similarity)

        candidates = int(os.environ.get("MANAGER_MEMORY_CANDIDATES_K", str(max(out_k * 4, 20))))
        candidates = max(candidates, out_k)

        try:
            qv = self._embedder.embed_query(topic)
            res = self._collection.query(
                query_embeddings=[qv],
                n_results=candidates,
                include=["documents", "embeddings"],
            )
        except Exception as exc:  # noqa: BLE001
            _trace(f"ManagerMemory.retrieve failed: {exc}")
            return []

        docs_field = (res or {}).get("documents") or [[]]
        emb_field = (res or {}).get("embeddings") or [[]]
        first_docs = docs_field[0] if _safe_len(docs_field) > 0 else []
        first_embs = emb_field[0] if _safe_len(emb_field) > 0 else []

        scored: list[tuple[float, str]] = []
        if _safe_len(first_embs) > 0 and _safe_len(first_embs) == _safe_len(first_docs):
            for doc, emb in zip_longest(first_docs, first_embs, fillvalue=None):
                if not isinstance(doc, str) or not doc.strip():
                    continue
                emb_f = _embedding_to_floats(emb)
                if emb_f is None:
                    continue
                sim = _cosine_similarity(qv, emb_f)
                if sim >= min_sim:
                    scored.append((sim, doc))
        else:
            _trace(
                "ManagerMemory.retrieve: no embeddings in query response; "
                "falling back to distance-only (weaker filter). "
                "Ensure Chroma returns embeddings (include=embeddings)."
            )
            try:
                res2 = self._collection.query(
                    query_embeddings=[qv],
                    n_results=candidates,
                    include=["documents", "distances"],
                )
                df = (res2 or {}).get("documents") or [[]]
                distf = (res2 or {}).get("distances") or [[]]
                d0 = df[0] if _safe_len(df) > 0 else []
                s0 = distf[0] if _safe_len(distf) > 0 else []
                max_d_raw = os.environ.get("MANAGER_MEMORY_MAX_DISTANCE", "0.35").strip()
                try:
                    max_d = float(max_d_raw)
                except ValueError:
                    max_d = 0.35
                for doc, dist in zip_longest(d0, s0, fillvalue=None):
                    if not isinstance(doc, str) or not doc.strip():
                        continue
                    if dist is None or float(dist) > max_d:
                        continue
                    scored.append((1.0 - float(dist), doc))
            except Exception as exc:  # noqa: BLE001
                _trace(f"ManagerMemory.retrieve fallback failed: {exc}")
                return []

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in scored[:out_k]]

        if not results and _safe_len(first_docs) > 0:
            _trace(
                f"ManagerMemory.retrieve: no chunks above similarity {min_sim:g} for topic {topic[:60]!r} "
                f"(Chroma returned {len(first_docs)} neighbor(s), all below threshold). "
                f"Lower MANAGER_MEMORY_MIN_SIMILARITY for more recall, or raise it to require closer matches."
            )
        else:
            _trace(
                f"ManagerMemory.retrieve: {len(results)} doc(s) for topic {topic[:60]!r} "
                f"with cosine similarity ≥ {min_sim:g} (candidates={candidates}, output_k={out_k})."
            )
        return results

    def add(self, topic: str, content: str) -> None:
        if not self.is_ready or not content.strip():
            return
        try:
            tkey = normalize_topic_key(topic)
            doc = f"Topic: {topic.strip()}\n\n{content.strip()}"
            vec = self._embedder.embed_documents([doc])[0]
            self._collection.add(
                ids=[str(uuid.uuid4())],
                documents=[doc],
                embeddings=[vec],
                metadatas=[
                    {
                        "topic": topic.strip()[:512],
                        "topic_key": tkey[:512],
                    }
                ],
            )
            _trace(f"ManagerMemory.add: stored memory for topic '{topic[:60]}'")
        except Exception as exc:  # noqa: BLE001
            _trace(f"ManagerMemory.add failed: {exc}")


def format_memory_context(snippets: list[str], max_chars: int = 1500) -> str:
    """Render retrieved memory snippets as a compact section for the manager prompt."""
    if not snippets:
        return ""
    parts: list[str] = []
    for i, s in enumerate(snippets, 1):
        body = s.strip()
        if len(body) > max_chars:
            body = body[:max_chars] + "…"
        parts.append(f"[Memory {i}]\n{body}")
    return "\n\n".join(parts)
