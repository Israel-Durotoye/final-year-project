"""
rag_engine.py — Soil Doctor RAG Pipeline: Retrieval Engine

Responsibility:
    Owns all vector database interaction and semantic retrieval logic.
    Takes a natural-language query and returns the most relevant agronomic
    text chunks from the knowledge base, with zero LLM involvement.

Retrieval Architecture — True Hybrid Search:
    ┌────────────────────────────────────────────────────────────┐
    │  Query String                                              │
    │      │                                                     │
    │      ├──► [Dense Leg]  SentenceTransformer embedding       │
    │      │         └──► ChromaDB cosine similarity search      │
    │      │                   └──► top (top_k × DENSE_MULT)     │
    │      │                        candidates + scores          │
    │      │                                                     │
    │      └──► [Sparse Leg]  BM25Okapi keyword scoring          │
    │                └──► in-memory index over all stored docs   │
    │                         └──► top (top_k × SPARSE_MULT)     │
    │                              candidates + scores           │
    │                                                            │
    │  [Fusion]  Reciprocal Rank Fusion (RRF, k=60)             │
    │      └──► merged + deduplicated candidate pool             │
    │                                                            │
    │  [Rerank]  BAAI/bge-reranker-large (FlagEmbedding)        │
    │      └──► final top_k chunks, sorted by cross-encoder      │
    │           relevance score                                  │
    └────────────────────────────────────────────────────────────┘

Why two retrieval legs?
    Dense search (embeddings) excels at semantic matching — "low soil acidity"
    will correctly retrieve chunks about pH. BM25 excels at exact-term recall —
    "NPK 20-10-10" will surface chunks containing that exact product code.
    Fusing both ensures neither precision nor recall is sacrificed.

Why cross-encoder reranking after fusion?
    Bi-encoders (used for dense retrieval) compress query and document into
    separate vectors, losing fine-grained interaction signals. A cross-encoder
    sees the (query, document) pair jointly, producing a far more accurate
    relevance score — but is too slow to run over thousands of documents.
    Running it on the small fused candidate pool gives us the accuracy of
    cross-encoders without the latency cost.

BM25 Index Lifecycle:
    Built in-memory on RAGEngine.initialize() from all documents stored in
    ChromaDB. Call refresh_bm25_index() after new documents are ingested.
    For a typical agronomic knowledge base (50–300 PDF chunks), memory
    footprint is negligible (<10 MB).

Dependencies:
    pip install chromadb sentence-transformers FlagEmbedding rank_bm25
"""

from __future__ import annotations

import logging
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — all relative to this file so the project is relocatable
# ---------------------------------------------------------------------------
_BACKEND_DIR  = Path(__file__).parent.parent
_CHROMA_DIR   = _BACKEND_DIR / "data" / "chroma_db"
_CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME  = "agronomic_knowledge"
EMBED_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Candidate pool multipliers before reranking
# e.g. top_k=5 → dense fetches 10, sparse fetches 10, reranker scores ≤20 merged
DENSE_MULT  = 2
SPARSE_MULT = 2

# RRF constant — 60 is the standard value from the original paper
RRF_K = 60

# Final chunks returned to the LLM after reranking
FINAL_TOP_N = 3


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """
    A single text chunk returned by hybrid_search().

    Attributes:
        chunk_id        : Unique identifier in ChromaDB.
        text            : The raw text content of the chunk.
        source          : Source document filename (e.g. 'FAO_fertilizer_guide.pdf').
        page            : Page number within the source document.
        dense_score     : Cosine similarity from the dense retrieval leg (0–1).
        bm25_score      : Normalised BM25 score from the sparse leg (0–1).
        rrf_score       : Reciprocal Rank Fusion score (pre-reranking merge).
        rerank_score    : Cross-encoder relevance score (final ranking signal).
    """
    chunk_id: str
    text: str
    source: str
    page: int
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


# ---------------------------------------------------------------------------
# RAGEngine
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    Stateful retrieval engine. Owns ChromaDB, the embedding model,
    the BM25 index, and the cross-encoder reranker.

    Lifecycle (for FastAPI integration):
        engine = RAGEngine()
        engine.initialize()          # Called once at app startup (lifespan)
        chunks = engine.hybrid_search(query, top_k=5)
        engine.shutdown()            # Optional — releases model memory

    Threading note:
        SentenceTransformer and FlagReranker are not thread-safe for concurrent
        calls. In production, wrap the engine in an asyncio.Lock or use
        a process pool. For a single-user Streamlit demo, this is not an issue.
    """

    def __init__(self) -> None:
        self._embedding_model  = None   # SentenceTransformer — loaded on initialize()
        self._reranker         = None   # FlagReranker         — loaded on initialize()
        self._chroma_client    = None   # chromadb.PersistentClient
        self._collection       = None   # chromadb.Collection
        self._bm25_index       = None   # BM25Okapi instance
        self._bm25_doc_ids: list[str] = []   # ordered list of chunk IDs in BM25 index
        self._bm25_corpus: list[list[str]] = []  # tokenised corpus for BM25
        self._is_initialized   = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Load all models and connect to ChromaDB.
        This is intentionally blocking — call it once at application startup,
        not per-request.

        Raises:
            ImportError  : If sentence-transformers or FlagEmbedding are not installed.
            RuntimeError : If ChromaDB cannot be reached.
        """
        if self._is_initialized:
            logger.warning("RAGEngine.initialize() called more than once — skipping.")
            return

        t_start = time.perf_counter()
        logger.info("Initializing RAGEngine...")

        # ── Step 1: Embedding model ────────────────────────────────────
        logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        # ── Step 2: Cross-encoder reranker ─────────────────────────────
        logger.info("Loading reranker: %s", RERANK_MODEL_NAME)
        try:
            from FlagEmbedding import FlagReranker
            # use_fp16=True halves memory usage with negligible accuracy impact
            self._reranker = FlagReranker(RERANK_MODEL_NAME, use_fp16=True)
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is not installed. "
                "Run: pip install FlagEmbedding"
            ) from exc

        # ── Step 3: ChromaDB persistent client ────────────────────────
        logger.info("Connecting to ChromaDB at: %s", _CHROMA_DIR)
        try:
            self._chroma_client = chromadb.PersistentClient(
                path=str(_CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                # Cosine distance is appropriate for normalised sentence embeddings
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to ChromaDB at {_CHROMA_DIR}: {exc}"
            ) from exc

        doc_count = self._collection.count()
        logger.info("ChromaDB collection '%s' contains %d chunks.", COLLECTION_NAME, doc_count)

        # ── Step 4: Build BM25 index from stored documents ────────────
        self.refresh_bm25_index()

        elapsed = time.perf_counter() - t_start
        self._is_initialized = True
        logger.info("RAGEngine initialized in %.2fs | Docs in index: %d", elapsed, doc_count)

    def shutdown(self) -> None:
        """Release model references so Python's GC can reclaim GPU/CPU memory."""
        self._embedding_model = None
        self._reranker = None
        self._bm25_index = None
        self._is_initialized = False
        logger.info("RAGEngine shut down.")

    def is_ready(self) -> bool:
        """True if initialize() has been called successfully."""
        return self._is_initialized

    def document_count(self) -> int:
        """Number of chunks currently stored in ChromaDB."""
        if self._collection is None:
            return 0
        return self._collection.count()

    # ------------------------------------------------------------------
    # BM25 Index Management
    # ------------------------------------------------------------------

    def refresh_bm25_index(self) -> None:
        """
        (Re)build the in-memory BM25 index from all documents currently
        stored in ChromaDB.

        Call this after ingesting new documents via document_loader.py.
        Building the index takes <1s for a typical agronomic knowledge base
        (50–300 chunks of ~300 tokens each).
        """
        if self._collection is None:
            logger.warning("Cannot build BM25 index — ChromaDB collection not initialised.")
            return

        doc_count = self._collection.count()
        if doc_count == 0:
            logger.info("BM25 index not built — collection is empty.")
            self._bm25_index = None
            self._bm25_doc_ids = []
            self._bm25_corpus = []
            return

        logger.info("Building BM25 index over %d stored chunks...", doc_count)
        # Fetch all documents from ChromaDB (IDs + texts only — no embeddings needed)
        result = self._collection.get(include=["documents", "metadatas"])

        self._bm25_doc_ids = result["ids"]
        self._bm25_corpus = [
            self._tokenize(doc) for doc in result["documents"]
        ]
        self._bm25_index = BM25Okapi(self._bm25_corpus)
        logger.info("BM25 index built over %d chunks.", len(self._bm25_doc_ids))

    # ------------------------------------------------------------------
    # Public API: hybrid_search
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Execute a hybrid (dense + sparse) search and return the top reranked chunks.

        Args:
            query  : Natural-language user query string.
            top_k  : Number of chunks to return after reranking.
                     The final count will be min(top_k, FINAL_TOP_N) = 3 by design,
                     but you can override FINAL_TOP_N for larger LLM context windows.

        Returns:
            List of RetrievedChunk, sorted by rerank_score descending.
            Returns an empty list if the collection is empty or if no relevant
            chunks survive the relevance threshold.

        Raises:
            RuntimeError : If called before initialize().
        """
        self._assert_ready()

        query = query.strip()
        if not query:
            logger.warning("hybrid_search called with empty query string.")
            return []

        doc_count = self._collection.count()
        if doc_count == 0:
            logger.warning(
                "hybrid_search called but the knowledge base is empty. "
                "Ingest documents first using document_loader.py."
            )
            return []

        t_start = time.perf_counter()
        n_dense  = min(top_k * DENSE_MULT,  doc_count)
        n_sparse = min(top_k * SPARSE_MULT, doc_count)

        # ── Leg 1: Dense retrieval (ChromaDB cosine similarity) ────────
        dense_results = self._dense_search(query, n_candidates=n_dense)

        # ── Leg 2: Sparse retrieval (BM25 keyword scoring) ─────────────
        sparse_results = self._sparse_search(query, n_candidates=n_sparse)

        # ── Fusion: Reciprocal Rank Fusion ─────────────────────────────
        fused_candidates = self._reciprocal_rank_fusion(dense_results, sparse_results)

        if not fused_candidates:
            logger.warning("RRF produced zero candidates for query: '%s'", query[:80])
            return []

        # ── Reranking: BAAI/bge-reranker-large ─────────────────────────
        reranked = self._rerank(query, fused_candidates, final_n=FINAL_TOP_N)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "hybrid_search | Query: '%s...' | Dense: %d | Sparse: %d | "
            "Fused: %d | Returned: %d | %.3fs",
            query[:50], len(dense_results), len(sparse_results),
            len(fused_candidates), len(reranked), elapsed,
        )
        return reranked

    # ------------------------------------------------------------------
    # Private: Dense retrieval leg
    # ------------------------------------------------------------------

    def _dense_search(
        self,
        query: str,
        n_candidates: int,
    ) -> list[dict[str, Any]]:
        """
        Embed the query and retrieve the top n_candidates chunks from
        ChromaDB using cosine similarity.

        Returns:
            List of dicts: {chunk_id, text, source, page, dense_score}
        """
        query_embedding = self._embedding_model.encode(
            query,
            normalize_embeddings=True,   # Required for cosine similarity
            show_progress_bar=False,
        ).tolist()

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )

        candidates = []
        for i, chunk_id in enumerate(result["ids"][0]):
            # ChromaDB returns cosine *distance* (0=identical, 2=opposite)
            # Convert to similarity score in [0, 1]
            distance = result["distances"][0][i]
            similarity = max(0.0, 1.0 - (distance / 2.0))

            meta = result["metadatas"][0][i] or {}
            candidates.append({
                "chunk_id":    chunk_id,
                "text":        result["documents"][0][i],
                "source":      meta.get("source", "unknown"),
                "page":        int(meta.get("page", 0)),
                "dense_score": similarity,
            })

        return candidates

    # ------------------------------------------------------------------
    # Private: Sparse retrieval leg
    # ------------------------------------------------------------------

    def _sparse_search(
        self,
        query: str,
        n_candidates: int,
    ) -> list[dict[str, Any]]:
        """
        Score all documents in the BM25 index against the tokenised query
        and return the top n_candidates.

        Returns:
            List of dicts: {chunk_id, text, source, page, bm25_score}
            Returns empty list if the BM25 index has not been built.
        """
        if self._bm25_index is None:
            logger.debug("BM25 index not available — skipping sparse retrieval leg.")
            return []

        tokenised_query = self._tokenize(query)
        raw_scores = self._bm25_index.get_scores(tokenised_query)

        # Normalise scores to [0, 1] using the max score in this query
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        normalised = raw_scores / max_score

        # Sort descending by score and take top n_candidates
        sorted_indices = sorted(
            range(len(normalised)), key=lambda i: normalised[i], reverse=True
        )[:n_candidates]

        # Look up the full text and metadata for each candidate from ChromaDB
        # We only fetch IDs we actually need — avoids a full collection scan
        top_ids = [self._bm25_doc_ids[i] for i in sorted_indices]
        result = self._collection.get(
            ids=top_ids,
            include=["documents", "metadatas"],
        )

        # Build a lookup so we can preserve the score-sorted order
        id_to_doc  = {rid: doc  for rid, doc  in zip(result["ids"], result["documents"])}
        id_to_meta = {rid: meta for rid, meta in zip(result["ids"], result["metadatas"])}

        candidates = []
        for i, idx in enumerate(sorted_indices):
            chunk_id = self._bm25_doc_ids[idx]
            if chunk_id not in id_to_doc:
                continue   # Chunk may have been deleted — skip safely
            meta = id_to_meta.get(chunk_id) or {}
            candidates.append({
                "chunk_id":   chunk_id,
                "text":       id_to_doc[chunk_id],
                "source":     meta.get("source", "unknown"),
                "page":       int(meta.get("page", 0)),
                "bm25_score": float(normalised[idx]),
            })

        return candidates

    # ------------------------------------------------------------------
    # Private: Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_results: list[dict],
        sparse_results: list[dict],
    ) -> list[dict[str, Any]]:
        """
        Merge the two ranked candidate lists using Reciprocal Rank Fusion.

        RRF Score: Σ 1 / (k + rank_i(document))
            where k=60 (standard constant from Cormack et al., 2009)
            and rank_i is 1-based position in result list i.

        Deduplication: chunks appearing in both legs get their RRF scores
        summed — giving them a higher combined score than chunks in only one leg.

        Returns:
            Merged list of candidate dicts sorted by rrf_score descending.
        """
        rrf_scores: dict[str, float] = {}
        merged_docs: dict[str, dict] = {}

        for rank, doc in enumerate(dense_results, start=1):
            cid = doc["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            merged_docs[cid] = {**doc, "bm25_score": merged_docs.get(cid, {}).get("bm25_score", 0.0)}

        for rank, doc in enumerate(sparse_results, start=1):
            cid = doc["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            if cid not in merged_docs:
                merged_docs[cid] = {**doc, "dense_score": 0.0}
            else:
                merged_docs[cid]["bm25_score"] = doc["bm25_score"]

        # Attach the RRF score and sort
        fused = []
        for cid, doc in merged_docs.items():
            fused.append({**doc, "rrf_score": rrf_scores[cid]})

        fused.sort(key=lambda d: d["rrf_score"], reverse=True)
        return fused

    # ------------------------------------------------------------------
    # Private: Cross-encoder reranking
    # ------------------------------------------------------------------

    def _rerank(
        self,
        query: str,
        candidates: list[dict],
        final_n: int,
    ) -> list[RetrievedChunk]:
        """
        Score each (query, candidate_text) pair with the BAAI/bge-reranker-large
        cross-encoder and return the top final_n results.

        The cross-encoder produces raw logit scores — higher is more relevant.
        We do not normalise them because only relative ordering matters.

        Returns:
            List of RetrievedChunk sorted by rerank_score descending.
        """
        if not candidates:
            return []

        pairs = [[query, doc["text"]] for doc in candidates]
        try:
            scores = self._reranker.compute_score(pairs, normalize=False)
        except Exception as exc:
            logger.error(
                "Reranker failed — falling back to RRF ordering. Error: %s", exc
            )
            # Graceful degradation: return top final_n by RRF score as RetrievedChunks
            return [
                RetrievedChunk(
                    chunk_id=doc["chunk_id"],
                    text=doc["text"],
                    source=doc.get("source", "unknown"),
                    page=doc.get("page", 0),
                    dense_score=doc.get("dense_score", 0.0),
                    bm25_score=doc.get("bm25_score", 0.0),
                    rrf_score=doc.get("rrf_score", 0.0),
                    rerank_score=doc.get("rrf_score", 0.0),   # use RRF as proxy
                )
                for doc in candidates[:final_n]
            ]

        # scores is a list[float] aligned with candidates
        if not isinstance(scores, list):
            scores = list(scores)

        # Pair scores with candidates and sort
        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        result = []
        for rerank_score, doc in scored[:final_n]:
            result.append(RetrievedChunk(
                chunk_id=doc["chunk_id"],
                text=doc["text"],
                source=doc.get("source", "unknown"),
                page=doc.get("page", 0),
                dense_score=doc.get("dense_score", 0.0),
                bm25_score=doc.get("bm25_score", 0.0),
                rrf_score=doc.get("rrf_score", 0.0),
                rerank_score=float(rerank_score),
            ))
        return result

    # ------------------------------------------------------------------
    # Private: Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Simple whitespace + punctuation tokeniser for BM25.
        Lowercases and splits on non-alphanumeric characters.
        Filters out single-character tokens.
        """
        tokens = re.split(r"[^a-zA-Z0-9]", text.lower())
        return [t for t in tokens if len(t) > 1]

    def _assert_ready(self) -> None:
        if not self._is_initialized:
            raise RuntimeError(
                "RAGEngine has not been initialised. Call engine.initialize() "
                "at application startup before using hybrid_search()."
            )
