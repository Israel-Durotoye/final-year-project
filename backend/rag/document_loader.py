"""
document_loader.py — Soil Doctor RAG Pipeline: Document Ingestion Engine

Responsibility:
    Reads every PDF in the knowledge base directory, extracts text using
    a two-strategy pipeline (PyMuPDF for text pages, PaddleOCR for
    image/scanned pages), splits the result into optimal chunks, embeds
    them with the same SentenceTransformer used by the retrieval engine,
    and upserts them into ChromaDB.

    After ingestion, if a live RAGEngine instance is provided, calls
    engine.refresh_bm25_index() so the sparse retrieval leg is immediately
    aware of the new content.

Ingestion Pipeline (per page):
    ┌─────────────────────────────────────────────────────────────┐
    │  PDF Page                                                   │
    │      │                                                      │
    │      ▼                                                      │
    │  [Strategy Router]                                          │
    │      ├── PyMuPDF text extraction (always attempted first)   │
    │      │       └── chars_extracted ≥ OCR_TRIGGER_THRESHOLD?   │
    │      │               YES → use PyMuPDF text only            │
    │      │               NO  → also run OCR on embedded images  │
    │      │                                                      │
    │      └── PaddleOCR (lazy-loaded on first actual need)       │
    │              ├── Extract embedded images via fitz xref      │
    │              ├── Render full-page pixmap as OCR fallback    │
    │              │   (if no extractable images found)           │
    │              └── Filter results by confidence threshold     │
    │                                                             │
    │  Combined text → RecursiveCharacterTextSplitter             │
    │      └── Chunk records (text + source + page metadata)      │
    │                                                             │
    │  Batch embedding (SentenceTransformer, batch_size=32)       │
    │      └── ChromaDB upsert (idempotent via SHA-256 chunk IDs) │
    └─────────────────────────────────────────────────────────────┘

Idempotency:
    Chunk IDs are SHA-256 hashes of (filename + page_number + text).
    Re-running the loader on already-ingested PDFs is safe — ChromaDB's
    upsert will silently skip existing chunks with identical IDs.

Design Decision — No RAGEngine Required for CLI:
    The loader initialises its own lightweight SentenceTransformer instance
    (no reranker, no BM25). This means:
      • The full bge-reranker-large model (~1.3 GB) is NOT loaded during
        ingestion, which would be wasteful.
      • The CLI can be run before the FastAPI server has ever started.
      • If a live RAGEngine is passed in (programmatic use), the loader
        calls refresh_bm25_index() on it to sync the sparse index.

CLI Usage:
    cd soil_doctor
    python -m backend.rag.document_loader                          # default KB dir
    python -m backend.rag.document_loader --kb-dir /path/to/pdfs
    python -m backend.rag.document_loader --clear                  # wipe & re-ingest
    python -m backend.rag.document_loader --no-ocr                 # skip OCR (faster)
    python -m backend.rag.document_loader --batch-size 16          # smaller batches

Dependencies:
    pip install PyMuPDF sentence-transformers chromadb paddleocr paddlepaddle
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import chromadb
from chromadb.config import Settings

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if TYPE_CHECKING:
    # Import only for type hints — avoids loading heavy models at module import
    from backend.rag.rag_engine import RAGEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (mirrors rag_engine.py so both modules use the same ChromaDB store)
# ---------------------------------------------------------------------------
_BACKEND_DIR  = Path(__file__).parent.parent
_KB_DIR       = _BACKEND_DIR / "data" / "knowledge_base"
_CHROMA_DIR   = _BACKEND_DIR / "data" / "chroma_db"

_KB_DIR.mkdir(parents=True, exist_ok=True)
_CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants — must be kept in sync with rag_engine.py
# ---------------------------------------------------------------------------
COLLECTION_NAME  = "agronomic_knowledge"
EMBED_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

# Chunking parameters
# multi-qa-mpnet-base-dot-v1 max sequence length: 512 WordPiece tokens.
# At ~4.5 chars/token, 1000 chars ≈ 222 tokens — safely within budget.
# Overlap of 200 chars preserves ~1 sentence of cross-chunk context.
CHUNK_SIZE    = 1000   # characters
CHUNK_OVERLAP = 200    # characters

# OCR routing heuristic: if PyMuPDF extracts fewer than this many meaningful
# characters from a page, the page is considered image-dominant and OCR runs.
OCR_TRIGGER_THRESHOLD = 100   # chars

# Minimum PaddleOCR confidence to include a recognised text block.
# 0.70 is a reasonable balance between recall and noise rejection.
OCR_MIN_CONFIDENCE = 0.70

# Page-level pixmap DPI for full-page OCR fallback (higher = better but slower)
OCR_PIXMAP_DPI = 150

# Minimum chars a chunk must contain to be worth embedding
MIN_CHUNK_CHARS = 40

# ChromaDB upsert batch size (balance between memory and speed)
UPSERT_BATCH_SIZE = 128

# Embedding batch size (keep low to avoid OOM on CPU)
EMBED_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class ExtractedPage:
    """Text extracted from a single PDF page."""
    filename: str
    page_num: int           # 0-indexed
    text: str
    char_count: int
    extraction_method: str  # "pymupdf" | "paddleocr" | "combined" | "empty"
    ocr_image_count: int = 0
    ocr_fallback_used: bool = False  # True if full-page pixmap was OCR'd


@dataclass
class ChunkRecord:
    """A single text chunk ready to be embedded and stored."""
    chunk_id: str
    text: str
    source: str        # PDF filename (no path)
    page: int          # 0-indexed (matches rag_engine.py metadata)
    chunk_index: int   # position of this chunk within its source page
    char_count: int
    extraction_method: str


@dataclass
class IngestionStats:
    """Summary statistics for one ingest_directory() or ingest_pdf() run."""
    pdf_count: int = 0
    page_count: int = 0
    ocr_page_count: int = 0
    chunk_count: int = 0
    skipped_chunk_count: int = 0    # below MIN_CHUNK_CHARS
    upserted_count: int = 0
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"PDFs processed : {self.pdf_count}\n"
            f"Pages processed: {self.page_count} "
            f"({self.ocr_page_count} required OCR)\n"
            f"Chunks created : {self.chunk_count} "
            f"({self.skipped_chunk_count} too short, skipped)\n"
            f"Upserted to DB : {self.upserted_count}\n"
            f"Total time     : {self.elapsed_seconds:.1f}s"
        )


# ---------------------------------------------------------------------------
# Recursive Character Text Splitter
# ---------------------------------------------------------------------------

class RecursiveCharacterSplitter:
    """
    A zero-dependency implementation of LangChain's RecursiveCharacterTextSplitter.

    Splits text on a prioritised list of separators, trying each one in order
    until the resulting pieces are all below chunk_size. Small pieces are
    merged back together with overlap to produce the final chunks.

    Separator priority (agronomic PDF optimised):
        1.  Double newline (paragraph boundary)
        2.  Single newline + optional whitespace (line break)
        3.  Period + space (sentence boundary)
        4.  Semicolon / colon (clause boundary — common in tables)
        5.  Comma + space (phrase boundary)
        6.  Single space (word boundary — last resort before char split)
        7.  Empty string (character-level — avoids infinite loops)
    """

    _SEPARATORS = ["\n\n", "\n", ". ", "; ", ": ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        """
        Split `text` into chunks of at most `chunk_size` characters,
        with `chunk_overlap` characters of context carried into the next chunk.

        Returns:
            List of non-empty string chunks.
        """
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        return self._split_recursive(text, self._SEPARATORS)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using the first separator that produces
        pieces small enough to fit within chunk_size, then merge small
        pieces with overlap.
        """
        # Find the first separator that actually splits this text
        chosen_sep = ""
        remaining_seps = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                chosen_sep = sep
                remaining_seps = separators[i + 1:]
                break

        # Split text on the chosen separator
        if chosen_sep == "":
            # Character-level split — last resort
            raw_splits = list(text)
        else:
            raw_splits = text.split(chosen_sep)

        # Recursively split any pieces that are still too large
        good_splits: list[str] = []
        for piece in raw_splits:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                good_splits.append(piece)
            elif remaining_seps:
                good_splits.extend(self._split_recursive(piece, remaining_seps))
            else:
                # Absolute fallback: hard-cut at chunk_size
                for i in range(0, len(piece), self.chunk_size - self.chunk_overlap):
                    good_splits.append(piece[i: i + self.chunk_size])

        # Merge small pieces back together with overlap
        return self._merge_with_overlap(good_splits, chosen_sep)

    def _merge_with_overlap(self, splits: list[str], separator: str) -> list[str]:
        """
        Greedily merge split pieces into chunks up to chunk_size,
        then carry `chunk_overlap` characters into the next chunk.
        """
        chunks: list[str] = []
        current: list[str] = []
        current_len: int = 0
        sep_len = len(separator)

        for piece in splits:
            piece_len = len(piece)
            projected_len = current_len + (sep_len if current else 0) + piece_len

            if projected_len > self.chunk_size and current:
                # Flush current chunk
                chunks.append(separator.join(current))

                # Build overlap tail: keep removing from the front of `current`
                # until the remaining content is within chunk_overlap chars
                while current and current_len > self.chunk_overlap:
                    removed_len = len(current[0]) + (sep_len if len(current) > 1 else 0)
                    current.pop(0)
                    current_len -= removed_len

            current.append(piece)
            current_len = sum(len(p) for p in current) + sep_len * (len(current) - 1)

        if current:
            chunks.append(separator.join(current))

        return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# DocumentLoader
# ---------------------------------------------------------------------------

class DocumentLoader:
    """
    Stateful ingestion engine.

    Args:
        embed_model_name  : SentenceTransformer model identifier.
                            Must match EMBED_MODEL_NAME in rag_engine.py.
        chroma_dir        : Path to the ChromaDB persistent store.
                            Must match _CHROMA_DIR in rag_engine.py.
        chunk_size        : Maximum characters per chunk (default: 1000).
        chunk_overlap     : Overlap characters between chunks (default: 200).
        enable_ocr        : Whether to run PaddleOCR on image-heavy pages.
                            Set False for pure-text PDFs to skip the slow OCR
                            initialisation entirely.

    Typical usage (standalone CLI):
        loader = DocumentLoader()
        stats  = loader.ingest_directory(kb_dir)
        print(stats)

    Typical usage (programmatic, with live RAGEngine):
        loader = DocumentLoader()
        stats  = loader.ingest_pdf(pdf_path, rag_engine=engine)
    """

    def __init__(
        self,
        embed_model_name: str = EMBED_MODEL_NAME,
        chroma_dir: Path = _CHROMA_DIR,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        enable_ocr: bool = True,
    ) -> None:
        self._embed_model_name = embed_model_name
        self._chroma_dir = chroma_dir
        self._enable_ocr = enable_ocr

        self._splitter = RecursiveCharacterSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Heavy objects — loaded lazily on first use
        self._embedding_model = None   # SentenceTransformer
        self._ocr_engine      = None   # PaddleOCR (only if enable_ocr=True)
        self._collection      = None   # chromadb.Collection

        logger.info(
            "DocumentLoader created | model: %s | chunk: %d/%d | OCR: %s",
            embed_model_name, chunk_size, chunk_overlap,
            "enabled" if enable_ocr else "disabled",
        )

    # ------------------------------------------------------------------
    # Public: ingestion entrypoints
    # ------------------------------------------------------------------

    def ingest_directory(
        self,
        kb_dir: Path = _KB_DIR,
        rag_engine: "RAGEngine | None" = None,
        clear: bool = False,
    ) -> IngestionStats:
        """
        Ingest all PDF files found in `kb_dir` (non-recursive).

        Args:
            kb_dir      : Directory containing agronomic PDF files.
            rag_engine  : If provided, refresh_bm25_index() is called after
                          all PDFs are processed so the sparse retrieval leg
                          is immediately up to date.
            clear       : If True, the entire ChromaDB collection is deleted
                          and recreated before ingestion begins.
                          Use this to fully rebuild the knowledge base.

        Returns:
            IngestionStats summarising the run.
        """
        kb_dir = Path(kb_dir)
        if not kb_dir.exists():
            raise FileNotFoundError(
                f"Knowledge base directory not found: {kb_dir}\n"
                "Create it and place your agronomic PDF files there."
            )

        pdf_files = sorted(kb_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in %s", kb_dir)
            return IngestionStats()

        logger.info(
            "Starting directory ingestion | Dir: %s | PDFs found: %d | Clear: %s",
            kb_dir, len(pdf_files), clear,
        )

        # Initialise shared resources once for the whole batch
        self._init_embedding_model()
        self._init_chroma_collection(clear=clear)

        t_start = time.perf_counter()
        total_stats = IngestionStats()

        for pdf_path in pdf_files:
            logger.info("─" * 55)
            pdf_stats = self._process_single_pdf(pdf_path)
            # Accumulate
            total_stats.pdf_count         += pdf_stats.pdf_count
            total_stats.page_count        += pdf_stats.page_count
            total_stats.ocr_page_count    += pdf_stats.ocr_page_count
            total_stats.chunk_count       += pdf_stats.chunk_count
            total_stats.skipped_chunk_count += pdf_stats.skipped_chunk_count
            total_stats.upserted_count    += pdf_stats.upserted_count

        total_stats.elapsed_seconds = round(time.perf_counter() - t_start, 2)

        # Sync the BM25 sparse index if a live engine was provided
        if rag_engine is not None:
            logger.info("Refreshing BM25 index on live RAGEngine...")
            rag_engine.refresh_bm25_index()
            logger.info("BM25 index refreshed.")
        else:
            logger.info(
                "No live RAGEngine provided. BM25 index will be rebuilt "
                "automatically on next server startup."
            )

        logger.info("─" * 55)
        logger.info("INGESTION COMPLETE\n%s", total_stats)
        return total_stats

    def ingest_pdf(
        self,
        pdf_path: Path,
        rag_engine: "RAGEngine | None" = None,
    ) -> IngestionStats:
        """
        Ingest a single PDF file.

        Args:
            pdf_path   : Absolute or relative path to a PDF file.
            rag_engine : If provided, refresh_bm25_index() is called after
                         ingestion.

        Returns:
            IngestionStats for this single file.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {pdf_path}")

        self._init_embedding_model()
        self._init_chroma_collection(clear=False)

        t_start = time.perf_counter()
        stats = self._process_single_pdf(pdf_path)
        stats.elapsed_seconds = round(time.perf_counter() - t_start, 2)

        if rag_engine is not None:
            logger.info("Refreshing BM25 index on live RAGEngine...")
            rag_engine.refresh_bm25_index()

        logger.info("PDF ingestion complete: %s\n%s", pdf_path.name, stats)
        return stats

    def clear_collection(self) -> None:
        """
        Delete all documents from the ChromaDB collection.
        Use before a full re-ingest to avoid stale data.
        """
        self._init_chroma_collection(clear=True)
        logger.info("ChromaDB collection '%s' cleared.", COLLECTION_NAME)

    # ------------------------------------------------------------------
    # Private: high-level processing orchestration
    # ------------------------------------------------------------------

    def _process_single_pdf(self, pdf_path: Path) -> IngestionStats:
        """Extract, chunk, embed, and store one PDF. Returns per-file stats."""
        import fitz

        filename = pdf_path.name
        stats = IngestionStats(pdf_count=1)

        logger.info("Processing: %s", filename)

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.error("Failed to open %s: %s — skipping.", filename, exc)
            return stats

        all_chunks: list[ChunkRecord] = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # ── Extract text from this page ────────────────────────────
            extracted = self._extract_page(page, page_num, filename, doc)
            stats.page_count += 1
            if extracted.extraction_method in ("paddleocr", "combined"):
                stats.ocr_page_count += 1

            if not extracted.text.strip():
                logger.debug(
                    "  Page %d/%d: empty after extraction — skipping.",
                    page_num + 1, len(doc),
                )
                continue

            # ── Chunk the extracted text ───────────────────────────────
            page_chunks = self._chunk_extracted_page(extracted)
            stats.chunk_count += len(page_chunks)

            # Filter out chunks that are too short to be meaningful
            valid_chunks = [c for c in page_chunks if c.char_count >= MIN_CHUNK_CHARS]
            skipped = len(page_chunks) - len(valid_chunks)
            if skipped:
                logger.debug(
                    "  Page %d/%d: skipped %d chunk(s) below %d chars.",
                    page_num + 1, len(doc), skipped, MIN_CHUNK_CHARS,
                )
            stats.skipped_chunk_count += skipped
            all_chunks.extend(valid_chunks)

            logger.debug(
                "  Page %d/%d [%s]: %d chars → %d chunks.",
                page_num + 1, len(doc),
                extracted.extraction_method,
                extracted.char_count,
                len(valid_chunks),
            )

        doc.close()

        if not all_chunks:
            logger.warning("  No usable chunks from %s — no data stored.", filename)
            return stats

        # ── Embed and upsert all chunks from this PDF ──────────────────
        upserted = self._embed_and_store(all_chunks)
        stats.upserted_count = upserted

        logger.info(
            "  Done: %d pages | %d chunks | %d upserted to ChromaDB",
            stats.page_count, stats.chunk_count - stats.skipped_chunk_count, upserted,
        )
        return stats

    # ------------------------------------------------------------------
    # Private: text extraction
    # ------------------------------------------------------------------

    def _extract_page(
        self,
        page: Any,
        page_num: int,
        filename: str,
        doc: Any,
    ) -> ExtractedPage:
        """
        Extract all readable text from a single page.

        Strategy:
            1. Always attempt PyMuPDF text extraction first (fast, lossless).
            2. If extracted chars < OCR_TRIGGER_THRESHOLD AND the page has
               embedded images → run PaddleOCR on each extracted image.
            3. If OCR yields nothing AND the page visually has content
               (detected by checking the page's bounding box) → render the
               full page as a pixmap and OCR that as a fallback.
        """
        # ── Step 1: PyMuPDF text extraction ────────────────────────────
        raw_text = page.get_text("text")
        cleaned_text = self._clean_text(raw_text)
        pymupdf_chars = len(cleaned_text)

        # If PyMuPDF got enough text, we're done
        if pymupdf_chars >= OCR_TRIGGER_THRESHOLD or not self._enable_ocr:
            method = "pymupdf" if pymupdf_chars > 0 else "empty"
            return ExtractedPage(
                filename=filename,
                page_num=page_num,
                text=cleaned_text,
                char_count=pymupdf_chars,
                extraction_method=method,
            )

        # ── Step 2: PaddleOCR on embedded images ───────────────────────
        logger.debug(
            "  Page %d: PyMuPDF got %d chars (< threshold %d) — trying OCR.",
            page_num + 1, pymupdf_chars, OCR_TRIGGER_THRESHOLD,
        )

        self._init_ocr_engine()
        ocr_text_parts: list[str] = []
        ocr_image_count = 0
        ocr_fallback_used = False

        embedded_images = page.get_images(full=True)
        for img_info in embedded_images:
            xref = img_info[0]
            img_text = self._ocr_single_image_xref(doc, xref, page_num)
            if img_text:
                ocr_text_parts.append(img_text)
                ocr_image_count += 1

        # ── Step 3: Full-page pixmap fallback ──────────────────────────
        # If no embedded images produced text, render the whole page
        # as a raster image (handles pages that are scanned PDFs with
        # no extractable image objects).
        if not ocr_text_parts:
            pixmap_text = self._ocr_full_page_pixmap(page, page_num)
            if pixmap_text:
                ocr_text_parts.append(pixmap_text)
                ocr_fallback_used = True
                logger.debug(
                    "  Page %d: used full-page pixmap OCR fallback.",
                    page_num + 1,
                )

        # Combine PyMuPDF text (if any) with OCR results
        all_text_parts = []
        if cleaned_text:
            all_text_parts.append(cleaned_text)
        all_text_parts.extend(ocr_text_parts)
        combined = "\n".join(all_text_parts)

        if not combined.strip():
            return ExtractedPage(
                filename=filename, page_num=page_num,
                text="", char_count=0, extraction_method="empty",
            )

        method = (
            "combined" if (cleaned_text and ocr_text_parts)
            else "paddleocr" if ocr_text_parts
            else "pymupdf"
        )

        return ExtractedPage(
            filename=filename,
            page_num=page_num,
            text=combined,
            char_count=len(combined),
            extraction_method=method,
            ocr_image_count=ocr_image_count,
            ocr_fallback_used=ocr_fallback_used,
        )

    def _ocr_single_image_xref(
        self,
        doc: Any,
        xref: int,
        page_num: int,
    ) -> str:
        """
        Extract a single embedded image from the PDF by xref, convert it
        to a numpy array, and run PaddleOCR on it.

        Returns:
            Concatenated OCR text, or empty string on failure.
        """
        try:
            img_data = doc.extract_image(xref)
            img_bytes = img_data["image"]
        except Exception as exc:
            logger.debug("  Could not extract image xref %d: %s", xref, exc)
            return ""

        return self._run_paddleocr_on_bytes(img_bytes, context=f"page {page_num+1} xref {xref}")

    def _ocr_full_page_pixmap(self, page: Any, page_num: int) -> str:
        """
        Render the entire page as a PNG at OCR_PIXMAP_DPI and OCR it.
        Used when no individual image objects are extractable but the page
        clearly contains visual/scanned content.

        Returns:
            OCR text string, or empty string on failure.
        """
        try:
            import fitz
            mat = fitz.Matrix(OCR_PIXMAP_DPI / 72, OCR_PIXMAP_DPI / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img_bytes = pix.tobytes("png")
        except Exception as exc:
            logger.debug("  Failed to render page %d pixmap: %s", page_num + 1, exc)
            return ""

        return self._run_paddleocr_on_bytes(
            img_bytes, context=f"page {page_num+1} full-pixmap"
        )

    def _run_paddleocr_on_bytes(self, img_bytes: bytes, context: str = "") -> str:
        """
        Convert raw image bytes to a numpy array and run PaddleOCR.
        Filters results by OCR_MIN_CONFIDENCE.

        Returns:
            Concatenated text from all high-confidence OCR blocks.
        """
        try:
            import numpy as np
            from PIL import Image

            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_array = np.array(pil_img)
        except Exception as exc:
            logger.debug("  Image decode failed (%s): %s", context, exc)
            return ""

        try:
            ocr_result = self._ocr_engine.ocr(img_array, cls=True)
        except Exception as exc:
            logger.warning("  PaddleOCR failed on %s: %s", context, exc)
            return ""

        if not ocr_result or not ocr_result[0]:
            return ""

        # ocr_result structure: list of pages, each page is a list of:
        #   [bounding_box_coords, [text_string, confidence_score]]
        text_blocks: list[str] = []
        for line in ocr_result[0]:
            if line is None:
                continue
            try:
                text, confidence = line[1][0], line[1][1]
                if confidence >= OCR_MIN_CONFIDENCE and text.strip():
                    text_blocks.append(text.strip())
            except (IndexError, TypeError):
                continue

        return " ".join(text_blocks)

    # ------------------------------------------------------------------
    # Private: chunking
    # ------------------------------------------------------------------

    def _chunk_extracted_page(self, extracted: ExtractedPage) -> list[ChunkRecord]:
        """
        Split a single ExtractedPage's text into ChunkRecord objects.
        Each chunk inherits the source filename and page number.
        """
        raw_chunks = self._splitter.split(extracted.text)
        records: list[ChunkRecord] = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            records.append(ChunkRecord(
                chunk_id=self._build_chunk_id(
                    extracted.filename, extracted.page_num, idx, chunk_text
                ),
                text=chunk_text,
                source=extracted.filename,
                page=extracted.page_num,
                chunk_index=idx,
                char_count=len(chunk_text),
                extraction_method=extracted.extraction_method,
            ))
        return records

    # ------------------------------------------------------------------
    # Private: embedding & storage
    # ------------------------------------------------------------------

    def _embed_and_store(self, chunks: list[ChunkRecord]) -> int:
        """
        Embed all chunks in batches and upsert them into ChromaDB.

        Uses ChromaDB's `upsert` (not `add`) so that re-running the loader
        on the same PDFs is safe — existing chunks are silently overwritten
        with identical data, and no duplicates are created.

        Returns:
            Number of chunks successfully upserted.
        """
        if not chunks:
            return 0

        total_upserted = 0
        n_batches = (len(chunks) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE

        for batch_idx in range(n_batches):
            start = batch_idx * UPSERT_BATCH_SIZE
            end   = start + UPSERT_BATCH_SIZE
            batch = chunks[start:end]

            texts = [c.text for c in batch]

            # ── Embed ──────────────────────────────────────────────────
            t_embed = time.perf_counter()
            try:
                embeddings = self._embedding_model.encode(
                    texts,
                    batch_size=EMBED_BATCH_SIZE,
                    normalize_embeddings=True,   # Required for cosine similarity
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).tolist()
            except Exception as exc:
                logger.error(
                    "Embedding failed for batch %d/%d: %s — skipping batch.",
                    batch_idx + 1, n_batches, exc,
                )
                continue

            embed_time = time.perf_counter() - t_embed
            logger.debug(
                "  Batch %d/%d: embedded %d chunks in %.2fs",
                batch_idx + 1, n_batches, len(batch), embed_time,
            )

            # ── Upsert to ChromaDB ─────────────────────────────────────
            try:
                self._collection.upsert(
                    ids=[c.chunk_id for c in batch],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[
                        {
                            "source":             c.source,
                            "page":               c.page,
                            "chunk_index":        c.chunk_index,
                            "char_count":         c.char_count,
                            "extraction_method":  c.extraction_method,
                        }
                        for c in batch
                    ],
                )
                total_upserted += len(batch)
            except Exception as exc:
                logger.error(
                    "ChromaDB upsert failed for batch %d/%d: %s",
                    batch_idx + 1, n_batches, exc,
                )

        return total_upserted

    # ------------------------------------------------------------------
    # Private: lazy initialisation
    # ------------------------------------------------------------------

    def _init_embedding_model(self) -> None:
        """Load SentenceTransformer on first call. No-op on subsequent calls."""
        if self._embedding_model is not None:
            return
        logger.info("Loading embedding model: %s", self._embed_model_name)
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self._embed_model_name)
            logger.info("Embedding model loaded.")
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

    def _init_ocr_engine(self) -> None:
        """
        Lazy-load PaddleOCR on first call. Subsequent calls are no-ops.

        PaddleOCR is only imported and instantiated when a page actually
        needs OCR, keeping startup time low for text-only PDF collections.
        """
        if self._ocr_engine is not None:
            return
        if not self._enable_ocr:
            raise RuntimeError("OCR was disabled in DocumentLoader constructor.")

        logger.info(
            "Initialising PaddleOCR (first OCR request — this may take ~10s)..."
        )
        try:
            from paddleocr import PaddleOCR
            # lang='en' for agronomic English documents
            # use_gpu=False forces CPU mode (safe default for servers without CUDA)
            # show_log=False suppresses PaddlePaddle's verbose model download logs
            self._ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=False,
                show_log=False,
            )
            logger.info("PaddleOCR initialised.")
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed. "
                "Run: pip install paddlepaddle paddleocr"
            ) from exc

    def _init_chroma_collection(self, clear: bool = False) -> None:
        """Connect to ChromaDB and get/create the collection."""
        if self._collection is not None and not clear:
            return

        client = chromadb.PersistentClient(
            path=str(self._chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        if clear:
            logger.warning(
                "Clearing ChromaDB collection '%s'...", COLLECTION_NAME
            )
            try:
                client.delete_collection(COLLECTION_NAME)
                logger.info("Collection deleted.")
            except Exception:
                pass   # Collection may not exist yet — that's fine

        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        doc_count = self._collection.count()
        logger.info(
            "ChromaDB collection '%s' ready | Current chunks: %d",
            COLLECTION_NAME, doc_count,
        )

    # ------------------------------------------------------------------
    # Private: utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chunk_id(
        filename: str,
        page_num: int,
        chunk_idx: int,
        text: str,
    ) -> str:
        """
        Generate a deterministic, collision-resistant chunk ID using SHA-256.

        The ID encodes the source file, page, position, and text content.
        Identical text on the same page of the same file always produces
        the same ID → safe for idempotent upserts.

        Returns:
            First 32 hex characters of the SHA-256 digest (128-bit ID).
            This is sufficient entropy to avoid collisions in a knowledge
            base of <10 million chunks.
        """
        content = f"{filename}::page{page_num}::chunk{chunk_idx}::{text[:200]}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Normalise whitespace and remove common PDF extraction artefacts.

        Operations:
            1. Replace form-feed characters (page breaks) with spaces
            2. Collapse runs of whitespace (excluding newlines) to single space
            3. Normalise multiple consecutive newlines to double newline
            4. Remove zero-width and non-printable characters
            5. Strip leading/trailing whitespace
        """
        if not text:
            return ""
        # Remove non-printable characters except newlines and tabs
        text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", " ", text)
        # Replace form-feed with newline
        text = text.replace("\f", "\n")
        # Collapse horizontal whitespace runs (not newlines)
        text = re.sub(r"[ \t\r]+", " ", text)
        # Normalise 3+ consecutive newlines to double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Soil Doctor — Agronomic Knowledge Base Ingestion Tool\n"
            "Processes PDFs in the knowledge base directory and populates ChromaDB."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=_KB_DIR,
        help=f"Path to the directory containing agronomic PDF files. "
             f"Default: {_KB_DIR}",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        default=False,
        help="Delete the existing ChromaDB collection before ingesting. "
             "Use this to do a clean rebuild of the knowledge base.",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        default=False,
        help="Disable PaddleOCR. Useful for text-only PDFs or when "
             "PaddleOCR is not installed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBED_BATCH_SIZE,
        metavar="N",
        help=f"Embedding batch size. Lower values use less memory. "
             f"Default: {EMBED_BATCH_SIZE}",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Maximum characters per chunk. Default: {CHUNK_SIZE}",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Overlap characters between consecutive chunks. "
             f"Default: {CHUNK_OVERLAP}",
    )
    args = parser.parse_args()

    # ── Validate args ──────────────────────────────────────────────────────
    if not args.kb_dir.exists():
        print(f"\n[ERROR] Knowledge base directory not found: {args.kb_dir}")
        print(f"        Create it and place your PDF files there, then re-run.")
        sys.exit(1)

    pdf_count = len(list(args.kb_dir.glob("*.pdf")))
    if pdf_count == 0:
        print(f"\n[WARNING] No PDF files found in: {args.kb_dir}")
        print(f"          Place agronomic PDF files there and re-run.")
        sys.exit(0)

    # ── Print startup banner ───────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  SOIL DOCTOR — KNOWLEDGE BASE INGESTION")
    print("━" * 60)
    print(f"  Knowledge base  : {args.kb_dir}")
    print(f"  PDFs to process : {pdf_count}")
    print(f"  ChromaDB path   : {_CHROMA_DIR}")
    print(f"  Embedding model : {EMBED_MODEL_NAME}")
    print(f"  Chunk size      : {args.chunk_size} chars / {args.chunk_overlap} overlap")
    print(f"  OCR             : {'DISABLED (--no-ocr)' if args.no_ocr else 'ENABLED'}")
    print(f"  Clear existing  : {'YES — will delete current collection' if args.clear else 'No'}")
    print("━" * 60 + "\n")

    if args.clear:
        confirm = input(
            "[WARNING] --clear will permanently delete all existing chunks. "
            "Continue? [y/N] "
        ).strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

    # ── Run ingestion ──────────────────────────────────────────────────────
    loader = DocumentLoader(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        enable_ocr=not args.no_ocr,
    )

    try:
        stats = loader.ingest_directory(
            kb_dir=args.kb_dir,
            rag_engine=None,   # No live server during CLI run
            clear=args.clear,
        )
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Ingestion stopped by user. "
              "Partial data may have been written to ChromaDB.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[ERROR] Ingestion failed: {exc}")
        logger.exception("Unhandled error during ingestion")
        sys.exit(1)

    # ── Print final summary ────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("  INGESTION SUMMARY")
    print("━" * 60)
    for line in str(stats).splitlines():
        print(f"  {line}")
    print("━" * 60)
    print("\n  Knowledge base is ready.")
    print("  Start the API server with:")
    print("      uvicorn main:app --host 0.0.0.0 --port 8000\n")