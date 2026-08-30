"""Build the live agronomic ChromaDB from the canonical JSONL facts.

The source JSONL is expected to contain one complete agronomic fact per line.
Each record's ``knowledge_text`` is embedded with the same model used by the
runtime RAG engine, and ``fact_id`` is used unchanged as the Chroma document ID.

The importer intentionally does not read or copy ``rag_import/chroma_db``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from backend.rag.document_loader import (
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBED_MODEL_NAME,
    UPSERT_BATCH_SIZE,
    _resolve_model_device,
)


logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = _PROJECT_ROOT / "rag_import" / "agronomic_knowledge.jsonl"
DEFAULT_DATA_DIR = _PROJECT_ROOT / "backend" / "data" / "agronomic_knowledge"
LIVE_DATASET_NAME = "agronomic_knowledge.jsonl"

REQUIRED_FIELDS = (
    "fact_id",
    "knowledge_text",
    "topic",
    "crops",
    "parameter",
    "fact_type",
    "source_name",
    "page_start",
    "page_end",
    "confidence",
)


def _load_records(input_path: Path) -> list[dict[str, Any]]:
    """Load and validate every canonical fact before writing live data."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Canonical knowledge dataset not found: {input_path}")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            missing = [field for field in REQUIRED_FIELDS if field not in record]
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing required fields: {', '.join(missing)}"
                )

            fact_id = record["fact_id"]
            knowledge_text = record["knowledge_text"]
            if not isinstance(fact_id, str) or not fact_id.strip():
                raise ValueError(f"Line {line_number} has an invalid fact_id")
            if fact_id in seen_ids:
                raise ValueError(f"Duplicate fact_id on line {line_number}: {fact_id}")
            if not isinstance(knowledge_text, str) or not knowledge_text.strip():
                raise ValueError(f"Line {line_number} has empty knowledge_text")
            if not isinstance(record["crops"], list):
                raise ValueError(f"Line {line_number} has non-list crops metadata")

            seen_ids.add(fact_id)
            records.append(record)

    if not records:
        raise ValueError(f"Canonical knowledge dataset is empty: {input_path}")
    return records


def _metadata_for(record: dict[str, Any]) -> dict[str, str | int | float]:
    """Convert required fact metadata to Chroma-supported scalar values."""
    source_name = str(record["source_name"])
    page_start = int(record["page_start"])
    return {
        "topic": str(record["topic"]),
        # Chroma metadata values are scalar, so retain the full crop list as JSON.
        "crops": json.dumps(record["crops"], ensure_ascii=False, separators=(",", ":")),
        "parameter": str(record["parameter"]),
        "fact_type": str(record["fact_type"]),
        "source_name": source_name,
        "page_start": page_start,
        "page_end": int(record["page_end"]),
        "confidence": float(record["confidence"]),
        # Compatibility aliases consumed by the existing RetrievedChunk contract.
        "source": source_name,
        "page": page_start,
    }


def ingest_jsonl(
    input_path: Path = DEFAULT_INPUT_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
    batch_size: int = EMBED_BATCH_SIZE,
) -> int:
    """Rebuild the live collection and return its verified document count."""
    input_path = input_path.resolve()
    data_dir = data_dir.resolve()
    chroma_dir = data_dir / "chroma_db"
    data_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(input_path)
    logger.info("Validated %d canonical agronomic facts from %s", len(records), input_path)

    from sentence_transformers import SentenceTransformer

    device = _resolve_model_device()
    logger.info("Loading backend embedding model %s on %s", EMBED_MODEL_NAME, device)
    embedding_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Removed the previous live collection before rebuilding")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBED_MODEL_NAME,
            "chroma_version": chromadb.__version__,
            "dataset": LIVE_DATASET_NAME,
            "document_id_field": "fact_id",
            "document_text_field": "knowledge_text",
        },
    )

    for start in range(0, len(records), UPSERT_BATCH_SIZE):
        batch = records[start : start + UPSERT_BATCH_SIZE]
        documents = [record["knowledge_text"].strip() for record in batch]
        embeddings = embedding_model.encode(
            documents,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()
        collection.upsert(
            ids=[record["fact_id"] for record in batch],
            documents=documents,
            embeddings=embeddings,
            metadatas=[_metadata_for(record) for record in batch],
        )
        logger.info("Ingested %d/%d facts", min(start + len(batch), len(records)), len(records))

    stored_count = collection.count()
    if stored_count != len(records):
        raise RuntimeError(
            f"Collection verification failed: expected {len(records)} facts, found {stored_count}"
        )

    # Keep a byte-for-byte runtime copy of the canonical input beside the live DB.
    live_dataset_path = data_dir / LIVE_DATASET_NAME
    if input_path != live_dataset_path:
        shutil.copy2(input_path, live_dataset_path)

    dataset_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    manifest = {
        "collection_name": COLLECTION_NAME,
        "document_count": stored_count,
        "embedding_model": EMBED_MODEL_NAME,
        "chroma_version": chromadb.__version__,
        "source_dataset": LIVE_DATASET_NAME,
        "source_sha256": dataset_sha256,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return stored_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the backend agronomic ChromaDB from canonical JSONL facts."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    stored_count = ingest_jsonl(args.input, args.data_dir, args.batch_size)
    print(
        f"Knowledge base rebuilt successfully: {stored_count} facts in "
        f"{args.data_dir.resolve() / 'chroma_db'}"
    )


if __name__ == "__main__":
    main()
