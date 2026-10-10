"""Build a fresh Milvus collection from text and image-caption JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import Settings
from .retrieval import read_jsonl


def build_index(settings: Settings, inputs: list[Path], *, batch_size: int = 32, client: Any = None, embedder: Any = None) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = [record for path in inputs for record in read_jsonl(Path(path))]
    if not records:
        raise ValueError("No knowledge records were found")
    owns_client = client is None
    if owns_client:
        if "://" not in settings.milvus_uri:
            Path(settings.milvus_uri).expanduser().parent.mkdir(parents=True, exist_ok=True)
        from pymilvus import MilvusClient

        client = MilvusClient(uri=settings.milvus_uri)
    try:
        if client.has_collection(collection_name=settings.collection):
            raise ValueError(
                f"Collection '{settings.collection}' already exists. Choose a new --collection; existing data will not be overwritten."
            )
        if embedder is None:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(settings.embedding_model)
        dimension = int(embedder.get_sentence_embedding_dimension())
        client.create_collection(
            collection_name=settings.collection,
            dimension=dimension,
            metric_type="COSINE",
            auto_id=True,
            enable_dynamic_field=True,
        )
        inserted = 0
        for offset in range(0, len(records), batch_size):
            batch = records[offset:offset + batch_size]
            embeddings = embedder.encode(
                [record["content"] for record in batch],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding result count does not match the input count")
            if any(len(vector) != dimension for vector in vectors):
                raise RuntimeError("Embedding vector dimension does not match the collection dimension")
            payload = [{**record, "vector": vector} for record, vector in zip(batch, vectors)]
            client.insert(collection_name=settings.collection, data=payload)
            inserted += len(payload)
        client.flush(collection_name=settings.collection)
        return {"collection": settings.collection, "records": inserted, "dimension": dimension, "embedding_model": settings.embedding_model}
    finally:
        if owns_client:
            client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="JSONL file; repeat for multiple inputs")
    parser.add_argument("--collection", help="New collection name (default: MEDRAG_COLLECTION)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.collection:
        from dataclasses import replace

        settings = replace(settings, collection=args.collection)
    summary = build_index(settings, args.input, batch_size=args.batch_size)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
