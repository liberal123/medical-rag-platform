"""Small offline demo retriever and lazily constructed Milvus/model backends."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from threading import RLock
from typing import Any

from .config import Settings


REFERENCE_FIELDS = ["content", "source_file", "page", "source", "image_path", "caption"]


def normalize_record(record: dict) -> dict:
    content = record.get("content", record.get("text", ""))
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Each knowledge record needs a non-empty 'content' or 'text' field")
    raw_page = record.get("page", 0)
    try:
        page = int(raw_page or 0)
    except (ValueError, TypeError) as exc:
        raise ValueError("Knowledge-record page must be an integer") from exc
    return {
        "content": content.strip(),
        "source_file": str(record.get("source_file") or ""),
        "page": page,
        "source": str(record.get("source") or ""),
        "image_path": str(record.get("image_path") or ""),
        "caption": str(record.get("caption") or ""),
    }


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record must be a JSON object")
                records.append(normalize_record(value))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid knowledge record at {path.name}:{line_number}: {exc}") from exc
    return records


def _terms(text: str) -> Counter:
    """Chinese character bigrams plus Latin words; no learned embeddings."""
    result: Counter = Counter()
    for token in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) == 1:
                result[token] += 1
            else:
                result.update(token[i:i + 2] for i in range(len(token) - 1))
        else:
            result[token] += 1
    return result


class DemoRetriever:
    backend_name = "lexical_demo"

    def __init__(self, knowledge_path: Path) -> None:
        self.records = read_jsonl(knowledge_path)
        self._vectors = [_terms(ref["content"] + " " + ref["caption"]) for ref in self.records]

    def search(self, question: str, top_k: int) -> list[dict]:
        query = _terms(question)
        query_norm = math.sqrt(sum(count * count for count in query.values()))
        if not query_norm:
            return []
        hits = []
        for record, vector in zip(self.records, self._vectors):
            dot = sum(count * vector.get(term, 0) for term, count in query.items())
            norm = math.sqrt(sum(count * count for count in vector.values()))
            if dot and norm:
                hits.append({**record, "score": dot / (query_norm * norm)})
        return sorted(hits, key=lambda ref: ref["score"], reverse=True)[:top_k]


class MilvusRetriever:
    backend_name = "milvus_dense"

    def __init__(self, settings: Settings, *, client: Any = None, embedder: Any = None) -> None:
        self.settings = settings
        self._client = client
        self._embedder = embedder
        self._lock = RLock()
        self._owns_client = client is None

    def _resources(self) -> tuple[Any, Any]:
        with self._lock:
            if self._client is None:
                if "://" not in self.settings.milvus_uri and not Path(self.settings.milvus_uri).is_file():
                    raise RuntimeError("The local Milvus index is missing. Build the configured knowledge index first.")
                from pymilvus import MilvusClient

                self._client = MilvusClient(uri=self.settings.milvus_uri)
            if not self._client.has_collection(collection_name=self.settings.collection):
                raise RuntimeError("Milvus collection is missing. Build the configured knowledge index first.")
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.settings.embedding_model)
            return self._client, self._embedder

    def search(self, question: str, top_k: int) -> list[dict]:
        client, embedder = self._resources()
        query = embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)
        data = query.tolist() if hasattr(query, "tolist") else query
        result = client.search(
            collection_name=self.settings.collection,
            data=data,
            limit=top_k,
            output_fields=REFERENCE_FIELDS,
            search_params={"metric_type": "COSINE", "params": {}},
        )
        references = []
        for hit in result[0] if result else []:
            entity = hit.get("entity", {})
            ref = normalize_record(entity)
            ref["score"] = float(hit.get("distance", 0.0))
            references.append(ref)
        return references

    def close(self) -> None:
        with self._lock:
            if self._owns_client and self._client is not None:
                self._client.close()
                self._client = None


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, model: Any = None) -> None:
        self.model_name = model_name
        self._model = model
        self._lock = RLock()

    def rerank(self, question: str, references: list[dict], top_k: int) -> list[dict]:
        if not references:
            return []
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            scores = self._model.predict(
                [(question, ref["content"]) for ref in references],
                show_progress_bar=False,
            )
        ranked = [{**ref, "rerank_score": float(score)} for ref, score in zip(references, scores)]
        return sorted(ranked, key=lambda ref: ref["rerank_score"], reverse=True)[:top_k]
