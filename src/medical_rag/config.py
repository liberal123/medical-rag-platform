"""Environment-based configuration without model loading or side effects."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    mode: str = "demo"
    data_dir: Path = field(default_factory=lambda: REPOSITORY_ROOT / "data")
    demo_knowledge_path: Path = field(
        default_factory=lambda: REPOSITORY_ROOT / "examples" / "demo_knowledge.jsonl"
    )
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    llm_base_url: str = "http://127.0.0.1:8001/v1"
    llm_model: str = "glm-4"
    llm_api_key: str = field(default="", repr=False)
    milvus_uri: str = ""
    collection: str = "dr_screening_knowledge"
    request_timeout: float = 90.0

    def __post_init__(self) -> None:
        if self.mode not in {"demo", "live"}:
            raise ValueError("MEDRAG_MODE must be 'demo' or 'live'")
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser().resolve())
        object.__setattr__(self, "demo_knowledge_path", Path(self.demo_knowledge_path).expanduser().resolve())
        if not self.milvus_uri:
            object.__setattr__(self, "milvus_uri", str(self.data_dir / "index" / "medical.db"))
        if not self.collection.strip():
            raise ValueError("MEDRAG_COLLECTION cannot be empty")
        if self.request_timeout <= 0:
            raise ValueError("MEDRAG_REQUEST_TIMEOUT must be positive")

    @property
    def image_dir(self) -> Path:
        return self.data_dir / "images"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        return cls(
            mode=env.get("MEDRAG_MODE", "demo").strip().lower(),
            data_dir=Path(env.get("MEDRAG_DATA_DIR", str(REPOSITORY_ROOT / "data"))),
            demo_knowledge_path=Path(env.get("MEDRAG_DEMO_KNOWLEDGE", str(REPOSITORY_ROOT / "examples" / "demo_knowledge.jsonl"))),
            embedding_model=env.get("MEDRAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            reranker_model=env.get("MEDRAG_RERANKER_MODEL", "BAAI/bge-reranker-base"),
            llm_base_url=env.get("MEDRAG_LLM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/"),
            llm_model=env.get("MEDRAG_LLM_MODEL", "glm-4"),
            llm_api_key=env.get("MEDRAG_LLM_API_KEY", ""),
            milvus_uri=env.get("MEDRAG_MILVUS_URI", ""),
            collection=env.get("MEDRAG_COLLECTION", "dr_screening_knowledge"),
            request_timeout=float(env.get("MEDRAG_REQUEST_TIMEOUT", "90")),
        )
