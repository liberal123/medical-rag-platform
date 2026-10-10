"""Retrieval, optional reranking, and grounded response generation."""

from __future__ import annotations

from threading import RLock
from typing import Any

from .config import Settings
from .llm import VLLMClient
from .retrieval import CrossEncoderReranker, DemoRetriever, MilvusRetriever


class Pipeline:
    def __init__(self, settings: Settings, *, retriever: Any = None, reranker: Any = None, llm: Any = None) -> None:
        self.settings = settings
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm
        self._lock = RLock()

    def _get_retriever(self) -> Any:
        with self._lock:
            if self._retriever is None:
                self._retriever = (
                    DemoRetriever(self.settings.demo_knowledge_path)
                    if self.settings.mode == "demo"
                    else MilvusRetriever(self.settings)
                )
            return self._retriever

    def _get_reranker(self) -> Any:
        with self._lock:
            if self._reranker is None:
                self._reranker = CrossEncoderReranker(self.settings.reranker_model)
            return self._reranker

    def _get_llm(self) -> Any:
        with self._lock:
            if self._llm is None:
                self._llm = VLLMClient(self.settings)
            return self._llm

    def chat(
        self,
        question: str,
        retrieval_top_k: int = 10,
        final_top_k: int = 3,
        use_rerank: bool = True,
        chunk_chars: int = 400,
        max_tokens: int = 512,
    ) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        if not 1 <= final_top_k <= retrieval_top_k <= 100:
            raise ValueError("Expected 1 <= final_top_k <= retrieval_top_k <= 100")
        if not 50 <= chunk_chars <= 4000:
            raise ValueError("chunk_chars must be between 50 and 4000")
        if not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be between 1 and 4096")

        retriever = self._get_retriever()
        references = retriever.search(question, retrieval_top_k)
        rerank_applied = False
        if references and self.settings.mode == "live" and use_rerank:
            references = self._get_reranker().rerank(question, references, final_top_k)
            rerank_applied = True
        else:
            references = references[:final_top_k]
        # Return the exact evidence sent to generation so citations are auditable.
        references = [{**ref, "content": ref["content"][:chunk_chars]} for ref in references]
        generation_mode = "extractive_demo" if self.settings.mode == "demo" else "vllm"
        if not references:
            answer = "当前知识库中没有检索到可用资料，无法依据资料回答该问题。请补充相关资料或咨询专业医务人员。"
            generation_mode = "no_evidence"
        elif self.settings.mode == "demo":
            excerpts = "\n\n".join(f"[{i}] {ref['content']}" for i, ref in enumerate(references, 1))
            answer = (
                "以下为本地示例资料的检索摘录（示例模式，未调用大模型）：\n\n"
                + excerpts
                + "\n\n这些资料用于演示检索与引用，不构成个人诊断或治疗建议。"
            )
        else:
            answer = self._get_llm().generate(question, references, max_tokens=max_tokens)
        return {
            "answer": answer,
            "references": references,
            "mode": self.settings.mode,
            "generation_mode": generation_mode,
            "retrieval_backend": getattr(retriever, "backend_name", "injected"),
            "rerank_applied": rerank_applied,
        }

    def close(self) -> None:
        with self._lock:
            for resource in (self._retriever, self._reranker, self._llm):
                close = getattr(resource, "close", None)
                if callable(close):
                    close()
