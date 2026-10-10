"""HTTP API and local, dependency-free retrieval demonstration page."""

from __future__ import annotations

import logging
import math
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import Settings

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    retrieval_top_k: int = Field(10, ge=1, le=50)
    final_top_k: int = Field(3, ge=1, le=10)
    use_rerank: bool = True
    chunk_chars: int = Field(400, ge=100, le=2000)
    max_tokens: int = Field(512, ge=32, le=2048)


def _local_image(image_dir: Path, name: str) -> Path | None:
    """Resolve only existing images inside the configured image directory."""
    if not name or "\x00" in name:
        return None
    root = image_dir.resolve()
    # Backslashes are separators on Windows and must not bypass URL checks.
    candidate_name = name.replace("\\", "/")
    if candidate_name.startswith("/") or re.match(r"^[A-Za-z]:", candidate_name):
        return None
    if ".." in Path(candidate_name).parts:
        return None
    try:
        candidate = (root / candidate_name).resolve()
        candidate.relative_to(root)
        if candidate.is_file() and candidate.suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
            return candidate
    except (ValueError, OSError, RuntimeError):
        return None
    return None


def _image_url(image_dir: Path, value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    # A local index may contain an absolute path; only publish its safe relative URL.
    try:
        candidate = Path(value)
        if candidate.is_absolute():
            value = candidate.resolve().relative_to(image_dir.resolve()).as_posix()
    except (ValueError, OSError, RuntimeError):
        return None
    resolved = _local_image(image_dir, value)
    if resolved is None:
        return None
    return "/dr/images/" + quote(resolved.relative_to(image_dir.resolve()).as_posix(), safe="/")


def _filename(value: Any) -> str:
    return str(value or "未标注来源").replace("\\", "/").rsplit("/", 1)[-1]


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _references(items: Any, image_dir: Path) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise ValueError("Pipeline references must be a list")
    references = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError("Invalid reference")
        ref = {
            "label": f"[{index}]",
            "score": _finite_number(item.get("score")),
            "content": str(item.get("content", "")),
            "source_file": _filename(item.get("source_file") or item.get("source")),
            "page": item.get("page"),
            "source": _filename(item.get("source") or item.get("source_file")),
        }
        if item.get("rerank_score") is not None:
            ref["rerank_score"] = _finite_number(item["rerank_score"])
        image_url = _image_url(image_dir, item.get("image_path"))
        if image_url:
            ref["image_url"] = image_url
            ref["caption"] = str(item.get("caption") or "来源文档关联图示")
        references.append(ref)
    return references


def create_app(settings: Any = None, pipeline: Any = None) -> FastAPI:
    """Create an application; loading the pipeline is deferred until startup."""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.settings is None:
            application.state.settings = Settings.from_env()
        if application.state.pipeline is None:
            try:
                from .pipeline import Pipeline

                application.state.pipeline = await run_in_threadpool(
                    Pipeline, application.state.settings
                )
            except Exception as exc:
                LOGGER.error("Pipeline initialization failed (%s)", type(exc).__name__)
                application.state.initialization_failed = True
        try:
            yield
        finally:
            close = getattr(application.state.pipeline, "close", None)
            if callable(close):
                await run_in_threadpool(close)

    application = FastAPI(
        title="智能医疗问答平台",
        description="文献检索、上下文问答与来源追溯。示例模式仅使用人工流程样例。",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.pipeline = pipeline
    application.state.initialization_failed = False

    @application.get("/health")
    @application.get("/dr/health")
    async def health():
        current = application.state.settings
        failed = application.state.initialization_failed
        return JSONResponse(
            status_code=503 if failed else 200,
            content={
                "status": "unavailable" if failed else "ok",
                "service": "medical-rag-platform",
                "mode": getattr(current, "mode", "unconfigured"),
                "pipeline_initialized": application.state.pipeline is not None,
                "dependencies": "not_checked",
                "note": "此接口仅报告 API 进程状态；模型服务与检索后端的可用性由实际查询验证。",
            },
        )

    @application.get("/", include_in_schema=False)
    @application.get("/dr/demo", include_in_schema=False)
    async def demo_page():
        page = PROJECT_ROOT / "web" / "index.html"
        if not page.is_file():
            raise HTTPException(404, "演示页面未安装，请从项目根目录运行服务。")
        return FileResponse(page, media_type="text/html")

    @application.get("/dr/images/{image_name:path}", include_in_schema=False)
    async def image_file(image_name: str):
        current = application.state.settings
        if current is None:
            raise HTTPException(503, "服务尚未初始化。")
        image = _local_image(Path(current.image_dir), image_name)
        if image is None:
            raise HTTPException(404, "图片不存在。")
        return FileResponse(
            image,
            headers={
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
            },
        )

    @application.post("/dr/chat")
    async def chat(payload: ChatRequest):
        question = payload.question.strip()
        if not question:
            raise HTTPException(422, "问题不能为空。")
        if payload.final_top_k > payload.retrieval_top_k:
            raise HTTPException(422, "最终来源数不能超过初次检索数量。")
        active_pipeline = application.state.pipeline
        current = application.state.settings
        if active_pipeline is None or current is None:
            raise HTTPException(503, "查询服务尚未就绪，请检查服务端配置。")
        try:
            started = time.perf_counter()
            result = await run_in_threadpool(
                active_pipeline.chat,
                question,
                retrieval_top_k=payload.retrieval_top_k,
                final_top_k=payload.final_top_k,
                use_rerank=payload.use_rerank,
                chunk_chars=payload.chunk_chars,
                max_tokens=payload.max_tokens,
            )
            latency_seconds = time.perf_counter() - started
            if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
                raise ValueError("Invalid pipeline response")
            if not result["answer"].strip():
                raise ValueError("Empty pipeline response")
            return {
                "answer": result["answer"],
                "references": _references(result.get("references"), Path(current.image_dir)),
                "mode": result.get("mode", current.mode),
                "generation_mode": result.get("generation_mode", "unspecified"),
                "retrieval_backend": result.get("retrieval_backend", "unspecified"),
                "rerank_applied": bool(result.get("rerank_applied", False)),
                "latency_seconds": round(latency_seconds, 6),
                "demo_notice": (
                    "人工流程样例 · 摘录展示，不代表医学证据或大模型生成效果。"
                    if current.mode == "demo" else None
                ),
            }
        except Exception as exc:
            LOGGER.error("Query failed (%s)", type(exc).__name__)
            raise HTTPException(503, "查询处理失败，请检查服务端数据、检索后端与模型服务配置。") from exc

    return application


app = create_app()
