"""Reproducible retrieval and answer checks, without claiming clinical accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} 必须是字符串或字符串列表")
    return list(dict.fromkeys(item.strip() for item in value if item.strip()))


def normalize_record(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    question = row.get("question", "")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"第 {index + 1} 条样本缺少非空 question")
    reference = row.get("reference_answer", row.get("reference", row.get("answer", "")))
    if reference is None:
        reference = ""
    if not isinstance(reference, str):
        raise ValueError(f"第 {index + 1} 条样本的 reference_answer 必须是字符串")
    return {
        "id": row.get("id", index + 1), "question": question.strip(),
        "reference_answer": reference.strip(),
        "expected_keywords": _string_list(row.get("expected_keywords", row.get("keywords")), "expected_keywords"),
        "expected_sources": _string_list(row.get("expected_sources", row.get("expected_source", row.get("source"))), "expected_sources"),
    }


def load_records(path: Path, limit: int | None = None) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("data", data.get("questions"))
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise ValueError("评估输入必须是 JSONL 对象行，或 JSON 对象数组")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        data = data[:limit]
    if not data:
        raise ValueError("评估输入不能为空")
    return [normalize_record(row, i) for i, row in enumerate(data)], hashlib.sha256(raw).hexdigest()


def _filename(value: str) -> str:
    return value.strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()


def source_names(references: list[Any]) -> set[str]:
    """Accept both current source_file and the legacy source field."""
    names: set[str] = set()
    for reference in references:
        if isinstance(reference, str):
            names.add(_filename(reference))
        elif isinstance(reference, Mapping):
            metadata = reference.get("metadata")
            candidates = [reference]
            if isinstance(metadata, Mapping):
                candidates.append(metadata)
            for candidate in candidates:
                for key in ("source_file", "source", "file", "filename", "path"):
                    value = candidate.get(key)
                    if isinstance(value, str) and value.strip():
                        names.add(_filename(value))
    return names


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate_records(
    records: list[dict[str, Any]],
    pipeline: Any,
    *,
    mode: str = "demo",
    seed: int = 42,
    similarity_scorer: Callable[[str, str], float] | None = None,
    chat_parameters: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score each attempt, retaining failed requests with zero applicable scores.

    Each metric uses the number of questions with its required annotations as
    its denominator, including failed questions.  Missing annotations produce
    null, never an invented score.  Latencies cover Pipeline.chat only.
    """
    random.seed(seed)
    rows: list[dict[str, Any]] = []
    for index, original in enumerate(records):
        record = normalize_record(original, index)
        keywords, expected_sources = record["expected_keywords"], record["expected_sources"]
        similarity_applicable = similarity_scorer is not None and bool(record["reference_answer"])
        result: dict[str, Any] = {
            **record, "answer": "", "references": [], "status": "error", "error": None,
            "mode": mode, "retrieval_backend": "unknown", "generation_mode": "unknown",
            "keyword_coverage": 0.0 if keywords else None,
            "source_hit": 0.0 if expected_sources else None,
            "answer_similarity": 0.0 if similarity_applicable else None,
            "matched_keywords": [], "matched_sources": [],
        }
        started = clock()
        try:
            response = pipeline.chat(record["question"], **(chat_parameters or {}))
            result["latency_seconds"] = max(0.0, clock() - started)
            if hasattr(response, "model_dump"):
                response = response.model_dump()
            if not isinstance(response, Mapping):
                raise TypeError("Pipeline.chat 必须返回字典或 Pydantic 对象")
            answer = response.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Pipeline.chat 未返回非空 answer")
            references = response.get("references", response.get("sources", [])) or []
            if not isinstance(references, list):
                raise TypeError("references 必须是列表")
            result.update({
                "answer": answer, "references": references,
                "mode": response.get("mode", mode),
                "generation_mode": response.get("generation_mode", "unknown"),
                "retrieval_backend": response.get("retrieval_backend", "unknown"),
            })
            matched_keywords = [word for word in keywords if word.casefold() in answer.casefold()]
            present_sources = source_names(references)
            matched_sources = [source for source in expected_sources if _filename(source) in present_sources]
            # A scorer error invalidates this attempt instead of silently reducing
            # the similarity denominator or presenting an incomplete evaluation.
            similarity = similarity_scorer(record["reference_answer"], answer) if similarity_applicable else None
            if similarity is not None:
                import math
                similarity = float(similarity)
                if not math.isfinite(similarity):
                    raise ValueError("相似度计算返回非有限值")
            result.update({
                "status": "success",
                "matched_keywords": matched_keywords, "matched_sources": matched_sources,
                "keyword_coverage": len(matched_keywords) / len(keywords) if keywords else None,
                "source_hit": float(bool(matched_sources)) if expected_sources else None,
                "answer_similarity": similarity,
            })
        except Exception as exc:
            if "latency_seconds" not in result:
                result["latency_seconds"] = max(0.0, clock() - started)
            result["error"] = f"{type(exc).__name__}: {exc}"[:1000]
        rows.append(result)

    keyword_scores = [row["keyword_coverage"] for row in rows if row["keyword_coverage"] is not None]
    source_scores = [row["source_hit"] for row in rows if row["source_hit"] is not None]
    similarity_scores = [row["answer_similarity"] for row in rows if row["answer_similarity"] is not None]
    latencies = [row["latency_seconds"] for row in rows]
    successful = sum(row["status"] == "success" for row in rows)
    summary = {
        "total_count": len(rows), "success_count": successful,
        "error_count": len(rows) - successful,
        "success_rate": successful / len(rows) if rows else None,
        "keyword_coverage": _mean(keyword_scores), "keyword_evaluated_count": len(keyword_scores),
        "source_hit_rate": _mean(source_scores), "source_evaluated_count": len(source_scores),
        "answer_similarity_mean": _mean(similarity_scores), "similarity_evaluated_count": len(similarity_scores),
        "latency_seconds": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
        "backend_modes": dict(Counter(str(row["mode"]) for row in rows)),
        "retrieval_backends": dict(Counter(str(row["retrieval_backend"]) for row in rows)),
        "generation_modes": dict(Counter(str(row["generation_mode"]) for row in rows)),
        "seed": seed, "seed_scope": "Python random; backend determinism is not guaranteed",
        "metric_definitions": {
            "keyword_coverage": "每题预期关键词在回答中出现的比例，再按已标注题目平均；失败计 0",
            "source_hit_rate": "预期来源中至少一个文件名出现在引用中的题目比例；失败计 0",
            "answer_similarity_mean": "可选句向量余弦相似度；不是医学准确率；失败计 0",
            "denominators": "每项指标使用具备该项标注的全部题目数，包含失败题；无标注项为 null",
            "latency_seconds": "全部 Pipeline.chat 调用（含失败）的端到端耗时；不含相似度打分时间",
        },
    }
    return rows, summary


def make_similarity_scorer(model_name: str) -> Callable[[str, str], float]:
    """Load sentence-transformers only when explicitly requested by the CLI."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("--similarity-model 需要 sentence-transformers，请先安装 live 依赖") from exc
    model = SentenceTransformer(model_name)

    def score(reference: str, answer: str) -> float:
        vectors = model.encode([reference, answer], normalize_embeddings=True)
        return float(vectors[0] @ vectors[1])

    return score


def reproducibility_configuration(settings: Any) -> dict[str, Any]:
    """Record an explicit allowlist; endpoint URIs can contain credentials."""
    fields = (
        "mode", "data_dir", "demo_knowledge_path", "embedding_model",
        "reranker_model", "llm_model", "collection", "request_timeout",
    )
    configuration = {}
    for name in fields:
        value = getattr(settings, name)
        if isinstance(value, Path):
            try:
                value = value.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix()
            except ValueError:
                value = "<external>/" + value.name
        configuration[name] = value
    configuration["demo_knowledge_sha256"] = (
        hashlib.sha256(Path(settings.demo_knowledge_path).read_bytes()).hexdigest()
        if settings.mode == "demo" else None
    )
    configuration["endpoint_uris_recorded"] = False
    return configuration


def save_evaluation(output: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    summary_path = output.with_suffix(".summary.json")
    if output == summary_path:
        raise ValueError("output 不能以 .summary.json 命名")
    for path in (output, summary_path):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有评估记录：{path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with summary_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="医疗 RAG 逐题评估与可复核结果保存")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("live", "demo"), default="demo")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity-model", help="显式启用句向量相似度；可能下载指定模型")
    parser.add_argument("--demo-knowledge", help="可选的 demo 知识 JSONL 路径")
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--final-top-k", type=int, default=3)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--chunk-chars", type=int, default=400)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    pipeline = None
    try:
        for path in (args.output, args.output.with_suffix(".summary.json")):
            if path.exists():
                raise FileExistsError(f"拒绝覆盖已有评估记录：{path}")
        records, digest = load_records(args.input, args.limit)
        from medical_rag.config import Settings
        from medical_rag.pipeline import Pipeline
        settings = replace(Settings.from_env(), mode=args.mode)
        if args.demo_knowledge:
            settings = replace(settings, demo_knowledge_path=Path(args.demo_knowledge))
        chat_parameters = {
            "retrieval_top_k": args.retrieval_top_k, "final_top_k": args.final_top_k,
            "use_rerank": not args.no_rerank, "chunk_chars": args.chunk_chars,
            "max_tokens": args.max_tokens,
        }
        if any(chat_parameters[name] <= 0 for name in ("retrieval_top_k", "final_top_k", "chunk_chars", "max_tokens")):
            raise ValueError("检索条数、片段长度和 max_tokens 必须大于 0")
        configuration = reproducibility_configuration(settings)
        pipeline = Pipeline(settings)
        scorer = make_similarity_scorer(args.similarity_model) if args.similarity_model else None
        rows, summary = evaluate_records(
            records, pipeline, mode=args.mode, seed=args.seed,
            similarity_scorer=scorer, chat_parameters=chat_parameters,
        )
        summary.update({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(args.input), "dataset_sha256": digest,
            "configuration": {
                **configuration, "limit": args.limit, "seed": args.seed,
                "similarity_model": args.similarity_model,
                "chat_parameters": chat_parameters,
            },
        })
        summary_path = save_evaluation(args.output, rows, summary)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"评估失败：{exc}\n")
    finally:
        if pipeline is not None:
            pipeline.close()
    print(json.dumps({"output": str(args.output), "summary_path": str(summary_path), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
