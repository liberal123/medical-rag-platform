"""Extract traceable text and embedded images from local PDF documents.

Images are retrieved through their captions and page context.  Extracted images
without either are saved, counted as skipped_unsearchable_images, and excluded
from images.jsonl because they cannot support text retrieval.  This module does
not perform OCR or interpret image pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def iter_chunks(text: str, chunk_size: int = 500, overlap: int = 80) -> Iterator[str]:
    """Yield character windows, ending immediately after the final full tail."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size 必须大于 0，且 0 <= chunk_overlap < chunk_size")
    text = re.sub(r"\s+", " ", text).strip()
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield text[start:end]
        if end == len(text):
            break
        start = end - overlap


def _caption(page: Any, xref: int) -> str:
    """Select the nearest text block within 180 points of an embedded image."""
    rectangles = page.get_image_rects(xref)
    if not rectangles:
        return ""
    candidates: list[tuple[float, str]] = []
    for block in page.get_text("blocks"):
        if len(block) > 6 and block[6] != 0:
            continue
        text = re.sub(r"\s+", " ", str(block[4])).strip()
        if not text:
            continue
        for rect in rectangles:
            horizontal_gap = max(rect.x0 - block[2], block[0] - rect.x1, 0)
            vertical_gap = max(rect.y0 - block[3], block[1] - rect.y1, 0)
            distance = (horizontal_gap**2 + vertical_gap**2) ** 0.5
            if distance <= 180:
                candidates.append((distance, text))
    return min(candidates, key=lambda item: item[0])[1][:300] if candidates else ""


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ingest_pdfs(
    pdf_dir: str | Path,
    output_dir: str | Path = "data/knowledge",
    image_dir: str | Path = "data/images",
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    min_image_size: int = 120,
) -> dict[str, Any]:
    """Create new JSONL indexes; never replace an existing index or image."""
    # Validate before opening files, including when every input page is empty.
    list(iter_chunks("", chunk_size, chunk_overlap))
    if min_image_size < 1:
        raise ValueError("min_image_size 必须大于 0")
    pdf_dir, output_dir, image_dir = map(Path, (pdf_dir, output_dir, image_dir))
    if not pdf_dir.is_dir():
        raise ValueError(f"PDF 目录不存在：{pdf_dir}")
    pdfs = sorted(
        (path for path in pdf_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.relative_to(pdf_dir).as_posix().casefold(),
    )
    if not pdfs:
        raise ValueError(f"目录中未找到 PDF 文件：{pdf_dir}")
    chunks_path, images_path = output_dir / "chunks.jsonl", output_dir / "images.jsonl"
    for path in (chunks_path, images_path):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有输出，请指定新的 --output-dir：{path}")

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF 导入需要 PyMuPDF，请先安装项目的 PDF 依赖") from exc

    chunks: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    run_id = "ingest_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    image_batch = image_dir / run_id
    image_batch.mkdir(parents=True, exist_ok=False)
    page_count = 0
    skipped_unsearchable_images = 0
    for pdf_path in pdfs:
        source_file = pdf_path.relative_to(pdf_dir).as_posix()
        document_id = hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:16]
        with fitz.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                page_count += 1
                page_text = re.sub(r"\s+", " ", page.get_text("text")).strip()
                for chunk_index, text in enumerate(iter_chunks(page_text, chunk_size, chunk_overlap)):
                    chunks.append({
                        "id": f"{document_id}:p{page_number}:c{chunk_index}",
                        "text": text,
                        "source_file": source_file,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "source": "DR_PDF",
                    })
                seen_xrefs: set[int] = set()
                for embedded in page.get_images(full=True):
                    xref, width, height = embedded[0], embedded[2], embedded[3]
                    if xref <= 0 or xref in seen_xrefs or min(width, height) < min_image_size:
                        continue
                    seen_xrefs.add(xref)
                    extracted = document.extract_image(xref)
                    if not extracted or not extracted.get("image"):
                        continue
                    extension = str(extracted.get("ext", "png")).lower()
                    if not re.fullmatch(r"[a-z0-9]{1,10}", extension):
                        raise ValueError(f"无法保存图片扩展名：{extension!r}")
                    image_number = len(seen_xrefs)
                    image_name = f"{document_id}_p{page_number:04d}_img{image_number:03d}.{extension}"
                    image_path = image_batch / image_name
                    with image_path.open("xb") as handle:
                        handle.write(extracted["image"])
                    caption = _caption(page, xref)
                    context = page_text[:1000]
                    image_text = "\n".join(part for part in (caption, context) if part)
                    if not image_text.strip():
                        skipped_unsearchable_images += 1
                        continue
                    images.append({
                        "id": f"{document_id}:p{page_number}:i{image_number}",
                        "text": image_text,
                        "image_path": image_path.relative_to(image_dir).as_posix(),
                        "source_file": source_file,
                        "page": page_number,
                        "caption": caption,
                        "context": context,
                        "source": "DR_PDF",
                        "width": width,
                        "height": height,
                    })
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(chunks_path, chunks)
    _write_jsonl(images_path, images)
    return {
        "pdf_count": len(pdfs), "page_count": page_count,
        "chunk_count": len(chunks), "image_count": len(images),
        "extracted_image_count": len(images) + skipped_unsearchable_images,
        "skipped_unsearchable_images": skipped_unsearchable_images,
        "chunks_path": str(chunks_path), "images_path": str(images_path),
        "image_batch": str(image_batch),
        "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        "image_retrieval": "caption_and_page_context", "ocr_enabled": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导入本地 PDF 文本、内嵌图片与来源元数据")
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--output-dir", default="data/knowledge")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument("--min-image-size", type=int, default=120)
    args = parser.parse_args(argv)
    try:
        result = ingest_pdfs(
            args.pdf_dir, args.output_dir, args.image_dir,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
            min_image_size=args.min_image_size,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"导入失败：{exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
