from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from medical_rag.evaluate import (
    evaluate_records, load_records, main, reproducibility_configuration,
    save_evaluation, source_names,
)
from medical_rag.ingest import ingest_pdfs, iter_chunks

try:
    import fitz
except ImportError:
    fitz = None


class ChunkingTests(unittest.TestCase):
    def test_overlap_windows_stop_at_the_final_tail(self):
        self.assertEqual(list(iter_chunks("abcdefghij", 4, 1)), ["abcd", "defg", "ghij"])
        self.assertEqual(list(iter_chunks("abc", 4, 3)), ["abc"])
        self.assertEqual(list(iter_chunks("", 4, 3)), [])

    def test_invalid_windows_rejected_even_when_text_empty(self):
        for size, overlap in ((0, 0), (5, 5), (5, 6), (5, -1)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                list(iter_chunks("", size, overlap))

    def test_no_pdf_fails_without_creating_indexes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "PDF"):
                ingest_pdfs(root, root / "knowledge", root / "images")
            self.assertFalse((root / "knowledge").exists())


@unittest.skipIf(fitz is None, "PDF extraction tests require PyMuPDF")
class PdfIngestionTests(unittest.TestCase):
    def test_scanned_page_without_text_keeps_image_but_skips_index_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_dir = root / "pdfs"
            pdf_dir.mkdir()
            with fitz.open() as document:
                page = document.new_page()
                pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 160, 160), False)
                pixmap.clear_with(255)
                page.insert_image(fitz.Rect(30, 30, 190, 190), stream=pixmap.tobytes("png"))
                document.save(pdf_dir / "scanned.pdf")
            output_dir, image_dir = root / "knowledge", root / "images"
            summary = ingest_pdfs(pdf_dir, output_dir, image_dir)
            self.assertEqual(summary["chunk_count"], 0)
            self.assertEqual(summary["image_count"], 0)
            self.assertEqual(summary["extracted_image_count"], 1)
            self.assertEqual(summary["skipped_unsearchable_images"], 1)
            self.assertEqual((output_dir / "images.jsonl").read_text(encoding="utf-8"), "")
            self.assertEqual(len(list(image_dir.rglob("*.png"))), 1)

    def test_text_images_and_source_pages_are_traceable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_dir = root / "pdfs"
            nested = pdf_dir / "nested"
            nested.mkdir(parents=True)
            pdf_path = nested / "reference.pdf"
            with fitz.open() as document:
                page = document.new_page()
                page.insert_textbox(fitz.Rect(30, 30, 550, 700), "Screening and follow-up. " * 40, fontsize=11)
                page = document.new_page()
                page.insert_text((30, 40), "Page two: educational example.")
                pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 160, 160), False)
                pixmap.clear_with(255)
                page.insert_image(fitz.Rect(30, 100, 190, 260), stream=pixmap.tobytes("png"))
                page.insert_text((30, 280), "Figure 1. Demonstration illustration.")
                document.save(pdf_path)
            output_dir, image_dir = root / "knowledge", root / "images"
            summary = ingest_pdfs(pdf_dir, output_dir, image_dir, chunk_size=100, chunk_overlap=20)
            chunks = [json.loads(line) for line in (output_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
            images = [json.loads(line) for line in (output_dir / "images.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["pdf_count"], 1)
            self.assertEqual(summary["page_count"], 2)
            self.assertGreater(len(chunks), 2)
            self.assertEqual({row["page"] for row in chunks}, {1, 2})
            self.assertTrue(all(row["source_file"] == "nested/reference.pdf" for row in chunks))
            self.assertTrue(all(len(row["text"]) <= 100 for row in chunks))
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0]["page"], 2)
            self.assertEqual(images[0]["source_file"], "nested/reference.pdf")
            self.assertIn("Figure 1", images[0]["caption"])
            self.assertIn("Page two", images[0]["context"])
            self.assertFalse(Path(images[0]["image_path"]).is_absolute())
            self.assertTrue((image_dir / images[0]["image_path"]).is_file())
            before = (output_dir / "chunks.jsonl").read_bytes()
            with self.assertRaises(FileExistsError):
                ingest_pdfs(pdf_dir, output_dir, image_dir)
            self.assertEqual((output_dir / "chunks.jsonl").read_bytes(), before)


class FakePipeline:
    def chat(self, question):
        if question == "broken":
            raise RuntimeError("simulated backend failure")
        return {
            "answer": "Screening is an educational keyword example.",
            "references": [{"source_file": "manuals/demo.pdf", "page": 2}],
            "mode": "demo", "retrieval_backend": "fake_test_backend",
            "generation_mode": "fake_test_answer",
        }


class EvaluationTests(unittest.TestCase):
    def test_public_configuration_hashes_demo_knowledge_without_secrets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            knowledge = root / "knowledge.jsonl"
            knowledge.write_bytes(b'{"text":"example"}\n')
            settings = SimpleNamespace(
                mode="demo", data_dir=root, demo_knowledge_path=knowledge,
                embedding_model="test-embedding", reranker_model="test-reranker",
                llm_model="test-llm", collection="test-collection", request_timeout=15,
                llm_api_key="secret-api-key", llm_base_url="https://secret-user:secret-password@example.test",
                milvus_uri="https://secret-milvus-token@example.test",
            )
            configuration = reproducibility_configuration(settings)
            self.assertEqual(configuration["demo_knowledge_sha256"], hashlib.sha256(knowledge.read_bytes()).hexdigest())
            self.assertEqual(configuration["collection"], "test-collection")
            self.assertEqual(configuration["embedding_model"], "test-embedding")
            self.assertNotIn("secret", json.dumps(configuration))
            self.assertNotIn("llm_api_key", configuration)
            self.assertNotIn("milvus_uri", configuration)

    def test_cli_records_effective_parameters_and_closes_pipeline(self):
        from medical_rag.config import Settings
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            knowledge, dataset, output = root / "knowledge.jsonl", root / "input.jsonl", root / "run.jsonl"
            knowledge.write_text('{"text":"example"}\n', encoding="utf-8")
            dataset.write_text('{"question":"ok"}\n', encoding="utf-8")
            settings = Settings(mode="demo", demo_knowledge_path=knowledge, data_dir=root)
            pipeline = MagicMock()
            pipeline.chat.return_value = FakePipeline().chat("ok")
            with patch("medical_rag.config.Settings.from_env", return_value=settings), patch("medical_rag.pipeline.Pipeline", return_value=pipeline), redirect_stdout(io.StringIO()):
                self.assertEqual(main([
                    "--input", str(dataset), "--output", str(output),
                    "--retrieval-top-k", "4", "--final-top-k", "2", "--no-rerank",
                    "--chunk-chars", "300", "--max-tokens", "99",
                ]), 0)
            pipeline.close.assert_called_once_with()
            expected = {"retrieval_top_k": 4, "final_top_k": 2, "use_rerank": False, "chunk_chars": 300, "max_tokens": 99}
            pipeline.chat.assert_called_once_with("ok", **expected)
            summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["configuration"]["chat_parameters"], expected)
            self.assertEqual(summary["configuration"]["demo_knowledge_sha256"], hashlib.sha256(knowledge.read_bytes()).hexdigest())

    def test_cli_closes_pipeline_when_scorer_initialization_fails(self):
        from medical_rag.config import Settings
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            knowledge, dataset = root / "knowledge.jsonl", root / "input.jsonl"
            knowledge.write_text('{"text":"example"}\n', encoding="utf-8")
            dataset.write_text('{"question":"ok"}\n', encoding="utf-8")
            settings = Settings(mode="demo", demo_knowledge_path=knowledge, data_dir=root)
            pipeline = MagicMock()
            with patch("medical_rag.config.Settings.from_env", return_value=settings), patch("medical_rag.pipeline.Pipeline", return_value=pipeline), patch("medical_rag.evaluate.make_similarity_scorer", side_effect=RuntimeError("test scorer unavailable")), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exit_context:
                    main(["--input", str(dataset), "--output", str(root / "run.jsonl"), "--similarity-model", "test-scorer"])
            self.assertEqual(exit_context.exception.code, 1)
            pipeline.close.assert_called_once_with()
            pipeline.chat.assert_not_called()

    def test_failed_questions_remain_in_metric_denominators(self):
        records = [
            {"question": question, "reference_answer": "Example reference.",
             "expected_keywords": ["screening", "absent"], "expected_sources": ["demo.pdf"]}
            for question in ("ok", "broken")
        ]
        clock = iter([0.0, 2.0, 5.0, 9.0])
        rows, summary = evaluate_records(records, FakePipeline(), similarity_scorer=lambda reference, answer: 0.8, clock=lambda: next(clock))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["status"], "error")
        self.assertIn("simulated backend failure", rows[1]["error"])
        self.assertEqual(rows[1]["keyword_coverage"], 0.0)
        self.assertEqual(summary["total_count"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["keyword_evaluated_count"], 2)
        self.assertEqual(summary["keyword_coverage"], 0.25)
        self.assertEqual(summary["source_hit_rate"], 0.5)
        self.assertEqual(summary["answer_similarity_mean"], 0.4)
        self.assertEqual(summary["latency_seconds"]["p50"], 3.0)
        self.assertAlmostEqual(summary["latency_seconds"]["p95"], 3.9)

    def test_unannotated_metrics_are_null(self):
        rows, summary = evaluate_records([{"question": "ok"}], FakePipeline())
        self.assertIsNone(rows[0]["keyword_coverage"])
        self.assertIsNone(summary["source_hit_rate"])
        self.assertIsNone(summary["answer_similarity_mean"])
        self.assertEqual(summary["keyword_evaluated_count"], 0)

    def test_source_matching_accepts_legacy_fields_and_windows_paths(self):
        self.assertEqual(source_names([
            {"source_file": "C:\\guides\\DEMO.PDF"},
            {"source": "manuals/other.pdf"},
            {"metadata": {"source_file": "third.pdf"}},
        ]), {"demo.pdf", "other.pdf", "third.pdf"})

    def test_similarity_failure_does_not_disappear_from_denominator(self):
        def failing_score(reference, answer):
            raise RuntimeError("scorer failed")
        rows, summary = evaluate_records(
            [{"question": "ok", "reference_answer": "reference", "expected_keywords": ["screening"]}],
            FakePipeline(), similarity_scorer=failing_score,
        )
        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(summary["similarity_evaluated_count"], 1)
        self.assertEqual(summary["answer_similarity_mean"], 0.0)
        self.assertEqual(summary["keyword_coverage"], 0.0)

    def test_input_hash_limit_and_existing_output_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "questions.jsonl"
            raw = b'{"question":"ok","keywords":["screening"]}\n{"question":"broken"}\n'
            dataset.write_bytes(raw)
            records, digest = load_records(dataset, limit=1)
            self.assertEqual(len(records), 1)
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            rows, summary = evaluate_records(records, FakePipeline())
            output = root / "run.jsonl"
            summary_path = save_evaluation(output, rows, summary)
            self.assertEqual(summary_path.name, "run.summary.json")
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8"))["total_count"], 1)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                save_evaluation(output, rows, summary)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
