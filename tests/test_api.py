"""API contract, validation and local image boundary checks without model loading."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from medical_rag.app import create_app


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.failure = None
        self.answer = "检索摘录 [1]"
        self.image_path = "workflow.svg"
        self.closed = False

    def close(self):
        self.closed = True

    def chat(self, question, **kwargs):
        self.calls.append((question, kwargs))
        if self.failure:
            raise self.failure
        return {
            "answer": self.answer,
            "mode": "demo",
            "generation_mode": "extractive",
            "retrieval_backend": "local",
            "rerank_applied": False,
            "references": [{
                "content": "人工流程说明",
                "source_file": r"C:\private\documents\sample.pdf",
                "source": "人工流程样例",
                "page": 2,
                "score": 0.75,
                "image_path": self.image_path,
                "caption": "流程示意",
            }],
        }


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        (self.images / "workflow.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8"
        )
        (self.root / "outside.svg").write_text("private", encoding="utf-8")
        (self.images / "private.txt").write_text("not an image", encoding="utf-8")
        self.settings = SimpleNamespace(mode="demo", data_dir=self.root, image_dir=self.images)
        self.pipeline = FakePipeline()
        self.client = TestClient(create_app(self.settings, self.pipeline))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_chat_includes_sources_mode_and_safe_image_url(self):
        response = self.client.post("/dr/chat", json={"question": " 来源是什么？ "})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "demo")
        self.assertIsInstance(body["latency_seconds"], (int, float))
        self.assertGreaterEqual(body["latency_seconds"], 0)
        self.assertIn("人工流程样例", body["demo_notice"])
        self.assertEqual(body["references"][0]["label"], "[1]")
        self.assertEqual(body["references"][0]["source_file"], "sample.pdf")
        self.assertEqual(body["references"][0]["image_url"], "/dr/images/workflow.svg")
        self.assertNotIn("image_path", body["references"][0])
        self.assertNotIn("C:\\private", response.text)
        self.assertEqual(self.pipeline.calls[0][0], "来源是什么？")

    def test_invalid_inputs_do_not_call_pipeline(self):
        cases = [
            {"question": " "},
            {"question": "x" * 2001},
            {"question": "q", "retrieval_top_k": 0},
            {"question": "q", "retrieval_top_k": 51},
            {"question": "q", "final_top_k": 11},
            {"question": "q", "retrieval_top_k": 2, "final_top_k": 3},
            {"question": "q", "max_tokens": 2049},
            {"question": "q", "chunk_chars": 99},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/dr/chat", json=payload).status_code, 422)
        self.assertEqual(self.pipeline.calls, [])

    def test_pipeline_errors_are_not_successful_answers(self):
        self.pipeline.failure = RuntimeError("secret token and local path")
        response = self.client.post("/dr/chat", json={"question": "问题"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("answer", response.json())
        self.assertNotIn("secret", response.text)

    def test_empty_answer_is_not_success(self):
        self.pipeline.answer = ""
        self.assertEqual(self.client.post("/dr/chat", json={"question": "问题"}).status_code, 503)

    def test_live_health_does_not_claim_dependencies_ready(self):
        self.settings.mode = "live"
        response = self.client.get("/dr/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "live")
        self.assertEqual(response.json()["dependencies"], "not_checked")
        self.assertNotIn("ready", response.json())

    def test_pipeline_is_closed_on_shutdown(self):
        pipeline = FakePipeline()
        with TestClient(create_app(self.settings, pipeline)) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertFalse(pipeline.closed)
        self.assertTrue(pipeline.closed)

    def test_image_endpoint_serves_only_images_inside_root(self):
        self.assertEqual(self.client.get("/dr/images/workflow.svg").status_code, 200)
        for url in [
            "/dr/images/%2e%2e%2foutside.svg",
            "/dr/images/..%5coutside.svg",
            "/dr/images/private.txt",
            "/dr/images/missing.png",
            "/dr/images/C%3A%2Foutside.svg",
        ]:
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 404)

    def test_missing_or_escaping_image_is_omitted_from_reference(self):
        for image_path in ["missing.png", "../outside.svg", str(self.root / "outside.svg")]:
            with self.subTest(image_path=image_path):
                self.pipeline.image_path = image_path
                response = self.client.post("/dr/chat", json={"question": "来源"})
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("image_url", response.json()["references"][0])

    def test_absolute_image_inside_root_is_published_as_relative_url(self):
        self.pipeline.image_path = str(self.images / "workflow.svg")
        response = self.client.post("/dr/chat", json={"question": "图引用"})
        self.assertEqual(response.status_code, 200)
        reference = response.json()["references"][0]
        self.assertEqual(reference["image_url"], "/dr/images/workflow.svg")
        self.assertNotIn(str(self.root), response.text)

    def test_image_symlink_outside_root_is_rejected(self):
        link = self.images / "linked.svg"
        try:
            link.symlink_to(self.root / "outside.svg")
        except (OSError, NotImplementedError):
            self.skipTest("This environment does not permit creating symbolic links")
        self.assertEqual(self.client.get("/dr/images/linked.svg").status_code, 404)
        self.pipeline.image_path = "linked.svg"
        response = self.client.post("/dr/chat", json={"question": "图引用"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("image_url", response.json()["references"][0])

    def test_demo_page_is_local_and_has_accessible_question_input(self):
        response = self.client.get("/dr/demo")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="question"', response.text)
        self.assertNotIn("cdn.jsdelivr", response.text)


if __name__ == "__main__":
    unittest.main()
