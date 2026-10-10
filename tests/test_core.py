from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import httpx

from medical_rag.build_index import build_index
from medical_rag.config import Settings
from medical_rag.llm import LLMError, VLLMClient
from medical_rag.pipeline import Pipeline
from medical_rag.retrieval import CrossEncoderReranker, DemoRetriever, MilvusRetriever, read_jsonl


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "knowledge.jsonl"
        self.records = [
            {"content": "糖尿病视网膜病变筛查使用眼底检查，具体计划由医生评估。", "source_file": "demo.md", "page": 2, "source": "text"},
            {"content": "图注：眼底检查的示例流程图。", "source_file": "demo.md", "page": 3, "source": "image", "image_path": "sample.svg", "caption": "示例流程图"},
        ]
        self.path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in self.records), encoding="utf-8")
        self.settings = Settings(demo_knowledge_path=self.path, data_dir=Path(self.temp.name))

    def test_demo_retrieval_ranks_relevant_evidence_and_preserves_metadata(self):
        results = DemoRetriever(self.path).search("糖尿病视网膜病变筛查", 2)
        self.assertEqual(results[0]["page"], 2)
        self.assertEqual(results[0]["source_file"], "demo.md")
        self.assertGreater(results[0]["score"], 0)

    def test_demo_explicitly_labels_generation_without_calling_models(self):
        llm, reranker = Mock(), Mock()
        result = Pipeline(self.settings, llm=llm, reranker=reranker).chat("糖尿病视网膜病变筛查")
        self.assertEqual(result["generation_mode"], "extractive_demo")
        self.assertEqual(result["retrieval_backend"], "lexical_demo")
        self.assertFalse(result["rerank_applied"])
        llm.generate.assert_not_called()
        reranker.rerank.assert_not_called()

    def test_no_evidence_does_not_call_live_model(self):
        retriever, llm, reranker = Mock(), Mock(), Mock()
        retriever.search.return_value = []
        result = Pipeline(Settings(mode="live"), retriever=retriever, llm=llm, reranker=reranker).chat("未知内容")
        self.assertEqual(result["generation_mode"], "no_evidence")
        llm.generate.assert_not_called()
        reranker.rerank.assert_not_called()

    def test_live_rerank_reorders_evidence_and_cites_exact_prompt(self):
        retriever, llm, reranker = Mock(), Mock(), Mock()
        retriever.search.return_value = self.records
        reranker.rerank.return_value = [{**self.records[1], "rerank_score": 0.9}]
        llm.generate.return_value = "见流程 [1]。"
        result = Pipeline(Settings(mode="live"), retriever=retriever, llm=llm, reranker=reranker).chat("眼底检查", final_top_k=1)
        self.assertTrue(result["rerank_applied"])
        self.assertEqual(result["references"][0]["image_path"], "sample.svg")
        self.assertEqual(llm.generate.call_args.args[1], result["references"])
        self.assertEqual(result["generation_mode"], "vllm")

    def test_live_generation_failure_is_not_replaced_with_demo(self):
        retriever, llm = Mock(), Mock()
        retriever.search.return_value = self.records
        llm.generate.side_effect = LLMError("server unavailable")
        with self.assertRaises(LLMError):
            Pipeline(Settings(mode="live"), retriever=retriever, llm=llm).chat("眼底检查", use_rerank=False)

    def test_invalid_parameters_are_rejected_before_loading_models(self):
        pipeline = Pipeline(self.settings)
        for parameters in ({"question": " "}, {"question": "筛查", "final_top_k": 11}, {"question": "筛查", "chunk_chars": 0}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                pipeline.chat(**parameters)
        self.assertIsNone(pipeline._retriever)

    def test_settings_use_environment_and_reject_invalid_mode(self):
        settings = Settings.from_env({"MEDRAG_MODE": "live", "MEDRAG_DATA_DIR": self.temp.name, "MEDRAG_LLM_MODEL": "local-model"})
        self.assertEqual(settings.image_dir, Path(self.temp.name).resolve() / "images")
        self.assertEqual(settings.llm_model, "local-model")
        with self.assertRaises(ValueError):
            Settings.from_env({"MEDRAG_MODE": "pretend"})

    def test_malformed_jsonl_reports_line(self):
        self.path.write_text('{"content": "ok"}\n{"content": ""}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "knowledge.jsonl:2"):
            read_jsonl(self.path)

    def test_milvus_search_uses_dense_vectors_and_preserves_image_metadata(self):
        client, embedder = Mock(), Mock()
        client.has_collection.return_value = True
        client.search.return_value = [[{"entity": self.records[1], "distance": 0.8}]]
        embedder.encode.return_value = [[0.2, 0.4]]
        results = MilvusRetriever(Settings(mode="live"), client=client, embedder=embedder).search("眼底检查", 5)
        self.assertEqual(results[0]["caption"], "示例流程图")
        self.assertEqual(client.search.call_args.kwargs["data"], [[0.2, 0.4]])

    def test_reranker_orders_by_predicted_scores(self):
        model = Mock()
        model.predict.return_value = [0.1, 0.8]
        results = CrossEncoderReranker("test", model=model).rerank("眼底检查", self.records, 1)
        self.assertEqual(results[0]["page"], 3)
        self.assertEqual(results[0]["rerank_score"], 0.8)

    def test_index_refuses_existing_collection_before_embedding(self):
        client, embedder = Mock(), Mock()
        client.has_collection.return_value = True
        with self.assertRaisesRegex(ValueError, "already exists"):
            build_index(self.settings, [self.path], client=client, embedder=embedder)
        client.create_collection.assert_not_called()
        embedder.encode.assert_not_called()

    def test_index_build_preserves_reference_fields_and_dimension(self):
        client, embedder = Mock(), Mock()
        client.has_collection.return_value = False
        embedder.get_sentence_embedding_dimension.return_value = 2
        embedder.encode.return_value = [[0.2, 0.4], [0.5, 0.6]]
        summary = build_index(self.settings, [self.path], client=client, embedder=embedder)
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["dimension"], 2)
        payload = client.insert.call_args.kwargs["data"]
        self.assertEqual(payload[1]["image_path"], "sample.svg")
        self.assertEqual(payload[0]["image_path"], "")

    def test_llm_request_uses_configured_model_and_grounded_prompt(self):
        observed = {}

        def respond(request):
            observed["payload"] = json.loads(request.content)
            observed["authorization"] = request.headers.get("authorization")
            return httpx.Response(200, json={"choices": [{"message": {"content": "资料摘录 [1]"}}]})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            llm = VLLMClient(Settings(llm_model="local-model", llm_api_key="test-key"), client=client)
            self.assertEqual(llm.generate("眼底检查", self.records), "资料摘录 [1]")
        self.assertEqual(observed["payload"]["model"], "local-model")
        self.assertIn("demo.md", observed["payload"]["messages"][1]["content"])
        self.assertEqual(observed["authorization"], "Bearer test-key")

    def test_llm_http_error_is_explicit_and_does_not_leak_response(self):
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401, text="secret-server-data"))) as client:
            with self.assertRaises(LLMError) as raised:
                VLLMClient(self.settings, client=client).generate("眼底检查", self.records)
        self.assertNotIn("secret-server-data", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
