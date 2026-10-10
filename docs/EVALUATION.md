# 评估口径与结果追溯

## 可运行的工程回归

```bash
python -m medical_rag.evaluate --input examples/eval_demo.jsonl \
  --output data/eval-runs/demo.jsonl --mode demo
```

生成 `demo.jsonl`（逐题回答、引用、耗时与错误）及 `demo.summary.json`（汇总、分母、运行模式与数据 SHA256）。已有输出不会覆盖，请为每次运行选择新的文件名。

`examples/eval_demo.jsonl` 只检验自编流程样例的检索和引用，不是医学能力评测。本次实际执行结果见 [工程验证](../reports/VALIDATION.md)。

## 真实知识库评估

题集格式：

```json
{"question":"需要检验的问题", "reference_answer":"经过资料核对的参考答案", "expected_keywords":["关键词"], "expected_sources":["guideline.pdf"]}
```

```bash
python -m dotenv -f .env run -- python -m medical_rag.evaluate \
  --input data/eval/questions.jsonl --output data/eval-runs/live-001.jsonl --mode live
```

通过 `--similarity-model /path/to/embedding-model` 显式开启答案语义相似度；默认不加载额外模型，不生成这一指标。评估前固定题集与知识库版本、检索配置、生成模型、硬件、预热方式和代码 commit。用于开发的题目与最终报告题目应分开。

| 指标 | 实际含义 | 局限 |
| --- | --- | --- |
| `success_rate` | 调用正常返回的题目比例 | 成功返回不等于正确 |
| `keyword_coverage` | 各题预期关键词出现比例的平均 | 关键词出现不证明结论正确 |
| `source_hit_rate` | 至少一个预期来源文件名被引用的题目比例 | 不代表全部所需证据被召回 |
| `answer_similarity_mean` | 答案与参考文本的句向量余弦相似度 | 不等于医学准确率 |
| 延迟 P50/P95 | `Pipeline.chat` 调用耗时分位数，包含失败调用 | 受硬件、预热和语料规模影响 |

每项指标以具备对应标注的全部题目为分母，失败题计 0；缺标注指标为 `null`。相似度打分错误也保留为失败，避免只挑成功题统计。语义相似度打分时间不计入问答耗时。

临床结论需要独立的医学审核，至少核对证据支持度、事实一致性、适用人群、过时知识和无证据时的处理方式。当前脚本没有实现临床正确性评分或自动事实验证。

## 历史说明材料

旧项目文档包含 baseline、重排和补充语料的指标表，但当前文件中没有找到与其对应的完整逐题输出、数据指纹和运行环境记录。因此发布首页没有将这些数字列为已复核实验结论。旧资料保留在本机备份中，未来找到原始记录后可按上述口径核对。

本仓库提供真实执行的工程检查，以及可以重新运行的模型评估工具；工程测试通过不能替代 GLM-4 / Milvus 真实链路的效果评估。
