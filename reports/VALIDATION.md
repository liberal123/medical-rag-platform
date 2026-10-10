# 本地工程验证

执行时间（UTC）：2026-10-10T15:37:11.083910+00:00。环境：Windows 11，Python 3.12.13。

| 检查 | 实际结果 |
| --- | --- |
| 单元与集成测试 | 运行 38 项，失败 0，错误 0，跳过 0 |
| 真实 PDF 临时样本 | 文字切块、页码、嵌入图片、无文字扫描页处理通过 |
| 本地 uvicorn HTTP 服务 | 页面、健康状态、问答、图片和非法参数响应通过 |
| demo 逐题回归 | 5/5 正常返回 |
| GPU / Milvus / BGE / GLM-4 实际后端 | 本次未运行；对应接口行为使用测试替身验证 |

执行命令：

```bash
python scripts/validate_release.py --output-dir reports/run-new
```

脚本调用完整 unittest、三个 CLI 帮助入口、演示题集评估，并启动真实 HTTP 进程验证响应。选择新的目录重跑，以保留每次输出。

- [测试日志](unittest.log)
- [API 实际响应](api_smoke.json)
- [逐题 demo 记录](demo_eval.jsonl) / [汇总](demo_eval.summary.json)
- [环境与源码 SHA256](local_validation.json)

demo 回归用于自编流程样例的工程验证，不能据此报告医学准确率或 GLM-4 的生成质量。没有运行浏览器视觉验收；HTTP 检查确认 HTML 与相关图片可正常提供。
