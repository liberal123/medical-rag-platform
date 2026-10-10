# 部署与环境配置

## 本地演示

使用 Python 3.10 或更新版本，在独立虚拟环境运行 README 的快速体验命令。`scripts/prepare_demo.py` 将仓库中的原创示意图复制到配置的数据目录。演示只依赖 CPU，不下载模型。

Windows PowerShell 若无法激活虚拟环境，可以直接使用 `.\.venv\Scripts\python.exe` 替换命令中的 `python`，无需更改系统执行策略。

## 真实模型：两个独立服务

RAG 应用负责解析、检索、重排；GPU 环境负责运行 vLLM。两者通过 OpenAI 兼容 HTTP 接口连接，避免为了网页应用重新安装整套 GPU 推理依赖。

RAG 环境建议使用 Linux、Python 3.10–3.12：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-live.txt
python -m pip install -e . --no-deps
```

`requirements-live.txt` 是兼容范围声明，尚未在本次本地整理中完成 GPU 全链路验证。PyTorch/CUDA 应按服务器驱动和实际 GPU 配套。Milvus Lite 的本地 `.db` 模式请在受支持的 Linux 环境运行；Windows 应连接独立 Milvus 服务，并单独安装对应版本的 `pymilvus` 客户端。

如果已有 vLLM 服务，检查其 `/v1/models` 返回的模型 ID，并填入 `MEDRAG_LLM_MODEL`。下列为独立 GPU 环境的启动模板，模型目录需替换为实际位置：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/your/glm-4-model \
  --served-model-name glm-4 \
  --host 127.0.0.1 --port 8001
```

应选用支持具体 GLM checkpoint 的 vLLM 版本并遵照该模型说明；部分旧模型需要 `--trust-remote-code`，仅在审阅并信任模型代码后启用。仓库不捆绑权重。旧环境笔记曾使用 vLLM 0.6.3.post1、torch 2.4.0 / CUDA 12.1、Transformers 4.45.2；这些是历史背景，不是本次验证的版本锁定文件。

## 配置与建库

在项目根目录将 `.env.example` 复制为 `.env`。程序读取进程环境变量；使用文件时，应显式加载：

```bash
python -m dotenv -f .env run -- python -m medical_rag.build_index \
  --input data/knowledge/chunks.jsonl --input data/knowledge/images.jsonl
python -m uvicorn medical_rag.app:app --env-file .env --host 127.0.0.1 --port 9001
```

运行前先按 README 导入 PDF，并把 `.env` 中 `MEDRAG_MODE` 改成 `live`。`dotenv` 命令用于 CLI，`--env-file` 用于 uvicorn；单纯复制文件不会自动加载配置。

| 环境变量 | 作用 |
| --- | --- |
| `MEDRAG_MODE` | `demo` 或 `live` |
| `MEDRAG_DATA_DIR` | PDF 衍生数据、图片及默认本地索引根目录 |
| `MEDRAG_DEMO_KNOWLEDGE` | 自编演示 JSONL 的路径 |
| `MEDRAG_EMBEDDING_MODEL` | BGE 模型名称或本地目录 |
| `MEDRAG_RERANKER_MODEL` | CrossEncoder 模型名称或本地目录 |
| `MEDRAG_MILVUS_URI` | 本地 `.db` 文件或 Milvus 服务地址 |
| `MEDRAG_COLLECTION` | 集合名；建库拒绝覆盖已有集合 |
| `MEDRAG_LLM_BASE_URL` | 包含 `/v1` 的模型服务地址 |
| `MEDRAG_LLM_MODEL` | 服务实际提供的模型 ID |
| `MEDRAG_LLM_API_KEY` | 可选模型服务凭据，不提交到 Git |
| `MEDRAG_REQUEST_TIMEOUT` | 模型 HTTP 请求超时，默认 90 秒 |

更改 embedding 模型后必须建立新集合，并切换查询配置；不同模型的向量不能混用。导入时 `--image-dir` 必须与服务的 `MEDRAG_DATA_DIR/images` 一致。导入记录中的图片路径相对该目录保存。

## 验证与故障定位

1. `/dr/health` 报告 API 进程状态，不预先承诺模型后端已连通。
2. 在 `/dr/demo` 提交与知识库匹配的问题，确认 `mode=live`、`retrieval_backend=milvus_dense` 与 `generation_mode=vllm`。
3. 检查引用文件名、页码和图片是否能回到原文核对。
4. 用独立题集运行 [评估脚本](EVALUATION.md)，保存完整结果。

首次查询可能包含模型加载开销；评估延迟应说明是否预热及所用硬件。服务默认仅绑定本机；需要远程展示时，通过受控反向代理配置认证与 HTTPS。
