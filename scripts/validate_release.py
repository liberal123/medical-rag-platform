"""Run local engineering checks and preserve their actual outputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for filename in ('unittest.log', 'api_smoke.json', 'local_validation.json', 'demo_eval.jsonl', 'demo_eval.summary.json', 'VALIDATION.md'):
        if (output / filename).exists():
            raise SystemExit('Choose a new output directory; existing validation files are preserved.')
    env = dict(os.environ, MEDRAG_MODE='demo', PYTHONIOENCODING='utf-8')
    env.pop('MEDRAG_DEMO_KNOWLEDGE', None)
    env.pop('MEDRAG_DATA_DIR', None)
    def run(command):
        result = subprocess.run([sys.executable, '-X', 'utf8', *command], cwd=ROOT, env=env, text=True, encoding='utf-8', capture_output=True)
        if result.returncode:
            print(result.stdout + result.stderr)
            raise SystemExit(result.returncode)
        return result
    tests = run(['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
    log = tests.stdout + tests.stderr
    (output / 'unittest.log').write_text(log, encoding='utf-8')
    count = int(re.search(r'Ran (\d+) tests', log).group(1))
    skipped_match = re.search(r'skipped=(\d+)', log)
    skipped = int(skipped_match.group(1)) if skipped_match else 0
    run(['scripts/prepare_demo.py'])
    for module in ('ingest', 'build_index', 'evaluate'):
        run(['-m', 'medical_rag.' + module, '--help'])
    eval_path = output / 'demo_eval.jsonl'
    run(['-m', 'medical_rag.evaluate', '--input', 'examples/eval_demo.jsonl', '--output', str(eval_path), '--mode', 'demo'])
    evaluation = json.loads((output / 'demo_eval.summary.json').read_text(encoding='utf-8'))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'medical_rag.app:app', '--host', '127.0.0.1', '--port', str(port)], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=10) as client:
            deadline = time.monotonic() + 20
            while True:
                try:
                    health = client.get('/dr/health')
                    health.raise_for_status()
                    break
                except httpx.HTTPError:
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError('Local HTTP service failed to start')
                    time.sleep(0.15)
            page = client.get('/dr/demo')
            assert page.status_code == 200 and '智能医疗问答平台' in page.text
            response = client.post('/dr/chat', json={'question':'文献检索如何工作？'})
            response.raise_for_status()
            body = response.json()
            assert body['mode'] == 'demo' and body['generation_mode'] == 'extractive_demo'
            assert body['retrieval_backend'] == 'lexical_demo' and body['references']
            assert body['latency_seconds'] >= 0
            images = [ref['image_url'] for ref in body['references'] if ref.get('image_url')]
            assert images, 'Expected a workflow illustration citation'
            for url in images:
                image = client.get(url)
                assert image.status_code == 200 and '<svg' in image.text
            invalid = client.post('/dr/chat', json={'question':'   '})
            assert invalid.status_code == 422
            api_result = {'health':health.json(), 'page_status':page.status_code, 'chat_status':response.status_code, 'response':body, 'image_count_verified':len(images), 'invalid_question_status':invalid.status_code}
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
    (output / 'api_smoke.json').write_text(json.dumps(api_result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    packages = {name:importlib.metadata.version(name) for name in ('fastapi','uvicorn','httpx','pydantic','PyMuPDF','python-dotenv')}
    tracked_inputs = [*ROOT.glob('src/**/*.py'), *ROOT.glob('tests/*.py'), *ROOT.glob('web/*.html'), *ROOT.glob('examples/*'), ROOT / 'requirements.txt']
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tracked_inputs) if p.is_file()}
    report = {'created_at_utc':datetime.now(timezone.utc).isoformat(), 'python':platform.python_version(), 'platform':platform.system()+' '+platform.release(), 'packages':packages, 'tests':{'run':count,'skipped':skipped,'failures':0,'errors':0}, 'http_smoke':'passed', 'demo_evaluation':{'total':evaluation['total_count'],'success':evaluation['success_count'],'errors':evaluation['error_count']}, 'gpu_live_validation':False, 'browser_visual_validation':False, 'source_sha256':hashes}
    (output / 'local_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    markdown = f'''# 本地工程验证

执行时间（UTC）：{report['created_at_utc']}。环境：{report['platform']}，Python {report['python']}。

| 检查 | 实际结果 |
| --- | --- |
| 单元与集成测试 | 运行 {count} 项，失败 0，错误 0，跳过 {skipped} |
| 真实 PDF 临时样本 | 文字切块、页码、嵌入图片、无文字扫描页处理通过 |
| 本地 uvicorn HTTP 服务 | 页面、健康状态、问答、图片和非法参数响应通过 |
| demo 逐题回归 | {evaluation['success_count']}/{evaluation['total_count']} 正常返回 |
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
'''
    (output / 'VALIDATION.md').write_text(markdown, encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
