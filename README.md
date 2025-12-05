# NJU RAG Hub

将已有 RAG 体系的 legacy Agent 与全新 llama-index 流水线封装到同一仓库，复用一份 SQLite / Chroma 数据源，方便回归、基准对比与新特性迭代。未来会以 Llama 作为统一入口，按需载入不同的 RAG 模块，通过完备的检索、回答、评测工作流输送更可控、更理想的结果。

## 仓库结构

| 路径 | 说明 |
| --- | --- |
| `legacy_baseline/` | 原始 `rag_base.py` / `Agent.py` / `evaluate_rag.py`，作为对照基线。 |
| `llama_pipeline/` | llama-index 实现，切分、提示词、输出结构与 legacy 对齐。 |
| `sqlite_db/` `chroma_db/` | 共享的结构化与向量存储，可在 `.env` 中自定义路径。 |
| `requirements.txt` | 统一依赖清单。 |

## 快速上手

```powershell
git clone <repo>
cd NJU_RAG_hub
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> `.env` 至少需要：`EMBEDDING_API_KEY`, `EMBEDDING_MODEL`, `LLM_API_KEY`, `LLM_MODEL`, `SQLITE_PATH`, `CHROMA_PATH`。文档可直接写入 `sqlite_db/sqlite.db` 的 `documents` 表，或通过 `python -m llama_pipeline.ingest` 自动增量入库。

## 数据与索引生命周期

### Legacy baseline

```powershell
python legacy_baseline/Agent.py
python legacy_baseline/evaluate_rag.py --limit 10
```

- `rag_base.py` 负责语义切分与写入 SQLite/Chroma，返回结构为 `{"question","answer","sources"}`。
- TODO: 似乎在原有仓库更新了内容，需要整合

### Llama pipeline

```powershell
python llama_pipeline\ingest.py              # 增量刷新
python llama_pipeline\ingest.py --force      # 清空后全量重建
python llama_pipeline\ingest.py --doc-ids 12 45
```

- 复用 legacy 切分策略，自动生成 `doc_id/title/chunk_index` 及 `earliest_year/latest_year` 元数据。
- 默认写入 `.env` 中指定的 SQLite/Chroma 路径，可直接与 legacy 共用数据。
- 注意: 此文档写入方法并未测试，可能需要后续处理

## 查询工作流

```python
from llama_pipeline.query_service import LlamaRAGService

service = LlamaRAGService()
result = service.query("本周PBL学习组需要准备什么材料？")
print(result["answer"])
print(result["temporal_queries"], result["task_queries"], result["sources"])
```

- `.query_with_time_focus(...)`：在问题缺少年份时也强制启用时间过滤。
- `.async_query(...)`：在后台线程中运行，方便批量评测或并发请求。
- 返回体包含调试字段，便于排查召回路径与加权策略。

### 检索增强细节

| 模块 | 功能 | 触发条件 |
| --- | --- | --- |
| `extract_keywords` | LLM + 尾句启发式融合关键词用于 rerank。 | `ENABLE_FOCUS_HINTS=true` |
| `TemporalQueryRewriter` | 解析数字/中文日期并扩写为周区间、月份描述，加入检索队列；可配合元数据过滤。 | 自动启用；过滤由 `ENABLE_TIME_FILTERS` 决定 |
| `task_intent_tool` | 识别任务型问题，生成文档标题候选或 fallback 列表，弥补召回缺口。 | 任务/计划类问题自动触发 |

## 评测与对齐

- Legacy：`python legacy_baseline/evaluate_rag.py --limit 50`
- Llama：`python llama_pipeline/evaluate_llama.py --limit 50`

Llama 评测脚本特性：

1. 每个样本独立运行问答 + RAGAS，失败样本仅记录 `error`。
2. 输出 CSV 包含答案、RAGAS 指标以及 `keywords/temporal_queries/task_queries` 等上下文，方便对比分析。
3. 可在同一 benchmark CSV 上并排 legacy / llama 结果，快速评估修复收益。

## 环境变量

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `EMBEDDING_API_KEY` | 向量模型密钥 | 必填 |
| `EMBEDDING_BASE_URL` | 向量模型接口 | `https://api.openai.com/v1` |
| `EMBEDDING_MODEL` | 向量模型名称 | `text-embedding-3-large` |
| `LLM_API_KEY` | 生成模型密钥 | 必填 |
| `LLM_BASE_URL` | 生成模型接口 | `https://api.openai.com/v1` |
| `LLM_MODEL` | 生成模型名称 | `gpt-4o-mini` |
| `SQLITE_PATH` | SQLite 文件路径 | `./sqlite_db/sqlite.db` |
| `CHROMA_PATH` | Chroma 目录 | `./chroma_db` |
| `PROJECT_DESCRIPTION` | Chroma 集合描述 | `NJU RAG Hub` |
| `EMBEDDING_MAX_RETRIES` | Embedding 重试次数 | `3` |
| `EMBEDDING_RETRY_BACKOFF` | Embedding 重试初始间隔 | `1.5` |
| `ENABLE_FOCUS_HINTS` | 启用关键词聚焦 | `true` |
| `ENABLE_TIME_FILTERS` | 启用时间过滤 | `true` |
| `LLM_TEMPERATURE` | 生成随机性 | `0.7` |

## 贡献指南

1. 拉取最新 `main`，并确保 `pip install -r requirements.txt`。
2. 修改流程或依赖时同步更新本文档以及 `.env` 变量说明。
3. 提交 PR 前至少验证：
   - `python legacy_baseline/Agent.py` 能完成一次问答；
   - 若涉及评测，运行 `python llama_pipeline/evaluate_llama.py --limit 5`。

## 未来规划

- 以 Llama 为工作流大脑，按需装配不同检索器或 reranker，实现“模块即插件”的 RAG 体系。
- 扩展统一的评测与回归管线，自动比较各模块组合在准确率、覆盖率、延迟方面的表现。
- 将任务意图、时间推理等工具抽象为可复用接口，方便快速试验更细粒度的 RAG 方案。

### 说人话版本

设想的项目框架: 
```
NJU_RAG_Hub/
├── .env
├── requirements.txt
├── config.py                       # 从.env加载配置，初始化全局Settings
├── main.py                         # 主入口
├── pipeline/                       # 核心流水线（预处理 + 查询路由）
│   ├── __init__.py
│   ├── ingest.py                   # 预处理入库
│   ├── query_router.py
│   └── base_rag.py                 # 定义统一的RAG接口
├── rag_architectures/              # RAG架构实现
│   ├── __init__.py
│   ├── standard_rag.py             # 标准RAG
│   ├── graph_rag.py                # GraphRAG实现
│   └── light_rag.py                # LightRAG实现
|   └── ...
├── tools/                          # 增强工具集
│   ├── __init__.py
│   ├── temporal.py                 # 时间推理
│   ├── keywords.py                 # 关键词提取
│   └── ...                         # 其他现有工具
├── scripts/
│   ├── evaluate.py                 # 主评测脚本
│   └── ...                         # 其他脚本 e.g.导入数据，对比测评
├── chroma_db/                      # Chroma向量库
└── sqlite.db/                      # SQLite数据库
```

#### 工作流程
##### 输入

文档->调用导入数据脚本/其他组的数据库存储方案

##### 测试

调用测评脚本，指定使用的模型参数 e.g.单light，单graph，light与graph对比

##### 输出

问题->main.py->pipeline预处理->RAG检索->返回答案

#### 个人建议

由于这个框架并没有真正落实，而且也只有我目前这个命名诡异、结构混乱的项目，建议使用Copilot等重构一次
