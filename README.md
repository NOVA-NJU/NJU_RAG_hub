## NJU RAG Hub (Next)

本目录是对 `NJU_RAG_hub` 的重构实现，将原先的 legacy agent 与 llama-index 流水线抽象为可配置、可插拔的 RAG 平台。目标：**一份配置 + 一套存储 + 多种检索/生成策略**。

### 顶层结构

```
NJU_RAG_hub_new/
├── config.py              # 从 .env 读取并校验全局配置
├── main.py                # CLI 入口，统一路由到 ingest / query / eval
├── pipeline/
│   ├── base_rag.py        # RAG 接口约束 + 工具函数
│   ├── ingest.py          # 文档入库、索引构建
│   └── query_router.py    # 根据路由策略实例化具体 RAG
├── rag_architectures/
│   ├── standard_rag.py    # llama-index + Chroma 的默认实现
│   ├── graph_rag.py       # 占位，待接入 GraphRAG
│   └── light_rag.py       # 占位，待接入轻量 RAG
├── tools/
│   ├── keywords.py        # 关键词提取
│   └── temporal.py        # 时间推理/重写
└── scripts/
	└── evaluate.py        # 统一评测流程
```

### 设计原则

1. **配置集中**：所有 API Key、模型、存储路径、运行开关统一由 `config.py` 暴露 `Settings`。
2. **接口稳定**：`pipeline.base_rag.BaseRAG` 规定了 `ingest`、`query`、`async_query` 等必要方法，方便测试与热插拔。
3. **共享存储**：默认沿用 `sqlite_db/sqlite.db` + `chroma_db/`，支持在 `.env` 中重写路径。
4. **增强模块化**：关键词提取、时间重写、任务意图扩写等能力收敛到 `tools/`，避免散落。
5. **单一入口**：`python -m main ingest/query/eval` 即可完成常见操作，降低上手成本。

### 当前进度

- [x] `config.py` 暴露 `Settings`/`load_settings`。
- [x] `pipeline/` 已迁移语义切分、索引构建与路由。
- [x] `standard_rag.py` 对齐 llama-index 实现，GraphRAG / LightRAG 仍为占位。
- [x] `scripts/evaluate.py` 整合 RAGAS + 关键词日志。
- [x] `requirements.txt` 精简依赖。

### 运行方式

```powershell
python main.py ingest --force         # 全量重建索引
python main.py query "本周PBL学习组需要准备什么？"
python main.py eval --limit 20 --output results.csv
```

如需后续扩展（GraphRAG、LightRAG 等），只需在 `rag_architectures/` 中新增实现并调用 `register_architecture` 注册即可。

### 环境配置

1. 在仓库根目录创建 `.env`，至少包含：
	```ini
	EMBEDDING_API_KEY=xxx
	EMBEDDING_MODEL=text-embedding-3-large
	LLM_API_KEY=xxx
	LLM_MODEL=gpt-4o-mini
	SQLITE_PATH=./sqlite_db/sqlite.db
	CHROMA_PATH=./chroma_db
	```
	其余可选项：`EMBEDDING_BASE_URL`、`LLM_BASE_URL`、`SIMILARITY_TOP_K`、`ANSWER_CONTEXT_K`、`ENABLE_FOCUS_HINTS`、`ENABLE_TIME_FILTERS`、`LLM_TEMPERATURE` 等。
2. 执行 `pip install -r requirements.txt` 安装依赖。
3. 若需本地数据库/向量库，请确保 `sqlite_db/` 与 `chroma_db/` 目录可写，或在 `.env` 中指向自定义路径。
