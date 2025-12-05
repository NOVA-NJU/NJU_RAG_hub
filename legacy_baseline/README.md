# RAG Knowledge Base Agent

这是一个基于 Python 的 超级轻量级的基础RAG (Retrieval-Augmented Generation) 知识库问答代理系统。它结合了向量数据库 (ChromaDB) 和大语言模型 (LLM)，能够基于本地文档提供问答服务。用于在NOVA社团中测试RAG模型效果，并探索优化流程

## 功能特性

*   **双模型支持**：支持独立配置 Embedding 模型和 LLM 模型（兼容 OpenAI 接口），可灵活组合不同厂商的服务。
*   **智能切分**：内置基于语义相似度的文本切分算法，相比传统按字符切分更能保留上下文语义。
*   **持久化存储**：
    *   向量数据存储于本地 ChromaDB。
    *   支持 SQLite 数据库连接（已集成初始化）。
*   **灵活配置**：支持通过环境变量 (`.env`) 或代码参数进行配置。

## 环境要求

*   Python 3.10+
*   主要依赖库：
    *   `chromadb`
    *   `openai`
    *   `python-dotenv`

## 快速开始

### 1. 安装依赖

```bash
pip install chromadb openai python-dotenv
```

### 2. 配置环境变量

在项目根目录下创建 `.env` 文件，填入您的 API 密钥和配置。系统支持为 Embedding 和 LLM 分别配置，也支持使用通用的 OpenAI 配置。

```ini
# --- Embedding 模型配置 (用于向量化) ---
EMBEDDING_API_KEY=sk-xxxx
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-v4

# --- LLM 模型配置 (用于生成回答) ---
LLM_API_KEY=sk-yyyy
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus


### 3. 使用示例

您可以参考 `Agent.py` 文件：

```python
import rag_base as rb


# 1. 初始化 Agent
# 如果已配置 .env，无需传入 key 和 url
agent = rb.RAGAgent(
    description="我的知识库示例"
)

# 或者显式传入参数
# agent = rb.RAGAgent(
#     description="我的知识库",
#     llm_api_key="sk-...",
#     llm_model="gpt-4"
# )

# 2. 添加文档
agent.add_document(content, title="测试文档", time="2025-12-01")

# 3. 进行问答
result = agent.query("文档的核心内容是什么？")

print("问题:", result["question"])
print("回答:", result["answer"])
print("参考来源:", result["sources"])
```

### 4. 运行

```bash
python Agent.py
```
## 项目详解和技术依赖

### 核心流程

1.  **初始化 (Initialization)**:
    *   加载 `.env` 环境变量配置。
    *   初始化 SQLite 数据库用于存储原始文档（表名 `documents`）。
    *   初始化 ChromaDB 客户端（持久化存储在 `./chroma_db`），创建名为 `knowledge_base` 的集合，使用余弦相似度 (`cosine`)。

2.  **文档处理 (Document Processing)**:
    *   **语义切分**: 使用正则表达式按句子分割文本，计算句子的 Embedding 向量，根据相邻句子的余弦相似度（阈值 0.5，可自行调整）动态合并成段落 (Chunk)。
    *   **双重存储**:
        *   原始完整文档存入 SQLite。
        *   切分后的 Chunk 及其 Embedding 向量存入 ChromaDB，附带元数据（标题、时间、父文档ID等）。

3.  **问答流程 (RAG Pipeline)**:
    *   **检索 (Retrieval)**: 将用户问题向量化，在 ChromaDB 中检索最相似的 Top-5 片段(可在retrieve函数定义中调整)。
    *   **生成 (Generation)**: 将检索到的片段作为上下文 (Context) 拼接到 Prompt 中，调用 LLM 生成回答。系统提示词限制模型仅基于文档回答，避免幻觉。

### 技术栈

*   **编程语言**: Python 3.10+
*   **向量数据库**: `chromadb` (本地轻量级向量库)
*   **大语言模型接口**: `openai` (支持 OpenAI 及兼容协议的模型，如通义千问、DeepSeek 等)
*   **配置管理**: `python-dotenv`
*   **数据存储**: `sqlite3` (内置，用于元数据和原始文档), `chromadb` (用于向量数据)
*   **文本处理**: `re` (正则), 用于分句



## 项目结构

*   `rag_base.py`: 核心代码，包含 `RAGAgent` 类定义。
*   `Agent.py`: 启动脚本和使用示例。
*   `chroma_db/`: 向量数据库存储目录（自动生成，请勿提交到 git）。
*   `sqlite_db/`: SQLite 数据库存储目录（自动生成，请勿提交到 git）。
*   `.env`: 配置文件（请勿提交到 git）。

## 许可证

MIT

```
