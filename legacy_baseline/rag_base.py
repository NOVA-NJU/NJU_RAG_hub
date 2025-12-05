import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
import openai
import uuid
import re
import json
import asyncio
from typing import List, Dict, Any, Optional, cast
import sqlite3

def table_exists(cursor, table_name):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cursor.fetchone() is not None

class RAGAgent:
    def __init__(self, 
                 description: str = "RAG Knowledge Base", 
                 # Embedding Config
                 embedding_api_key: Optional[str] = None,
                 embedding_base_url: Optional[str] = None,
                 embedding_model: Optional[str] = None,
                 # LLM Config
                 llm_api_key: Optional[str] = None,
                 llm_base_url: Optional[str] = None,
                 llm_model: Optional[str] = None,
                 # DB Config
                 db_path: str = "./sqlite_db/sqlite.db",
                 chroma_path: str = "./chroma_db"):
        load_dotenv()
        
        # Resolve Embedding Config
        self.embedding_api_key = embedding_api_key or os.getenv("EMBEDDING_API_KEY")
        self.embedding_base_url = embedding_base_url or os.getenv("EMBEDDING_BASE_URL")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL")
        
        # Resolve LLM Config
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY")
        self.llm_base_url = llm_base_url or os.getenv("LLM_BASE_URL")
        self.llm_model = llm_model or os.getenv("LLM_MODEL")
        
        if not self.embedding_model:
            raise ValueError("EMBEDDING_MODEL must be set")
        if not self.llm_model:
            raise ValueError("LLM_MODEL must be set")

        if os.path.dirname(db_path):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        if not table_exists(self.cursor, 'documents'):
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    content TEXT
                )
            """)
            self.conn.commit()
            
        self.chroma_path = chroma_path or "./chroma_db"
        os.makedirs(self.chroma_path, exist_ok=True)

        client = chromadb.PersistentClient(path=self.chroma_path)
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.embedding_api_key,
            api_base=self.embedding_base_url, 
            model_name=self.embedding_model
        )
        
        self.collection = client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=cast(Any, openai_ef),
            metadata={
                "description": description,
                "hnsw:space": "cosine"
            }
        )
        
        self.embedding_client = openai.OpenAI(
            api_key=self.embedding_api_key,
            base_url=self.embedding_base_url
        )
        
        self.llm_client = openai.OpenAI(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url
        )
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.embedding_model:
            raise ValueError("Embedding model is not configured")
            
        batch_size = 10
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.embedding_client.embeddings.create(
                input=batch,
                model=self.embedding_model
            )
            embeddings.extend([data.embedding for data in response.data])
        return embeddings

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = sum(a * a for a in v1) ** 0.5
        norm_v2 = sum(b * b for b in v2) ** 0.5
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def semantic_split(self, text: str, threshold: float = 0.5) -> List[str]:
        """基于语义相似度的文本切分"""

        sentences = re.split(r'(?<=[。！？!?])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        try:
            embeddings = self._get_embeddings(sentences)
        except Exception as e:
            print(f"Embedding error: {e}")
            return [text]

        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i-1], embeddings[i])
            is_prev_short = len(sentences[i-1]) < 20

            if sim < threshold and not is_prev_short:
                chunks.append("".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
                
        if current_chunk:
            chunks.append("".join(current_chunk))
            
        return chunks

    def add_document(self, content: str, title: str, time: str):
        """添加文档到知识库（按照语义切分）"""

        chunks = self.semantic_split(content)
        
        ids = []
        documents = []
        metadatas: List[Dict[str, Any]] = []
        
        self.cursor.execute(
            "INSERT INTO documents (title, content) VALUES (?, ?)",
            (title, content)
        )
        self.conn.commit()
        parent_id = str(self.cursor.lastrowid)
        
        for i, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk))
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({
                "topic": title, 
                "time": time,
                "chunk_index": i,
                "parent_id": parent_id
            })
            
        batch_size = 10
        if ids:
            for i in range(0, len(ids), batch_size):
                end_idx = i + batch_size
                self.collection.add(
                    ids=ids[i:end_idx],
                    documents=documents[i:end_idx],
                    metadatas=cast(Any, metadatas[i:end_idx])
                )
    
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict]:
        """检索相关文档"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["documents"] or not results["metadatas"] or not results["distances"]:
            return []
            
        return [{
            "content": doc,
            "metadata": meta,
            "similarity": 1-dist
        } for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )]
    
    def generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """基于检索到的文档生成回答"""
        if not self.llm_model:
            raise ValueError("LLM model is not configured")

        context = "\n".join([f"文档 {i+1}: {doc['content']}" 
                           for i, doc in enumerate(retrieved_docs)])
        
        prompt = f"""
        基于以下文档内容回答用户问题,回答不要无中生有，不要解释原因，尽可能简短并包含必要信息：
        
        相关文档:
        {context}
        
        用户问题: {query}
        
        请基于上述文档内容提供准确回答:
        """

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "你是一个基于文档内容回答问题的助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content or ""
    
    def query(self, question: str) -> Dict[str, Any]:
        """完整的 RAG 查询流程"""
        retrieved_docs = self.retrieve(question)
        answer = self.generate_response(question, retrieved_docs)
        
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_docs
        }

    async def async_retrieve(self, query: str, n_results: int = 5) -> List[Dict]:
        """异步检索相关文档"""
        loop = asyncio.get_running_loop()
        
        def _query():
            return self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
        results = await loop.run_in_executor(None, _query)
        
        if not results["documents"] or not results["metadatas"] or not results["distances"]:
            return []
            
        return [{
            "content": doc,
            "metadata": meta,
            "similarity": 1-dist
        } for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )]

    async def async_generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """异步基于检索到的文档生成回答"""
        if not self.llm_model:
            raise ValueError("LLM model is not configured")

        context = "\n".join([f"文档 {i+1}: {doc['content']}" 
                           for i, doc in enumerate(retrieved_docs)])
        
        prompt = f"""
        基于以下文档内容回答用户问题,回答不要无中生有，不要解释原因，尽可能简短并包含必要信息：
        
        相关文档:
        {context}
        
        用户问题: {query}
        
        请基于上述文档内容提供准确回答:
        """
        
        async_client = openai.AsyncOpenAI(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url
        )

        response = await async_client.chat.completions.create(
            model=cast(str, self.llm_model),
            messages=[
                {"role": "system", "content": "你是一个基于文档内容回答问题的助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content or ""

    async def async_query(self, question: str) -> Dict[str, Any]:
        """异步完整的 RAG 查询流程"""
        retrieved_docs = await self.async_retrieve(question)
        answer = await self.async_generate_response(question, retrieved_docs)
        
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_docs
        }

    async def generate_benchmark_dataset(self, num_questions_per_doc: int = 3, concurrency: int = 5):
        """基于sqlite.db中的文档生成基准测试数据集并存入数据库 (异步版本)"""
        if not self.llm_model:
            raise ValueError("LLM model is not configured")

        # Ensure benchmark table exists
        if not table_exists(self.cursor, 'benchmark_qa'):
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_qa (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER,
                    doc_title TEXT,
                    question TEXT,
                    answer TEXT,
                    FOREIGN KEY(doc_id) REFERENCES documents(id)
                )
            """)
            self.conn.commit()
        
        self.cursor.execute("SELECT id, title, content FROM documents")
        documents = self.cursor.fetchall()
        
        # Capture validated model name for inner function
        model_name = cast(str, self.llm_model)
        
        async_client = openai.AsyncOpenAI(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url
        )
        
        semaphore = asyncio.Semaphore(concurrency)
        db_lock = asyncio.Lock()
        
        # Shared state
        state = {
            "consecutive_errors": 0,
            "stop": False
        }

        async def process_doc(doc_id, title, content):
            if state["stop"]:
                return

            async with semaphore:
                if state["stop"]:
                    return
                
                async with db_lock:
                    self.cursor.execute("SELECT COUNT(*) FROM benchmark_qa WHERE doc_id = ?", (doc_id,))
                    count = self.cursor.fetchone()[0]
                
                if count >= num_questions_per_doc:
                    print(f"Skipping document {title} (already has QA pairs)")
                    return

                prompt = f"""
                基于以下文档内容，生成{num_questions_per_doc}个问答对。
                问题应该是具体的，并且可以从文档中找到答案。
                回答应该是简洁的。
                
                文档标题: {title}
                文档内容:
                {content[:2000]}
                
                请严格按照以下JSON格式返回结果，不要包含其他文字：
                [
                    {{
                        "question": "问题1",
                        "answer": "回答1"
                    }},
                    ...
                ]
                """
                
                try:
                    response = await async_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "你是一个专门用于生成问答数据集的助手。请只返回JSON格式的数据。"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7
                    )
                    
                    content_str = response.choices[0].message.content
                    if not content_str:
                        return
                        
                    # 尝试清理可能的markdown标记
                    content_str = content_str.strip()
                    if content_str.startswith("```json"):
                        content_str = content_str[7:]
                    elif content_str.startswith("```"):
                        content_str = content_str[3:]
                    if content_str.endswith("```"):
                        content_str = content_str[:-3]
                    
                    qa_pairs = json.loads(content_str.strip())
                    
                    async with db_lock:
                        for qa in qa_pairs:
                            self.cursor.execute(
                                "INSERT INTO benchmark_qa (doc_id, doc_title, question, answer) VALUES (?, ?, ?, ?)",
                                (doc_id, title, qa["question"], qa["answer"])
                            )
                        self.conn.commit()
                        
                    print(f"Generated and saved {len(qa_pairs)} QA pairs for document: {title}")
                    state["consecutive_errors"] = 0
                    
                except Exception as e:
                    print(f"Error generating QA for document {title}: {e}")
                    state["consecutive_errors"] += 1
                    if state["consecutive_errors"] >= 5:
                        print("Too many consecutive errors, stopping benchmark generation.")
                        state["stop"] = True

        tasks = [process_doc(doc[0], doc[1], doc[2]) for doc in documents]
        if tasks:
            await asyncio.gather(*tasks)
            
        print(f"Benchmark dataset generation completed and saved to database.")
