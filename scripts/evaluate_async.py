"""异步评测脚本，基于 benchmark_qa 表并发运行 RAG 查询，然后使用 RAGAS 评测。"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from datasets import Dataset
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from ragas import aevaluate, dataset_schema
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from tqdm.asyncio import tqdm_asyncio

from config import load_settings
from pipeline.query_router import build_rag


class SingleResponseChatModel(BaseChatModel):
    """LangChain ChatModel，强制单次回复以兼容部分供应商。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> None:
        super().__init__()
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @staticmethod
    def _convert_message(message: BaseMessage) -> Dict[str, str]:
        role = {
            "system": "system",
            "human": "user",
            "ai": "assistant",
        }.get(message.type, "user")
        content = message.content if isinstance(message.content, str) else str(message.content)
        return {"role": role, "content": content}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = [self._convert_message(msg) for msg in messages]
        raise NotImplementedError("SingleResponseChatModel only supports async generation via _agenerate")
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = [self._convert_message(msg) for msg in messages]
        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=payload,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        text = (response.choices[0].message.content or "").strip()
        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:  # noqa: D401
        return "single_response_openai_like"


@dataclass
class EvaluationResult:
    success: int
    failed: int
    output_path: Path


def _gather_questions(limit: int, cursor: sqlite3.Cursor) -> List[tuple[str, str]]:
    cursor.execute("SELECT question, answer FROM benchmark_qa")
    rows = cursor.fetchall()
    if not rows:
        return []
    if limit > 0:
        return rows[:limit]
    return rows


def _serialize_terms(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return ""


async def _process_single_question(
    rag, question: str, semaphore: asyncio.Semaphore
) -> tuple[Dict[str, Any], str] | None:
    async with semaphore:
        try:
            result = await rag.async_query(question)
            keywords = _serialize_terms(result.get("keywords"))
            return result, keywords
        except Exception as exc:
            logger = logging.getLogger(__name__)
            logger.error("问题 '%s' 评测失败: %s", question, exc)
            return None


async def run_evaluation_async(
    *,
    architecture: str = "standard",
    limit: int = 10,
    output: str = "ragas_results.csv",
    concurrency: int = 5,
) -> EvaluationResult:
    rag = build_rag(architecture)
    settings = rag.settings

    conn = sqlite3.connect(str(settings.storage.sqlite_path))
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='benchmark_qa'")
        if not cursor.fetchone():
            raise RuntimeError("benchmark_qa 表不存在，请先生成基准数据。")
        rows = _gather_questions(limit, cursor)
    finally:
        conn.close()

    if not rows:
        raise RuntimeError("benchmark_qa 表为空，无法评测。")

    questions = [row[0] for row in rows]
    references = [row[1] for row in rows]

    # 并发查询
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_process_single_question(rag, q, semaphore) for q in questions]
    
    results_with_keywords = await tqdm_asyncio.gather(*tasks, desc="Async Evaluating", unit="question")

    answers: List[str] = []
    contexts: List[List[str]] = []
    keyword_logs: List[str] = []
    valid_indices: List[int] = []

    for idx, item in enumerate(results_with_keywords):
        if item is None:
            continue
        
        result, keywords = item
        answers.append(result.get("answer", ""))
        contexts.append([source.get("content", "") for source in result.get("sources", [])])
        keyword_logs.append(keywords)
        valid_indices.append(idx)

    if not answers:
        raise RuntimeError("所有问题均调用失败，无法评测。")

    filtered_questions = [questions[i] for i in valid_indices]
    filtered_references = [references[i] for i in valid_indices]

    dataset = Dataset.from_dict(
        {
            "question": filtered_questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": filtered_references,
        }
    )

    if not settings.llm.api_key:
        raise ValueError("LLM_API_KEY 缺失，无法运行 RAGAS 评测。")

    llm = SingleResponseChatModel(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        temperature=0.0,
    )
    embeddings = OpenAIEmbeddings(
        model=settings.embedding.model,
        api_key=settings.embedding.api_key,
        base_url=settings.embedding.base_url,
        check_embedding_ctx_length=False,
    )

    # RAGAS 异步评测
    evaluation = await aevaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    result_cls = dataset_schema.EvaluationResult
    if isinstance(evaluation, result_cls):
        result = evaluation
    elif hasattr(result_cls, "model_validate"):
        result = result_cls.model_validate(evaluation)
    elif hasattr(result_cls, "parse_obj"):
        result = result_cls.parse_obj(evaluation)
    else:
        result = result_cls(**evaluation)
    
    df = result.to_pandas().reset_index(drop=True)
    
    # 对齐 keyword_logs
    expected = len(keyword_logs)
    if len(df) != expected:
        min_len = min(len(df), expected)
        df = df.iloc[:min_len].reset_index(drop=True)
        keyword_logs = keyword_logs[:min_len]

    df["keywords"] = keyword_logs
    output_path = Path(output).resolve()
    df.to_csv(output_path, index=False)

    return EvaluationResult(success=len(df), failed=len(questions) - len(df), output_path=output_path)


async def main_async(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Async Evaluate a RAG architecture using benchmark_qa")
    parser.add_argument("--arch", default="standard", help="架构名称，默认 standard")
    parser.add_argument("--limit", type=int, default=10, help="评测样本数")
    parser.add_argument("--output", default="ragas_results.csv", help="结果 CSV 路径")
    parser.add_argument("--concurrency", type=int, default=5, help="并发请求数")
    args = parser.parse_args(argv)

    result = await run_evaluation_async(
        architecture=args.arch, 
        limit=args.limit, 
        output=args.output,
        concurrency=args.concurrency
    )
    print(
        "评测完成: 成功 %d 条，失败 %d 条，结果写入 %s"
        % (result.success, result.failed, result.output_path)
    )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
