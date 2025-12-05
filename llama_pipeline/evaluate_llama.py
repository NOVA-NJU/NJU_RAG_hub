"""Evaluate the llama-index baseline using the benchmark_qa table and RAGAS metrics."""
from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

from datasets import Dataset
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from ragas import dataset_schema, evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from llama_pipeline.query_service import LlamaRAGService
else:  # pragma: no cover - normal package import
    from .query_service import LlamaRAGService


class SingleResponseChatModel(BaseChatModel):
    """LangChain chat model wrapper that never sends the 'n' parameter."""

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
        self._client = OpenAI(api_key=api_key, base_url=base_url)
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
        response = self._client.chat.completions.create(
            model=self._model,
            messages=payload,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        text = (response.choices[0].message.content or "").strip()
        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        return await asyncio.to_thread(self._generate, messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:  # noqa: D401
        return "single_response_openai_like"


def _gather_questions(limit: int | None, cursor: sqlite3.Cursor) -> List[tuple[str, str]]:
    cursor.execute("SELECT question, answer FROM benchmark_qa")
    rows = cursor.fetchall()
    if not rows:
        return []
    if limit and limit > 0:
        return rows[:limit]
    return rows


def _run_queries(questions: List[str], service: LlamaRAGService) -> List[Dict[str, Any] | None]:
    results: List[Dict[str, Any] | None] = []
    for question in tqdm(questions, desc="Running llama queries"):
        try:
            results.append(service.query(question))
        except Exception as exc:  # pragma: no cover - network/LLM issues
            print(f"Error processing question '{question}': {exc}")
            results.append(None)
    return results


def _serialize_terms(terms: Any) -> str:
    if isinstance(terms, str):
        return terms
    if isinstance(terms, Sequence):
        pieces: List[str] = []
        for term in terms:
            if term is None:
                continue
            text = str(term).strip()
            if text:
                pieces.append(text)
        return "; ".join(pieces)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the llama baseline with RAGAS")
    parser.add_argument("--limit", type=int, default=10, help="Number of benchmark questions to evaluate")
    parser.add_argument(
        "--output",
        default="ragas_evaluation_results_llama.csv",
        help="Path to save the evaluation results",
    )
    args = parser.parse_args()

    service = LlamaRAGService()
    settings = service.settings

    conn = sqlite3.connect(str(settings.storage.sqlite_path))
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='benchmark_qa'"
        )
        if not cursor.fetchone():
            print("benchmark_qa table not found. Please prepare benchmark data first.")
            return
        rows = _gather_questions(args.limit, cursor)
    finally:
        conn.close()

    if not rows:
        print("No benchmark data available.")
        return

    questions = [row[0] for row in rows]
    references = [row[1] for row in rows]

    results = _run_queries(questions, service)

    answers: List[str] = []
    contexts: List[List[str]] = []
    keyword_logs: List[str] = []
    llm_keyword_logs: List[str] = []
    heuristic_keyword_logs: List[str] = []
    valid_indices: List[int] = []
    for idx, result in enumerate(results):
        if not result:
            continue
        answers.append(result["answer"])
        contexts.append([source["content"] for source in result.get("sources", [])])
        combined_keywords = result.get("keywords")
        llm_keywords = result.get("llm_keywords")
        heuristic_keywords = result.get("heuristic_keywords")

        combined_str = _serialize_terms(combined_keywords)
        llm_str = _serialize_terms(llm_keywords)
        heuristic_str = _serialize_terms(heuristic_keywords)

        if not combined_str and llm_str:
            combined_str = llm_str
        elif not combined_str and heuristic_str:
            combined_str = heuristic_str

        keyword_logs.append(combined_str)
        llm_keyword_logs.append(llm_str)
        heuristic_keyword_logs.append(heuristic_str)
        valid_indices.append(idx)

    if not answers:
        print("Llama queries failed for all questions.")
        return

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
        raise ValueError("LLM_API_KEY is required for RAGAS evaluation")

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
    )

    evaluation = cast(
        dataset_schema.EvaluationResult,
        evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        ),
    )

    df = evaluation.to_pandas().reset_index(drop=True)
    expected = len(keyword_logs)
    if len(df) != expected:
        min_len = min(len(df), expected)
        print("Warning: row mismatch between metrics and keyword logs. Truncating to", min_len)
        df = df.iloc[:min_len].reset_index(drop=True)
        keyword_logs = keyword_logs[:min_len]
        llm_keyword_logs = llm_keyword_logs[:min_len]
        heuristic_keyword_logs = heuristic_keyword_logs[:min_len]

    df["keywords"] = keyword_logs
    df["llm_keywords"] = llm_keyword_logs
    df["heuristic_keywords"] = heuristic_keyword_logs
    df.to_csv(args.output, index=False)
    print(f"Evaluation completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()
