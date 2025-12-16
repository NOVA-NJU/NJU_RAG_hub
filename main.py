"""命令行入口：统一调度 ingest/query/eval。"""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from pipeline.ingest import run_ingest
from pipeline.query_router import available_architectures, build_rag
from scripts.evaluate import run_evaluation
from scripts.evaluate_async import run_evaluation_async
import asyncio


def _handle_ingest(args: argparse.Namespace) -> None:
	extra = []
	if args.force:
		extra.append("--force")
	if args.doc_ids:
		extra.append("--doc-ids")
		extra.extend(args.doc_ids)
	if args.doc_id_range:
		extra.append("--doc-id-range")
		extra.extend(map(str, args.doc_id_range))
	if args.no_progress:
		extra.append("--no-progress")
	run_ingest(extra or None)


def _handle_query(args: argparse.Namespace) -> None:
	rag = build_rag(args.arch)
	result = rag.query(args.question, enforce_time=args.enforce_time)
	print(json.dumps(result, ensure_ascii=False, indent=2))


def _handle_eval(args: argparse.Namespace) -> None:
	result = run_evaluation(architecture=args.arch, limit=args.limit, output=args.output)
	print(
		"评测完成: 成功 %d 条，失败 %d 条，结果写入 %s"
		% (result.success, result.failed, result.output_path)
	)


def _handle_eval_async(args: argparse.Namespace) -> None:
	result = asyncio.run(run_evaluation_async(
		architecture=args.arch,
		limit=args.limit,
		output=args.output,
		concurrency=args.concurrency
	))
	print(
		"评测完成: 成功 %d 条，失败 %d 条，结果写入 %s"
		% (result.success, result.failed, result.output_path)
	)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="NJU RAG Hub Next CLI")
	subparsers = parser.add_subparsers(dest="command", required=True)

	ingest_parser = subparsers.add_parser("ingest", help="构建或刷新向量索引")
	ingest_parser.add_argument("--force", action="store_true", help="全量重建索引")
	ingest_parser.add_argument("--doc-ids", nargs="+", help="只刷新指定文档 ID")
	ingest_parser.add_argument("--doc-id-range", nargs=2, type=int, metavar=("START", "END"), help="指定文档 ID 的范围（包含起始和结束 ID）")
	ingest_parser.add_argument("--no-progress", action="store_true", help="关闭进度条")
	ingest_parser.set_defaults(func=_handle_ingest)

	query_parser = subparsers.add_parser("query", help="运行单次问答")
	query_parser.add_argument("question", help="提问内容")
	query_parser.add_argument("--arch", default="standard", choices=available_architectures(), help="RAG 架构")
	query_parser.add_argument("--enforce-time", action="store_true", help="强制启用时间过滤")
	query_parser.set_defaults(func=_handle_query)

	eval_parser = subparsers.add_parser("eval", help="基于 benchmark_qa 运行评测")
	eval_parser.add_argument("--arch", default="standard", choices=available_architectures(), help="RAG 架构")
	eval_parser.add_argument("--limit", type=int, default=10, help="评测样本数")
	eval_parser.add_argument("--output", default="ragas_results.csv", help="输出 CSV 路径")
	eval_parser.set_defaults(func=_handle_eval)

	eval_async_parser = subparsers.add_parser("eval-async", help="基于 benchmark_qa 运行异步评测")
	eval_async_parser.add_argument("--arch", default="standard", choices=available_architectures(), help="RAG 架构")
	eval_async_parser.add_argument("--limit", type=int, default=10, help="评测样本数")
	eval_async_parser.add_argument("--output", default="ragas_results.csv", help="输出 CSV 路径")
	eval_async_parser.add_argument("--concurrency", type=int, default=5, help="并发请求数")
	eval_async_parser.set_defaults(func=_handle_eval_async)

	return parser


def main(argv: Sequence[str] | None = None) -> None:
	parser = build_parser()
	args = parser.parse_args(argv)
	args.func(args)


if __name__ == "__main__":
	main()
