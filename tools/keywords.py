"""LLM 驱动的关键词提取工具。"""
from __future__ import annotations

import json
from typing import List

from openai import OpenAI

_KEYWORD_SYSTEM_PROMPT = (
	"你是一名高校知识库检索优化助手,负责生成结构化关键词。"
	"所有输出必须严格符合 JSON 语法,且禁止添加额外文字。"
)

_KEYWORD_USER_TEMPLATE = (
	"请从以下问题中提取最能提升检索效果的关键信息点,并按照示例返回 JSON:\n"
	'{{"keywords": ["关键词1", "关键词2"]}}\n'
	"要求:\n"
	"1. 最多 {max_kw} 个条目,可以少于该数量。\n"
	"2. 仅使用名词或短语,每个条目 2~8 个汉字或 1~3 个英文单词。\n"
	"3. 输出不得包含任何解释或额外字段。\n\n"
	"问题: {question}"
)


def _extract_json_snippet(text: str) -> str:
	text = text.strip()
	if not text:
		return text
	start = text.find("{")
	end = text.rfind("}")
	if start != -1 and end != -1 and end > start:
		return text[start : end + 1]
	start = text.find("[")
	end = text.rfind("]")
	if start != -1 and end != -1 and end > start:
		return text[start : end + 1]
	return text


def extract_keywords(
	question: str,
	*,
	llm_client: OpenAI,
	model: str,
	max_keywords: int = 3,
) -> List[str]:
	user_prompt = _KEYWORD_USER_TEMPLATE.format(max_kw=max_keywords, question=question.strip())

	response = llm_client.chat.completions.create(
		model=model,
		messages=[
			{"role": "system", "content": _KEYWORD_SYSTEM_PROMPT},
			{"role": "user", "content": user_prompt},
		],
		temperature=0.0,
		max_tokens=150,
	)
	content = (response.choices[0].message.content or "").strip()
	payload = _extract_json_snippet(content)
	try:
		data = json.loads(payload)
		if isinstance(data, dict):
			values = data.get("keywords") or data.get("关键词")
			if isinstance(values, list):
				keywords = [str(item).strip() for item in values if str(item).strip()]
				if keywords:
					return keywords[:max_keywords]
		if isinstance(data, list):
			keywords = [str(item).strip() for item in data if str(item).strip()]
			if keywords:
				return keywords[:max_keywords]
	except json.JSONDecodeError:
		pass

	parts = [part.strip() for part in content.replace(";", "\n").replace(",", "\n").split("\n")]
	return [part for part in parts if part][:max_keywords]


__all__ = ["extract_keywords"]
