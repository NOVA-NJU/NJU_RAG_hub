"""LLM-assisted helpers to focus retrieval on task/plan documents."""
from __future__ import annotations

import json
from typing import List

from openai import OpenAI

_TASK_SYSTEM_PROMPT = (
    "你是一个教务任务检索助手,需要把用户的问题映射到文档标题或短语。"
    "输出必须是 JSON, 且只能包含 queries 字段,值为字符串数组。"
)

_TASK_USER_TEMPLATE = (
    "请阅读用户问题,提取其中涉及的学习任务/安排主题,返回最多 {max_items} 个最关键的检索短语。\n"
    "示例输出: {{\"queries\": [\"本周PBL学习组任务安排\", \"PBL学习组任务清单\"]}}\n"
    "要求:\n"
    "1. 每个短语 4~12 个汉字,集中在任务/安排/值班/学习组等实体。\n"
    "2. 禁止添加解释或额外字段,也不要复述原问题。\n"
    "3. 优先提及文档标题或常用栏目名(例如: 任务安排, 值班计划, PBL学习组任务)。\n\n"
    "问题: {question}"
)

_TASK_KEYWORDS = ["任务", "安排", "学习组", "PBL", "值班", "计划", "周"]


def looks_like_task_question(question: str) -> bool:
    text = question.lower()
    if any(token in question for token in _TASK_KEYWORDS):
        return True
    return "task" in text


def _parse_queries_from_json(payload: str, *, max_items: int) -> List[str]:
    data = json.loads(payload)
    values = []
    if isinstance(data, dict):
        values = data.get("queries") or data.get("keywords") or []
    elif isinstance(data, list):
        values = data
    if not isinstance(values, list):
        return []
    cleaned: List[str] = []
    for item in values:
        text = str(item).strip()
        if text:
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def generate_task_queries(
    question: str,
    *,
    llm_client: OpenAI,
    model: str,
    max_items: int = 3,
) -> List[str]:
    user_prompt = _TASK_USER_TEMPLATE.format(max_items=max_items, question=question.strip())
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _TASK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    content = (response.choices[0].message.content or "").strip()
    try:
        queries = _parse_queries_from_json(content, max_items=max_items)
        if queries:
            return queries
    except json.JSONDecodeError:
        pass
    return []


def fallback_task_queries(question: str) -> List[str]:
    seeds: List[str] = []
    lowered = question.lower()
    core = question.replace("本周", "").replace("这周", "").strip() or question
    if "pbl" in lowered:
        seeds.append("PBL学习组任务安排")
        seeds.append("PBL学习组任务清单")
    if "学习组" in question:
        seeds.append("学习组任务安排")
    if "值班" in question:
        seeds.append("值班计划")
    if "任务" in question:
        seeds.append("本周任务安排")
    seeds.append(f"{core} 任务安排")

    deduped: List[str] = []
    seen = set()
    for phrase in seeds:
        cleaned = phrase.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped[:3]

