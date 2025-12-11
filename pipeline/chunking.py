"""语义切分工具，复刻 legacy agent 的分段策略。"""
from __future__ import annotations

import math
import re
import ast
from typing import List, Sequence
from config import load_settings
from openai import OpenAI

try:
    from llama_index.core.base.embeddings.base import BaseEmbedding  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from llama_index.core.embeddings.base import BaseEmbedding  # type: ignore
    except ImportError:  # pragma: no cover
        try:
            from llama_index.embeddings.base import BaseEmbedding  # type: ignore
        except ImportError:  # pragma: no cover
            class BaseEmbedding(object):  # type: ignore
                def get_text_embedding(self, text: str):  # noqa: D401
                    raise NotImplementedError

_SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?])\s*")

def _split_sentences(text: str, max_length: int = 8192, overlap: int = 100) -> List[str]:
    sentences = [segment.strip() for segment in _SENTENCE_PATTERN.split(text) if segment and segment.strip()]
    processed_sentences = []
    for sentence in sentences:
        if len(sentence) > max_length:
            processed_sentences.extend(split_long_sentence(sentence, max_length, overlap))
        else:
            processed_sentences.append(sentence)

    if processed_sentences:
        return processed_sentences
    return [text.strip()] if text else []

def split_long_sentence(sentence: str, max_length: int, overlap: int) -> List[str]:
    """
    将过长的句子按照指定的最大长度和重叠长度切分。

    Args:
        sentence: 待切分的句子。
        max_length: 每段的最大长度。
        overlap: 每段之间的重叠长度。

    Returns:
        切分后的句子列表。
    """
    if len(sentence) <= max_length:
        return [sentence]

    chunks = []
    start = 0
    while start < len(sentence):
        end = min(start + max_length, len(sentence))
        chunks.append(sentence[start:end])
        start = end - overlap if end - overlap > start else end

    return chunks

def chunking(
    text: str,
    embed_model: BaseEmbedding,
    method: str = "semantic_split",
    **kwargs,
) -> List[str]:
    """根据指定方法对文本进行切分。

    Args:
        text: 待切分文本。
        embed_model: 用于语义切分的嵌入模型实例。
        method: 切分方法，可选 "semantic_split" 或 "simple_split"或"llm_split"。
        **kwargs: 传递给切分方法的额外参数。
    """
    # 验证切分后的片段长度是否在范围内
    def _validate_chunks(chunks: List[str]):
        # 移除长度为0的片段
        chunks[:] = [chunk for chunk in chunks if len(chunk) > 0]
        for chunk in chunks:
            if not (1 <= len(chunk) <= 8192):
                raise ValueError(f"切分后的片段长度超出范围: {len(chunk)}")

    # 在每种切分方法返回结果后进行验证
    if method == "semantic_split":
        result = semantic_split(text, embed_model, **kwargs)
        _validate_chunks(result)
        return result
    elif method == "simple_split":
        result = simple_split(text, **kwargs)
        _validate_chunks(result)
        return result
    elif method == "llm_split":
        llm_model = kwargs.get("llm_model")  # 确保 llm_model 定义
        if not llm_model:
            raise ValueError("llm_split 方法需要提供有效的 llm_model 参数")
        result = llm_split(text, **kwargs)
        _validate_chunks(result)
        return result
    else:
        raise ValueError(f"未知的切分方法：{method}")

def semantic_split(
    text: str,
    embed_model: BaseEmbedding,
    threshold: float = 0.5,
    min_sentence_len: int = 20,
    max_chunk_len: int = 6000, 
    **kwargs# 新增参数：最大片段长度
) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    try:
        embeddings: List[Sequence[float]] = [embed_model.get_text_embedding(sentence) for sentence in sentences]
    except Exception:
        return [text]

    chunks: List[List[str]] = [[sentences[0]]]
    current_chunk_len = len(sentences[0])

    def _cosine(v1: Sequence[float], v2: Sequence[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if not norm1 or not norm2:
            return 0.0
        return dot / (norm1 * norm2)

    for idx in range(1, len(sentences)):
        prev_sentence = sentences[idx - 1]
        similarity = _cosine(embeddings[idx - 1], embeddings[idx])
        is_prev_short = len(prev_sentence) < min_sentence_len
        current_sentence_len = len(sentences[idx])

        if similarity < threshold and not is_prev_short or (current_chunk_len + current_sentence_len > max_chunk_len):
            chunks.append([sentences[idx]])
            current_chunk_len = current_sentence_len
        else:
            chunks[-1].append(sentences[idx])
            current_chunk_len += current_sentence_len

    return ["".join(chunk) for chunk in chunks if chunk]

def simple_split(
    text: str,
    **kwargs,
) -> List[str]:
    """简单按长度切分文本，每段可以有重叠部分，默认每 4000 字为一段，重叠 500 字。"""
    # 确保每段的最大长度不超过 8192
    max_len = min(kwargs.get("max_len", 6000), 8192)  # 每段的最大长度
    overlap = kwargs.get("overlap", 500)  # 重叠部分的长度

    if overlap >= max_len:
        raise ValueError("Overlap must be smaller than max_len")

    sentences = _split_sentences(text)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_len + sentence_len <= max_len:
            current_chunk.append(sentence)
            current_len += sentence_len
        else:
            if current_chunk:
                chunks.append("".join(current_chunk))
            # 创建新的块，包含重叠部分
            if overlap > 0 and current_chunk:
                current_chunk = current_chunk[-(overlap // len(current_chunk)):] 
            else:
                current_chunk = []
            current_chunk.append(sentence)
            current_len = sum(len(s) for s in current_chunk)

            # 确保重叠部分不会导致片段长度超出 max_len
            if sum(len(s) for s in current_chunk) > max_len:
                current_chunk = []

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks

def llm_split(
    text: str,
    llm_model: None,
    **kwargs,
) -> List[str]:
    """使用 LLM 进行文本切分，确保返回 Python list 格式。"""
    # 如果文本过长，调用 simple_split 和 llm_split 进行处理，确保返回一个扁平化列表
    if len(text.strip()) >= 50000:
        full_res = []
        res = simple_split(text, max_length=20000)
        for r in res:
            resp = llm_split(r, llm_model, **kwargs)
            full_res.extend(resp)  # 使用 extend 将结果扁平化
        return full_res
        
    title = kwargs.get("title", "无标题文档")
    prompt_template = kwargs.get(
        "prompt_template",
        """
        任务：将提供的文本切分为多个语义完整的片段。
        
        要求：
        1. 输出必须是纯粹的python列表格式，例如：["片段1", "片段2"]。
        2. 不要包含任何Markdown格式（如```json）。
        3. 切分后的片段应保持原文的连贯性和完整性。
        4. 如果文本较短或无法切分，请返回包含原始文本的数组。
        5. 针对包含背景介绍和后续分点阐述的文本，切分时请确保每个分点片段都包含必要的背景信息，使其在脱离原文后依然语义完整。如果分点依赖于前文的背景，请在片段中补充该背景。
        6. **严禁**将时间、地点、人物等元数据信息单独切分为一个片段。这些信息必须与它们描述的具体事件或内容合并在一起。
        7. 避免生成过短的片段（例如少于50个字符），除非原文确实无法合并。
        8.每一个分块严格不超过5000字符。

        文档标题：{title}
        待切分文本：
        {text}
        """
    )
    prompt = prompt_template.format(text=text, title=title)
    client = OpenAI(
        api_key = llm_model.api_key,
        base_url = llm_model.base_url,
    )
    response = client.chat.completions.create(
        model=llm_model.model,
        messages=[
            {"role": "system", "content": "你是一个文本处理专家。请直接输出python数组，不要有任何其他解释。"},
            {"role": "user","content": prompt}
        ],
        temperature=0.3,
    )

    # 解析返回值
    raw_output = response.choices[0].message.content if hasattr(response, "choices") else str(response)
    try:
        parsed_output = ast.literal_eval(raw_output)  # 安全地解析为 Python 对象
        if not isinstance(parsed_output, list):  # 验证是否为列表
            raise ValueError("LLM 返回的结果不是列表格式")
        return parsed_output
    except (SyntaxError, ValueError) as e:
        raise RuntimeError(f"无法解析 LLM 返回的结果：{raw_output}") from e
    

__all__ = ["chunking"]

