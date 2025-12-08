"""时间相关工具：元数据抽取 + 查询重写。"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")


def _find_years(text: str) -> List[int]:
	return [int(match) for match in YEAR_PATTERN.findall(text)]


def extract_temporal_metadata(text: str) -> Dict[str, int]:
	years = _find_years(text)
	if not years:
		return {}
	return {
		"earliest_year": min(years),
		"latest_year": max(years),
	}


_CHINESE_DIGITS = {
	"零": 0,
	"〇": 0,
	"一": 1,
	"二": 2,
	"两": 2,
	"三": 3,
	"四": 4,
	"五": 5,
	"六": 6,
	"七": 7,
	"八": 8,
	"九": 9,
}

_FULL_DATE_PATTERN = re.compile(r"((?:19|20)\d{2})[./-](\d{1,2})[./-](\d{1,2})")
_MONTH_DAY_PATTERN = re.compile(r"(\d{1,2})[./-](\d{1,2})")
_MONTH_DAY_CN_PATTERN = re.compile(r"(\d{1,2})月(\d{1,2})日?")
_CHINESE_MONTH_DAY_PATTERN = re.compile(
	r"([零〇一二两三四五六七八九十]{1,3})月([零〇一二两三四五六七八九十]{1,3})[日号天]?"
)


class TemporalQueryRewriter:
	"""针对模糊时间表达生成额外检索候选。"""

	def __init__(self, reference_year: Optional[int] = None, reference_today: Optional[date] = None) -> None:
		today = reference_today or datetime.now().date()
		self._reference_year = reference_year or today.year
		self._reference_today = today

	def extract_and_normalize_date(self, query: str) -> Optional[date]:
		text = query.strip()
		if not text:
			return None

		match = _FULL_DATE_PATTERN.search(text)
		if match:
			year = int(match.group(1))
			month = int(match.group(2))
			day = int(match.group(3))
			return self._safe_build_date(year, month, day)

		match = _MONTH_DAY_PATTERN.search(text)
		if match:
			month = int(match.group(1))
			day = int(match.group(2))
			return self._safe_build_date(self._reference_year, month, day)

		match = _MONTH_DAY_CN_PATTERN.search(text)
		if match:
			month = int(match.group(1))
			day = int(match.group(2))
			return self._safe_build_date(self._reference_year, month, day)

		match = _CHINESE_MONTH_DAY_PATTERN.search(text)
		if match:
			month = self._parse_chinese_number(match.group(1))
			day = self._parse_chinese_number(match.group(2))
			if month and day:
				return self._safe_build_date(self._reference_year, month, day)

		if any(token in text for token in ("本周", "这周", "本週")):
			return self._reference_today

		return None

	def get_week_range(self, input_date: date) -> Tuple[date, date]:
		start = input_date - timedelta(days=input_date.weekday())
		end = start + timedelta(days=6)
		return start, end

	def rewrite_for_interval_retrieval(self, original_query: str, extracted_date: date) -> List[str]:
		if not original_query:
			return []

		week_start, week_end = self.get_week_range(extracted_date)
		date_fragment = self._find_date_substring(original_query)
		base_suffix = self._strip_date_from_query(original_query, date_fragment)
		base_suffix = base_suffix or original_query

		queries: List[str] = []
		concise_range = f"{week_start.month}.{week_start.day}至{week_end.month}.{week_end.day}"
		range_cn = f"{week_start.month}月{week_start.day}日 至 {week_end.month}月{week_end.day}日"
		around_phrase = f"{extracted_date.month}月{extracted_date.day}日前后"
		week_index = self._week_index_in_month(extracted_date)
		week_label = f"{extracted_date.month}月第{week_index}周"

		if date_fragment:
			queries.append(original_query.replace(date_fragment, concise_range))
		queries.append(f"{range_cn} {base_suffix}".strip())
		queries.append(f"{week_label} {base_suffix}".strip())
		queries.append(f"{around_phrase} {base_suffix}".strip())
		queries.append(f"{extracted_date.month}月的{base_suffix}".strip())

		deduped: List[str] = []
		seen = set()
		for item in queries:
			cleaned = item.strip()
			if not cleaned or cleaned == original_query or cleaned in seen:
				continue
			seen.add(cleaned)
			deduped.append(cleaned)
		return deduped

	@staticmethod
	def _week_index_in_month(target: date) -> int:
		first_day = target.replace(day=1)
		offset = first_day.weekday()
		adjusted_day = target.day + offset
		return (adjusted_day - 1) // 7 + 1

	@staticmethod
	def _strip_date_from_query(query: str, fragment: Optional[str]) -> str:
		if fragment and fragment in query:
			return query.replace(fragment, "").strip()
		return query

	@staticmethod
	def _find_date_substring(query: str) -> Optional[str]:
		for pattern in (
			_FULL_DATE_PATTERN,
			_MONTH_DAY_PATTERN,
			_MONTH_DAY_CN_PATTERN,
			_CHINESE_MONTH_DAY_PATTERN,
		):
			match = pattern.search(query)
			if match:
				return match.group(0)
		return None

	@staticmethod
	def _parse_chinese_number(text: str) -> Optional[int]:
		if not text:
			return None
		total = 0
		temp = 0
		for char in text:
			if char == "十":
				temp = temp or 1
				temp *= 10
			else:
				digit = _CHINESE_DIGITS.get(char)
				if digit is None:
					return None
				temp += digit
		total += temp
		return total or None

	@staticmethod
	def _safe_build_date(year: int, month: int, day: int) -> Optional[date]:
		try:
			return date(year, month, day)
		except ValueError:
			return None


__all__ = ["extract_temporal_metadata", "TemporalQueryRewriter"]
