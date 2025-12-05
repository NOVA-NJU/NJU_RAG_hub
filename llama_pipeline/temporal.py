"""Extract temporal metadata from text chunks."""
from __future__ import annotations

import re
from typing import Dict, List

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
