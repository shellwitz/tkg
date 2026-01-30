import calendar
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from dateutil import parser as date_parser


@dataclass
class TimestampRange:
    start_date: Optional[str]
    end_date: Optional[str]


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_YEAR_MONTH_RE = re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})$")
_YEAR_RE = re.compile(r"^\d{4}$")
_QUARTER_RE = re.compile(
    r"^(?:Q(?P<q1>[1-4])\s*(?P<y1>\d{4})|(?P<y2>\d{4})-Q(?P<q2>[1-4])|(?P<y3>\d{4})\s*Q(?P<q3>[1-4]))$",
    re.IGNORECASE,
)
_MONTH_ONLY_RE = re.compile(r"^[A-Za-z]+$")


def _month_range(year: int, month: int) -> Tuple[str, str]:
    end_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{end_day:02d}"


def _year_range(year: int) -> Tuple[str, str]:
    return f"{year}-01-01", f"{year}-12-31"


def _quarter_range(year: int, quarter: int) -> Tuple[str, str]:
    start_month = 3 * (quarter - 1) + 1
    end_month = start_month + 2
    end_day = calendar.monthrange(year, end_month)[1]
    return f"{year}-{start_month:02d}-01", f"{year}-{end_month:02d}-{end_day:02d}"


def _parse_quarter(text: str) -> Optional[Tuple[int, int]]:
    match = _QUARTER_RE.match(text)
    if not match:
        return None
    quarter = match.group("q1") or match.group("q2") or match.group("q3")
    year = match.group("y1") or match.group("y2") or match.group("y3")
    if not quarter or not year:
        return None
    return int(year), int(quarter)


def _parse_month_only(text: str) -> Optional[int]:
    text = text.strip()
    if not _MONTH_ONLY_RE.match(text):
        return None
    try:
        dt = date_parser.parse(text, default=datetime(2000, 1, 1))
    except (ValueError, OverflowError):
        return None
    return dt.month


def _parse_month_year(text: str) -> Optional[Tuple[int, int]]:
    if not re.search(r"\d{4}", text):
        return None
    try:
        dt = date_parser.parse(text, default=datetime(2000, 1, 1))
    except (ValueError, OverflowError):
        return None
    return dt.year, dt.month


def _parse_single_bound(text: str, is_start: bool) -> Optional[str]:
    text = text.strip()
    if not text:
        return None
    if _ISO_DATE_RE.match(text):
        return text

    quarter = _parse_quarter(text)
    if quarter:
        year, q = quarter
        start, end = _quarter_range(year, q)
        return start if is_start else end

    ym_match = _ISO_YEAR_MONTH_RE.match(text)
    if ym_match:
        year = int(ym_match.group("y"))
        month = int(ym_match.group("m"))
        if 1 <= month <= 12:
            start, end = _month_range(year, month)
            return start if is_start else end
        return None

    if _YEAR_RE.match(text):
        year = int(text)
        start, end = _year_range(year)
        return start if is_start else end

    month_year = _parse_month_year(text)
    if month_year:
        year, month = month_year
        start, end = _month_range(year, month)
        return start if is_start else end

    return None


def _split_to_range(text: str) -> Optional[Tuple[str, str]]:
    match = re.match(r"^(?P<start>.+?)\s+to\s+(?P<end>.+)$", text, re.IGNORECASE)
    if match:
        return match.group("start"), match.group("end")
    match = re.match(r"^(?P<start>.+?)\s+to\s*$", text, re.IGNORECASE)
    if match:
        return match.group("start"), ""
    match = re.match(r"^\s*to\s+(?P<end>.+)$", text, re.IGNORECASE)
    if match:
        return "", match.group("end")
    return None


def parse_timestamp_range(name: str) -> TimestampRange:
    name = name.strip()
    if not name:
        return TimestampRange(None, None)

    if _ISO_DATE_RE.match(name):
        return TimestampRange(name, name)

    ym_match = _ISO_YEAR_MONTH_RE.match(name)
    if ym_match:
        start = _parse_single_bound(name, True)
        end = _parse_single_bound(name, False)
        return TimestampRange(start, end)

    if _YEAR_RE.match(name):
        start = _parse_single_bound(name, True)
        end = _parse_single_bound(name, False)
        return TimestampRange(start, end)

    year_range = re.match(r"^(?P<y1>\d{4})\s*(?:-\s*|to\s+)(?P<y2>\d{4})$", name)
    if year_range:
        y1 = int(year_range.group("y1"))
        y2 = int(year_range.group("y2"))
        start, _ = _year_range(y1)
        _, end = _year_range(y2)
        return TimestampRange(start, end)

    month_range_single_year = re.match(
        r"^(?P<m1>[A-Za-z]+)\s*(?:-\s*|to\s+)(?P<m2>[A-Za-z]+)\s*(?P<y>\d{4})$",
        name,
        re.IGNORECASE,
    )
    if month_range_single_year:
        m1 = _parse_month_only(month_range_single_year.group("m1"))
        m2 = _parse_month_only(month_range_single_year.group("m2"))
        year = int(month_range_single_year.group("y"))
        if m1 and m2:
            start, _ = _month_range(year, m1)
            _, end = _month_range(year, m2)
            return TimestampRange(start, end)

    month_range_dual_year = re.match(
        r"^(?P<m1>[A-Za-z]+)\s+(?P<y1>\d{4})\s*(?:-\s*|to\s+)(?P<m2>[A-Za-z]+)\s+(?P<y2>\d{4})$",
        name,
        re.IGNORECASE,
    )
    if month_range_dual_year:
        left = _parse_month_year(
            f"{month_range_dual_year.group('m1')} {month_range_dual_year.group('y1')}"
        )
        right = _parse_month_year(
            f"{month_range_dual_year.group('m2')} {month_range_dual_year.group('y2')}"
        )
        if left and right:
            start, _ = _month_range(left[0], left[1])
            _, end = _month_range(right[0], right[1])
            return TimestampRange(start, end)

    range_split = _split_to_range(name)
    if range_split:
        start_raw, end_raw = range_split
        start = _parse_single_bound(start_raw, True)
        end = _parse_single_bound(end_raw, False)
        if start or end:
            return TimestampRange(start, end)

    quarter = _parse_quarter(name)
    if quarter:
        start, end = _quarter_range(quarter[0], quarter[1])
        return TimestampRange(start, end)

    single_start = _parse_single_bound(name, True)
    single_end = _parse_single_bound(name, False)
    if single_start or single_end:
        return TimestampRange(single_start, single_end)

    return TimestampRange(None, None)
