from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# ============================================================
# ✅✅✅ EDIT THIS SECTION (keywords + timeframe) ✅✅✅
# ============================================================
KEYWORDS_LIST = [
    "crude oil",
    "brent",
    "wti"
    # add more here...
]

LANGUAGE = "eng"
START_DATE = date(2026, 1, 26)
END_DATE   = date(2026, 1, 27)
MAXRECORDS = 75
SORT       = "hybridrel"

# 🔥 NEW: how to combine your keywords
BETWEEN_KEYWORDS = "OR"        # OR / AND across different entries in KEYWORDS_LIST
INSIDE_MULTIWORD = "AND"       # "crude oil" -> crude AND oil (matches your UI's "crude AND oil")
WRAP_PHRASES     = False       # True => "crude oil" treated as a phrase
# ============================================================


GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"


def to_yyyymmddhhmmss_utc(d: date, end_of_day: bool) -> str:
    """
    Convert a date to GDELT STARTDATETIME/ENDDATETIME format in UTC.
    start => YYYYMMDD000000
    end   => YYYYMMDD235959
    """
    if end_of_day:
        dt_ = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
    else:
        dt_ = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    return dt_.strftime("%Y%m%d%H%M%S")


# ----------------------------
# ✅ NEW: build ONE query from ALL keywords using OR
# ----------------------------
def build_query_from_list(
    keywords_list: List[str],
    sourcelang: str,
    *,
    between_keywords: str = "OR",
    inside_multiword: str = "AND",
    wrap_phrases: bool = False,
) -> str:
    """
    Builds ONE query containing ALL keywords together.

    Example:
      KEYWORDS_LIST = ["crude oil", "brent", "wti"]
      wrap_phrases=False ->
        ((crude AND oil) OR (brent) OR (wti)) sourcelang:eng

      wrap_phrases=True ->
        (("crude oil") OR (brent) OR (wti)) sourcelang:eng
    """
    if not keywords_list:
        raise ValueError("KEYWORDS_LIST cannot be empty")

    groups: List[str] = []
    for kw in keywords_list:
        kw = kw.strip()
        if not kw:
            continue

        if wrap_phrases and (" " in kw or "\t" in kw):
            groups.append(f'("{kw}")')
        else:
            parts = [p for p in kw.replace("\t", " ").split(" ") if p]
            if len(parts) == 1:
                groups.append(f"({parts[0]})")
            else:
                joiner = f" {inside_multiword.strip().upper()} "
                groups.append(f"({joiner.join(parts)})")

    if not groups:
        raise ValueError("No usable keywords after cleaning KEYWORDS_LIST")

    joiner2 = f" {between_keywords.strip().upper()} "
    combined = joiner2.join(groups)

    # Apply language restriction once to whole query
    return f"({combined}) sourcelang:{sourcelang}"


@dataclass
class GdeltClient:
    timeout_sec: int = 60
    max_retries: int = 6
    backoff_base: float = 1.8  # exponential backoff

    def fetch_json(self, params: Dict[str, str]) -> Dict[str, Any]:
        """
        Robust GET with retries for transient failures (429/5xx/timeouts).
        Returns parsed JSON dict.
        """
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "gdelt-doc-downloader/1.0 (+requests)",
                "Accept": "application/json",
            }
        )

        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = session.get(GDELT_DOC_ENDPOINT, params=params, timeout=self.timeout_sec)

                # Retry on common transient statuses
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"Transient HTTP {resp.status_code}: {resp.text[:200]}")

                resp.raise_for_status()

                # Ensure JSON parse
                try:
                    return resp.json()
                except Exception as e:
                    preview = resp.text[:300]
                    raise RuntimeError(
                        f"Response was not valid JSON. First 300 chars:\n{preview}"
                    ) from e

            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                sleep_s = self.backoff_base ** attempt
                time.sleep(sleep_s)

        raise RuntimeError(f"Failed after {self.max_retries} attempts. Last error: {last_err}")


def build_params(
    *,
    keywords_list: List[str],
    language: str,
    start_date: date,
    end_date: date,
    maxrecords: int,
    sort: str,
) -> Dict[str, str]:
    """
    Builds params for ONE combined query (all keywords together).

    - mode=artlist
    - format=json
    - startdatetime / enddatetime
    - query=(... OR ... OR ...) sourcelang:eng
    - maxrecords=75
    - sort=hybridrel
    """
    if end_date < start_date:
        raise ValueError("END_DATE must be >= START_DATE")

    startdt = to_yyyymmddhhmmss_utc(start_date, end_of_day=False)
    enddt = to_yyyymmddhhmmss_utc(end_date, end_of_day=True)

    # ✅ ONE query for all keywords (OR)
    query = build_query_from_list(
        keywords_list,
        sourcelang=language,
        between_keywords=BETWEEN_KEYWORDS,
        inside_multiword=INSIDE_MULTIWORD,
        wrap_phrases=WRAP_PHRASES,
    )

    return {
        "query": query,
        "mode": "artlist",
        "format": "json",          # IMPORTANT: keep ONLY json (no format=html)
        "startdatetime": startdt,
        "enddatetime": enddt,
        "maxrecords": str(maxrecords),
        "sort": sort,
    }


def save_json(data: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    client = GdeltClient()

    # ✅ Build ONE request for ALL keywords
    params = build_params(
        keywords_list=KEYWORDS_LIST,
        language=LANGUAGE,
        start_date=START_DATE,
        end_date=END_DATE,
        maxrecords=MAXRECORDS,
        sort=SORT,
    )

    # Safe filename from combined keywords
    combo = "_".join(KEYWORDS_LIST)
    safe_combo = "".join(c if c.isalnum() else "_" for c in combo)[:80]
    out_file = Path("gdelt_json") / f"{safe_combo}_{START_DATE.isoformat()}_to_{END_DATE.isoformat()}.json"

    data = client.fetch_json(params)
    save_json(data, out_file)

    articles = data.get("articles") if isinstance(data, dict) else None
    n = len(articles) if isinstance(articles, list) else "unknown"

    print("✅ Saved JSON:", out_file.resolve())
    print("ℹ️ Keywords combined with OR:", KEYWORDS_LIST, "| Articles:", n)
    print("ℹ️ Actual query sent:", params.get("query"))  # helpful for debugging


if __name__ == "__main__":
    main()
