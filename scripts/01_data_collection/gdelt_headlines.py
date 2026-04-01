from __future__ import annotations


import json

import time

from dataclasses import dataclass

from datetime import date, datetime, timedelta, timezone

from pathlib import Path

from typing import Any, Dict, Optional


import requests


KEYWORDS = "crude oil"

LANGUAGE = "eng"

START_DATE = date(2026, 1, 26)

END_DATE   = date(2026, 1, 27)

MAXRECORDS = 75

SORT       = "hybridrel"


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


def build_query(keywords: str, sourcelang: str) -> str:

    """
    Builds: "<keywords> sourcelang:eng"
    You can embed your own operators in keywords if you want (OR blocks, quotes, etc).
    """

    keywords = keywords.strip()

    if not keywords:

        raise ValueError("KEYWORDS cannot be empty")

    return f"{keywords} sourcelang:{sourcelang}"


@dataclass

class GdeltClient:

    timeout_sec: int = 60

    max_retries: int = 6

    backoff_base: float = 1.8


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


                if resp.status_code in (429, 500, 502, 503, 504):

                    raise RuntimeError(f"Transient HTTP {resp.status_code}: {resp.text[:200]}")


                resp.raise_for_status()


                try:

                    return resp.json()

                except Exception as e:

                    preview = resp.text[:300]

                    raise RuntimeError(f"Response was not valid JSON. First 300 chars:\n{preview}") from e


            except Exception as e:

                last_err = e

                if attempt >= self.max_retries:

                    break

                sleep_s = self.backoff_base ** attempt

                time.sleep(sleep_s)


        raise RuntimeError(f"Failed after {self.max_retries} attempts. Last error: {last_err}")


def build_params(
    *,
    keywords: str,
    language: str,
    start_date: date,
    end_date: date,
    maxrecords: int,
    sort: str,
) -> Dict[str, str]:

    """
    This matches your working reference URL (cleaned):
    - mode=artlist
    - format=json
    - startdatetime / enddatetime
    - query=<keywords> sourcelang:eng
    - maxrecords=75
    - sort=hybridrel
    """

    if end_date < start_date:

        raise ValueError("END_DATE must be >= START_DATE")


    startdt = to_yyyymmddhhmmss_utc(start_date, end_of_day=False)

    enddt = to_yyyymmddhhmmss_utc(end_date, end_of_day=True)


    query = build_query(keywords, language)


    return {
        "query": query,
        "mode": "artlist",
        "format": "json",
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


    params = build_params(
        keywords=KEYWORDS,
        language=LANGUAGE,
        start_date=START_DATE,
        end_date=END_DATE,
        maxrecords=MAXRECORDS,
        sort=SORT,
    )


    safe_kw = "".join(c if c.isalnum() else "_" for c in KEYWORDS)[:60]

    out_file = Path("gdelt_json") / f"{safe_kw}_{START_DATE.isoformat()}_to_{END_DATE.isoformat()}.json"


    data = client.fetch_json(params)

    save_json(data, out_file)


    articles = None

    if isinstance(data, dict):

        articles = data.get("articles")

    n = len(articles) if isinstance(articles, list) else "unknown"


    print("✅ Saved JSON:", out_file.resolve())

    print("ℹ️ Articles count:", n)

    print("ℹ️ Params used:", params)


if __name__ == "__main__":

    main()
