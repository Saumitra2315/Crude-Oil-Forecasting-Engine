from __future__ import annotations


import re

from dataclasses import dataclass

from datetime import date, timedelta

from pathlib import Path

from typing import Dict, List, Tuple, Optional


ROOT = Path(r"D:\ACADS\4-2\FIN SOP\Data\Code\Relevance\Final_Polarity_Relevance\Final_Polarity_Relevance")


YEARS = list(range(2017, 2027))


MIN_SIZE_KB = 10

MIN_SIZE_BYTES = MIN_SIZE_KB * 1024


CHECK_BOTH_CSV_XLSX = True


REPORT_TXT_PATH = ROOT / "missing_files_report.txt"


DATE_RE = re.compile(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})")


def parse_date_from_filename(name: str) -> Optional[date]:

    """
    Extracts YYYY-MM-DD from a filename. Works even if suffix like _nli_probs is present.
    Example: 2025-06-01_nli_probs.csv -> 2025-06-01
    """

    m = DATE_RE.search(name)

    if not m:

        return None

    y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))

    return date(y, mo, d)


def iter_dates_in_year(y: int) -> List[date]:

    """Returns all calendar dates in a year y."""

    start = date(y, 1, 1)

    end = date(y + 1, 1, 1)

    out = []

    cur = start

    while cur < end:

        out.append(cur)

        cur += timedelta(days=1)

    return out


def find_file_for_day(year_dir: Path, day: date) -> Tuple[Optional[Path], int]:

    """
    Tries to locate the day file anywhere under year_dir (month subfolders).
    Returns (path or None, size_bytes_if_found_else_0).
    Matches filenames containing YYYY-MM-DD anywhere (robust to suffixes).
    Prefers larger file if multiple matches.
    """

    ymd = day.strftime("%Y-%m-%d")

    candidates: List[Path] = []


    patterns = [f"*{ymd}*.csv"]

    if CHECK_BOTH_CSV_XLSX:

        patterns += [f"*{ymd}*.xlsx", f"*{ymd}*.xls"]


    for pat in patterns:

        candidates.extend(year_dir.rglob(pat))


    if not candidates:

        return None, 0


    best = None

    best_size = -1

    for p in candidates:

        try:

            sz = p.stat().st_size

        except OSError:

            continue

        if sz > best_size:

            best_size = sz

            best = p


    return best, max(best_size, 0)


@dataclass

class YearStats:

    year: int

    expected_days: int

    present_good: int

    missing: int


def main():

    if not ROOT.exists():

        raise FileNotFoundError(f"ROOT not found: {ROOT}")


    missing_files: List[str] = []

    year_stats: List[YearStats] = []


    carry_sum = 0

    carry_sums: Dict[int, int] = {}


    total_missing = 0


    for y in YEARS:

        year_dir = ROOT / str(y)

        if not year_dir.exists():

            days = iter_dates_in_year(y)

            expected = len(days)

            total_missing += expected

            for d in days:

                missing_files.append(f"{d.isoformat()} (missing: year folder not found)")

            year_stats.append(YearStats(y, expected, 0, expected))

            carry_sum += 0

            carry_sums[y] = carry_sum

            continue


        days = iter_dates_in_year(y)

        expected = len(days)


        present_good = 0

        missing = 0


        for d in days:

            p, size = find_file_for_day(year_dir, d)


            if p is None:

                missing += 1

                missing_files.append(f"{d.isoformat()} (missing: file not found)")

                continue


            if size < MIN_SIZE_BYTES:

                missing += 1

                missing_files.append(f"{d.isoformat()} (missing: <{MIN_SIZE_KB}KB) -> {p}")

                continue


            present_good += 1


        total_missing += missing

        year_stats.append(YearStats(y, expected, present_good, missing))


        carry_sum += present_good

        carry_sums[y] = carry_sum


    def missing_sort_key(s: str):

        dt = parse_date_from_filename(s)

        return dt or date(1900, 1, 1)

    missing_files_sorted = sorted(missing_files, key=missing_sort_key)


    print("\n==================== SUMMARY ====================")

    print(f"ROOT: {ROOT}")

    print(f"MIN SIZE: {MIN_SIZE_KB} KB")

    print("-------------------------------------------------")

    print("Year | ExpectedDays | GoodFiles | MissingFiles | CarrySumGoodFiles")

    print("-------------------------------------------------")

    for st in year_stats:

        print(f"{st.year} | {st.expected_days:12d} | {st.present_good:9d} | {st.missing:12d} | {carry_sums[st.year]:16d}")

    print("-------------------------------------------------")

    print(f"TOTAL missing files: {total_missing}")

    print("=================================================\n")


    print("Missing files (date ascending):")

    for x in missing_files_sorted:

        print(x)


    if REPORT_TXT_PATH is not None:

        REPORT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:

            f.write("SUMMARY\n")

            f.write(f"ROOT: {ROOT}\n")

            f.write(f"MIN SIZE: {MIN_SIZE_KB} KB\n\n")

            f.write("Year | ExpectedDays | GoodFiles | MissingFiles | CarrySumGoodFiles\n")

            for st in year_stats:

                f.write(f"{st.year} | {st.expected_days} | {st.present_good} | {st.missing} | {carry_sums[st.year]}\n")

            f.write(f"\nTOTAL missing files: {total_missing}\n\n")

            f.write("Missing files (date ascending):\n")

            for x in missing_files_sorted:

                f.write(x + "\n")

        print(f"\n[WROTE REPORT] {REPORT_TXT_PATH}")


if __name__ == "__main__":

    main()
