from __future__ import annotations


import csv

import re

import time

from datetime import date, timedelta

from pathlib import Path

from typing import Optional, Tuple, Dict, List, Set


import requests


RAW_URL = " https://api.gdeltproject.org/api/v2/doc/doc?format=html&startdatetime=20260126000000&enddatetime=20260126235959&query=%20(crude%20OR%20oil%20OR%20brent%20OR%20wti%20OR%20opec%20OR%20supply%20OR%20disruption%20OR%20gasoline%20OR%20embargo%20OR%20sanctions%20OR%20suez%20canal%20OR%20red%20sea%20OR%20panama%20canal%20OR%20inflation%20OR%20pmi%20OR%20interest%20rates%20OR%20inventory%20OR%20spr%20OR%20war%20OR%20ceasefire%20OR%20quota%20OR%20cuts)%20%20sourcelang:eng&mode=artlist&maxrecords=250&format=csv&sort=hybridrel "


BASE_FOLDER = Path(r"D:\ACADS\4-2\FIN SOP\Data\headlines data")


GLOBAL_START = date(2017, 1, 1)

GLOBAL_END   = date(2026, 1, 28)


MIN_SECONDS_BETWEEN_REQUESTS = 6.0


SKIP_IF_FILE_EXISTS = True


PATCH_SECONDS_BETWEEN_REQUESTS = 8.0


MONTH_FOLDERS = {
    1: "01_Jan", 2: "02_Feb", 3: "03_Mar", 4: "04_Apr",
    5: "05_May", 6: "06_Jun", 7: "07_Jul", 8: "08_Aug",
    9: "09_Sep", 10: "10_Oct", 11: "11_Nov", 12: "12_Dec"
}


def ensure_year_month_structure(base: Path, year: int) -> None:

    base.mkdir(parents=True, exist_ok=True)

    year_dir = base / str(year)

    year_dir.mkdir(exist_ok=True)

    for m in range(1, 13):

        (year_dir / MONTH_FOLDERS[m]).mkdir(exist_ok=True)


def daterange_inclusive(start: date, end: date):

    cur = start

    while cur <= end:

        yield cur

        cur += timedelta(days=1)


def gdelt_datetimes_for_day(d: date) -> Tuple[str, str]:

    ymd = d.strftime("%Y%m%d")

    return f"{ymd}000000", f"{ymd}235959"


def replace_only_datetimes(original_url: str, startdt: str, enddt: str) -> str:

    url = original_url


    if not re.search(r"startdatetime=\d{14}", url):

        raise ValueError("startdatetime=... not found in the provided URL")

    if not re.search(r"enddatetime=\d{14}", url):

        raise ValueError("enddatetime=... not found in the provided URL")


    url = re.sub(r"(startdatetime=)\d{14}", r"\g<1>" + startdt, url, count=1)

    url = re.sub(r"(enddatetime=)\d{14}", r"\g<1>" + enddt, url, count=1)

    return url


def output_path_for_day(base: Path, d: date) -> Path:

    year_dir = base / str(d.year)

    month_dir = year_dir / MONTH_FOLDERS[d.month]

    filename = f"{d.isoformat()}.csv"

    return month_dir / filename


def ensure_summary_csv(summary_path: Path) -> None:

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():

        with summary_path.open("w", newline="", encoding="utf-8") as f:

            w = csv.writer(f)

            w.writerow([
                "date",
                "status",
                "http_status",
                "bytes_written",
                "file_path",
                "error"
            ])


def append_summary(summary_path: Path, row: list) -> None:

    with summary_path.open("a", newline="", encoding="utf-8") as f:

        csv.writer(f).writerow(row)


def download_csv(url: str, out_path: Path, *, timeout: int = 90) -> Tuple[int, int]:

    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(".tmp")


    headers = {"User-Agent": "gdelt-csv-downloader/1.0"}


    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:

        http_status = r.status_code

        r.raise_for_status()


        written = 0

        with open(tmp_path, "wb") as f:

            for chunk in r.iter_content(chunk_size=1024 * 64):

                if chunk:

                    f.write(chunk)

                    written += len(chunk)


    tmp_path.replace(out_path)

    return http_status, written


def get_year_window(year: int) -> Tuple[date, date]:

    start = date(year, 1, 1)

    end = date(year, 12, 31)


    if start < GLOBAL_START:

        start = GLOBAL_START

    if end > GLOBAL_END:

        end = GLOBAL_END


    if end < start:

        raise ValueError(f"Year {year} is outside the global range {GLOBAL_START} to {GLOBAL_END}")


    return start, end


def run_one_year(year: int) -> None:

    ensure_year_month_structure(BASE_FOLDER, year)


    summary_path = BASE_FOLDER / "retrieval_summary.csv"

    ensure_summary_csv(summary_path)


    start_day, end_day = get_year_window(year)


    raw_url = RAW_URL.strip()


    print(f"Running year {year}: {start_day} -> {end_day}")

    print(f"Summary CSV: {summary_path.resolve()}")


    last_request_time = 0.0


    for day in daterange_inclusive(start_day, end_day):

        out_path = output_path_for_day(BASE_FOLDER, day)


        if SKIP_IF_FILE_EXISTS and out_path.exists():

            append_summary(summary_path, [
                day.isoformat(), "SKIP", "", 0, str(out_path), "File already exists"
            ])

            print(f"[SKIP] {day} -> already exists")

            continue


        now = time.time()

        wait = MIN_SECONDS_BETWEEN_REQUESTS - (now - last_request_time)

        if wait > 0:

            time.sleep(wait)


        last_request_time = time.time()


        startdt, enddt = gdelt_datetimes_for_day(day)

        final_url = replace_only_datetimes(raw_url, startdt, enddt)


        try:

            http_status, bytes_written = download_csv(final_url, out_path)

            append_summary(summary_path, [
                day.isoformat(), "SUCCESS", http_status, bytes_written, str(out_path), ""
            ])

            print(f"[OK]   {day} -> {out_path.name} ({bytes_written} bytes)")

        except Exception as e:

            append_summary(summary_path, [
                day.isoformat(), "FAIL", "", 0, str(out_path), str(e)[:500]
            ])

            print(f"[FAIL] {day} -> {e}")


def is_valid_gdelt_csv_bytes(head_bytes: bytes) -> Tuple[bool, str]:

    """
    Checks whether the downloaded content looks like a real GDELT CSV.
    We only inspect the first chunk (header line).
    """

    if not head_bytes:

        return False, "Empty content"


    text = head_bytes.decode("utf-8", errors="replace")

    lines = text.splitlines()

    first_line = lines[0] if lines else ""

    first_line = first_line.lstrip("\ufeff").strip()

    low = first_line.lower()


    if low.startswith("<!doctype") or low.startswith("<html") or "<html" in low:

        return False, "HTML returned instead of CSV"

    if low.startswith("{") or low.startswith("["):

        return False, "JSON returned instead of CSV"

    if "," not in first_line:

        return False, "No CSV commas in header"


    if ("url" not in low) or ("title" not in low):

        return False, f"Unexpected header: {first_line[:120]}"


    return True, ""


def is_valid_gdelt_csv_file(path: Path) -> Tuple[bool, str]:

    """
    Validates an existing CSV file quickly (header only).
    """

    try:

        with path.open("rb") as f:

            head = f.read(4096)

        return is_valid_gdelt_csv_bytes(head)

    except Exception as e:

        return False, f"Validation error: {e}"


def read_latest_status_map(paths: List[Path], year: int) -> Dict[date, str]:

    """
    Reads one or more summary CSVs and returns LATEST status per date for that year.
    Later files in `paths` override earlier ones.
    """

    latest: Dict[date, str] = {}


    for p in paths:

        if not p.exists():

            continue

        with p.open("r", newline="", encoding="utf-8", errors="replace") as f:

            r = csv.reader(f)

            _hdr = next(r, None)

            for row in r:

                if not row or len(row) < 2:

                    continue

                d_str = row[0].strip()

                status = row[1].strip().upper()

                try:

                    d = date.fromisoformat(d_str)

                except Exception:

                    continue

                if d.year != year:

                    continue

                latest[d] = status


    return latest


def safe_download_csv_patch(url: str, out_path: Path, *, timeout: int = 90) -> Tuple[int, int, bool, str]:

    """
    Patch download that:
      - downloads to .tmp
      - validates header of tmp
      - only replaces final file if valid
    Returns: (http_status, bytes_written, did_replace, error_reason_if_any)
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(".tmp")


    headers = {"User-Agent": "gdelt-csv-downloader/1.0"}


    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:

        http_status = r.status_code

        r.raise_for_status()


        written = 0

        first_chunk = b""

        first_chunk_captured = False


        with open(tmp_path, "wb") as f:

            for chunk in r.iter_content(chunk_size=1024 * 64):

                if chunk:

                    if not first_chunk_captured:

                        first_chunk = chunk[:4096]

                        first_chunk_captured = True

                    f.write(chunk)

                    written += len(chunk)


    ok, reason = is_valid_gdelt_csv_bytes(first_chunk)

    if not ok:

        try:

            tmp_path.unlink(missing_ok=True)

        except Exception:

            pass

        return http_status, written, False, reason


    tmp_path.replace(out_path)

    return http_status, written, True, ""


def scan_year_for_patch(year: int) -> Dict[int, Dict[str, List[date]]]:

    """
    Month-wise scan:
      - missing: file doesn't exist
      - failed: latest status for that date is FAIL/RETRY_FAIL
      - corrupt: file exists but header invalid (reported only; NOT retried)
      - to_retry: missing ∪ failed   (STRICTLY — no corrupt)
    """

    ensure_year_month_structure(BASE_FOLDER, year)

    start_day, end_day = get_year_window(year)


    main_summary = BASE_FOLDER / "retrieval_summary.csv"

    patch_summary = BASE_FOLDER / f"retrieval_patch_{year}.csv"

    latest_status = read_latest_status_map([main_summary, patch_summary], year)


    stats: Dict[int, Dict[str, List[date]]] = {
        m: {"missing": [], "failed": [], "corrupt": [], "to_retry": []} for m in range(1, 13)
    }


    for d in daterange_inclusive(start_day, end_day):

        out_path = output_path_for_day(BASE_FOLDER, d)


        if not out_path.exists():

            stats[d.month]["missing"].append(d)


        if out_path.exists():

            ok, _ = is_valid_gdelt_csv_file(out_path)

            if not ok:

                stats[d.month]["corrupt"].append(d)


        st = latest_status.get(d, "")

        if st in {"FAIL", "RETRY_FAIL"}:

            stats[d.month]["failed"].append(d)


    for m in range(1, 13):

        stats[m]["to_retry"] = sorted(set(stats[m]["missing"]) | set(stats[m]["failed"]))


    return stats


def patch_one_year(year: int) -> None:

    """
    Retries ONLY missing + failed days with 8-second delay.
    Does NOT retry corrupt days.
    If API returns junk (3 bytes etc.), it logs RETRY_FAIL and DOES NOT overwrite existing file.
    """

    ensure_year_month_structure(BASE_FOLDER, year)


    raw_url = RAW_URL.strip()


    patch_summary = BASE_FOLDER / f"retrieval_patch_{year}.csv"

    ensure_summary_csv(patch_summary)


    stats = scan_year_for_patch(year)


    print(f"\n=== PATCH SCAN RESULTS for {year} ===")

    total_retry = 0

    for m in range(1, 13):

        missing_n = len(stats[m]["missing"])

        failed_n  = len(stats[m]["failed"])

        corrupt_n = len(stats[m]["corrupt"])

        retry_n   = len(stats[m]["to_retry"])

        total_retry += retry_n

        print(f"{MONTH_FOLDERS[m]}: missing={missing_n}, failed={failed_n}, corrupt={corrupt_n}  --> to_retry={retry_n}")


    if total_retry == 0:

        print("\nNothing to patch (no missing + no failed). Corrupt files are intentionally left untouched.")

        return


    print(f"\nTotal to retry: {total_retry}")

    print(f"Patch summary CSV: {patch_summary.resolve()}\n")


    last_request_time = 0.0


    for m in range(1, 13):

        to_retry = stats[m]["to_retry"]

        if not to_retry:

            continue


        print(f"\n--- Retrying {year}/{MONTH_FOLDERS[m]} : {len(to_retry)} day(s) ---")


        for d in to_retry:

            out_path = output_path_for_day(BASE_FOLDER, d)


            now = time.time()

            wait = PATCH_SECONDS_BETWEEN_REQUESTS - (now - last_request_time)

            if wait > 0:

                time.sleep(wait)

            last_request_time = time.time()


            startdt, enddt = gdelt_datetimes_for_day(d)

            final_url = replace_only_datetimes(raw_url, startdt, enddt)


            try:

                http_status, bytes_written, replaced, reason = safe_download_csv_patch(final_url, out_path)


                if not replaced:

                    append_summary(patch_summary, [
                        d.isoformat(), "RETRY_FAIL", http_status, bytes_written, str(out_path),
                        f"Invalid CSV (not overwritten): {reason}"
                    ])

                    print(f"[RETRY_FAIL] {d} -> invalid CSV ({bytes_written} bytes): {reason}")

                    continue


                append_summary(patch_summary, [
                    d.isoformat(), "RETRY_SUCCESS", http_status, bytes_written, str(out_path), ""
                ])

                print(f"[RETRY_OK]   {d} -> {out_path.name} ({bytes_written} bytes)")


            except requests.exceptions.HTTPError as e:

                status = getattr(e.response, "status_code", "")

                if status == 429:

                    append_summary(patch_summary, [
                        d.isoformat(), "RETRY_FAIL", 429, 0, str(out_path),
                        "429 Too Many Requests (logged, not retried)"
                    ])

                    print(f"[RETRY_429]  {d} -> 429 Too Many Requests (skipping)")

                else:

                    append_summary(patch_summary, [
                        d.isoformat(), "RETRY_FAIL", status, 0, str(out_path), str(e)[:500]
                    ])

                    print(f"[RETRY_FAIL] {d} -> HTTPError {status}: {e}")


            except Exception as e:

                append_summary(patch_summary, [
                    d.isoformat(), "RETRY_FAIL", "", 0, str(out_path), str(e)[:500]
                ])

                print(f"[RETRY_FAIL] {d} -> {e}")


    print("\nPatch run completed.")


if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser(description="Download GDELT daily CSVs year-by-year (2017-01-01 to 2026-01-28).")

    parser.add_argument("--year", type=int, required=True, help="Year to run (e.g., 2017 ... 2026)")

    parser.add_argument("--patch", action="store_true", help="Scan month-by-month and re-download missing/failed/corrupt days with 8s delay")

    args = parser.parse_args()


    if args.patch:

        patch_one_year(args.year)

    else:

        run_one_year(args.year)
