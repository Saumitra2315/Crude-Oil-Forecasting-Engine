from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from playwright.sync_api import sync_playwright, Page, TimeoutError as PWTimeoutError


BASE_URL = "https://api.gdeltproject.org/api/v2/summary/summary"


# -----------------------------
# Utilities
# -----------------------------

def mmddyyyy(d: dt.date) -> str:
    # Matches the UI format you showed (e.g., 01/26/2026)
    return d.strftime("%m/%d/%Y")


def sanitize_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "", s)
    return s[:120] if len(s) > 120 else s


def day_bounds_for_one_day(day: dt.date) -> Tuple[dt.date, dt.date]:
    # In your UI you set start date and end date (for now one-day test).
    # If you want EXACT one-day, use same date for start and end.
    return day, day


# -----------------------------
# Robust locator helpers
# (Try name-based selectors first, then fallback to "text -> next input/select")
# -----------------------------

def select_dropdown_by_label_text(page: Page, left_label_text: str, option_label: str) -> None:
    """
    Finds the <select> immediately after a visible label text like "Dataset", "Output Type",
    "Time Period", "Limit To Language" and selects by visible label.
    Playwright supports selecting by label. :contentReference[oaicite:2]{index=2}
    """
    label = page.get_by_text(left_label_text, exact=True)
    select = label.locator("xpath=following::select[1]")
    select.wait_for(state="visible", timeout=15000)
    select.select_option(label=option_label)


def fill_input_next_to_label(page: Page, left_label_text: str, value: str) -> None:
    """
    Finds the <input> immediately after a visible label text like "Keyword(s)", "Start Date", "End Date"
    and fills it.
    """
    label = page.get_by_text(left_label_text, exact=True)
    inp = label.locator("xpath=following::input[1]")
    inp.wait_for(state="visible", timeout=15000)
    inp.fill(value)


def click_button_by_text(page: Page, button_text: str) -> None:
    btn = page.get_by_role("button", name=button_text)
    btn.wait_for(state="visible", timeout=15000)
    btn.scroll_into_view_if_needed()
    btn.click()


def click_top_articles_csv(page: Page) -> None:
    """
    On the results page, scroll to "Top Articles" and click the CSV link in that section.
    The results page includes a "Top Articles" section with [URL] [CSV] ... :contentReference[oaicite:3]{index=3}
    We want the FIRST CSV link after the "Top Articles" heading (not Top Images).
    """
    heading = page.get_by_role("heading", name="Top Articles")
    heading.wait_for(state="visible", timeout=30000)
    heading.scroll_into_view_if_needed()

    # First CSV link after Top Articles heading
    csv_link = heading.locator("xpath=following::a[normalize-space()='CSV'][1]")
    csv_link.wait_for(state="visible", timeout=15000)
    csv_link.click()


# -----------------------------
# Main automation
# -----------------------------

@dataclass
class GDELTSummaryRunConfig:
    keyword: str = "crude oil"
    language: str = "English"
    dataset: str = "Global Online News Coverage"
    output_type: str = "Summary Overview Dashboard"
    time_period: str = "Custom Date Range"
    start_date: dt.date = dt.date(2026, 1, 26)
    end_date: dt.date = dt.date(2026, 1, 26)

    headless: bool = True
    output_dir: Path = Path("gdelt_ui_downloads")

    # Safety/robustness
    navigation_timeout_ms: int = 60000


def download_one_csv_via_ui(cfg: GDELTSummaryRunConfig) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # File name like: crude_oil_2026-01-26_to_2026-01-26.csv
    fname = f"{sanitize_filename(cfg.keyword)}_{cfg.start_date.isoformat()}_to_{cfg.end_date.isoformat()}.csv"
    out_path = cfg.output_dir / fname

    with sync_playwright() as p:
        # Playwright download handling: expect_download + save_as :contentReference[oaicite:4]{index=4}
        browser = p.chromium.launch(headless=cfg.headless)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        page.set_default_timeout(cfg.navigation_timeout_ms)

        # Step 1: open base URL
        page.goto(BASE_URL, wait_until="domcontentloaded")

        # Step 2: keep dataset & output type as in your screenshot
        # The page literally lists "Step 1: Dataset" and "Step 2: Output Type" :contentReference[oaicite:5]{index=5}
        select_dropdown_by_label_text(page, "Dataset", cfg.dataset)
        select_dropdown_by_label_text(page, "Output Type", cfg.output_type)

        # Step 3: keyword, custom range, language
        fill_input_next_to_label(page, "Keyword(s)", cfg.keyword)
        select_dropdown_by_label_text(page, "Time Period", cfg.time_period)
        fill_input_next_to_label(page, "Start Date", mmddyyyy(cfg.start_date))
        fill_input_next_to_label(page, "End Date", mmddyyyy(cfg.end_date))
        select_dropdown_by_label_text(page, "Limit To Language", cfg.language)

        # Step 4: click "Create Summary"
        click_button_by_text(page, "Create Summary")

        # Wait for summary results to load (it will navigate)
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_load_state("networkidle")

        # Step 5: scroll to Top Articles and click CSV (download)
        try:
            with page.expect_download() as dl_info:
                click_top_articles_csv(page)
            download = dl_info.value
        except PWTimeoutError as e:
            # Save a screenshot to debug selector issues if the UI changes
            debug_path = cfg.output_dir / "debug_failed_to_download.png"
            page.screenshot(path=str(debug_path), full_page=True)
            raise RuntimeError(
                f"Timed out waiting for CSV download. Screenshot saved to: {debug_path}"
            ) from e

        # Step 6: save the downloaded CSV exactly where you want it
        download.save_as(str(out_path))

        context.close()
        browser.close()

    return out_path


if __name__ == "__main__":
    # ONE-DAY TEST exactly as you said:
    day = dt.date(2026, 1, 26)
    s, e = day_bounds_for_one_day(day)

    cfg = GDELTSummaryRunConfig(
        keyword="crude oil",
        start_date=s,
        end_date=e,
        headless=False,  # set True after first successful run
        output_dir=Path("gdelt_ui_downloads"),
    )

    saved = download_one_csv_via_ui(cfg)
    print(f"✅ Saved CSV to: {saved.resolve()}")
