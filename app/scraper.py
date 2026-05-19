"""
scraper.py  (FIXED)
-------------------
Scrapes the SHL product catalog (Individual Test Solutions only)
and saves results to catalog.json.

WHY THIS VERSION USES PLAYWRIGHT:
  The SHL catalog table is rendered by JavaScript. requests + BeautifulSoup
  only download the raw HTML shell — the table rows never arrive.
  Playwright launches a real headless Chromium browser that executes JS,
  waits for the table to appear, then hands us the rendered DOM.

Setup (one-time):
    pip install playwright beautifulsoup4 requests
    playwright install chromium

Run:
    python scraper.py
"""

import json
import time
import logging
import re
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL = "https://www.shl.com"

# type=1  →  Individual Test Solutions only (not Pre-Packaged Job Solutions)
CATALOG_URL = "https://www.shl.com/products/product-catalog/?start={start}&type=1"

PAGE_SIZE = 12
OUTPUT_FILE = "catalog.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# CSS selectors to try for the catalog table rows
# SHL has changed their markup before; we try multiple.
ROW_SELECTORS = [
    "tr.catalogue__row",
    "[data-course-id]",
    "table.custom__table tbody tr",
    ".custom__table tbody tr",
    "table tbody tr",
]

# Selectors for the description on individual assessment pages
DESCRIPTION_SELECTORS = [
    ".product-catalogue-training-headline__description",
    ".catalogue__description",
    ".product-description",
    "meta[name='description']",
    ".rich-text p",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text(element) -> str:
    """Safe text extraction from a BeautifulSoup element."""
    if not element:
        return ""
    return element.get_text(separator=" ", strip=True)


def parse_bool_icon(cell_html: str) -> bool:
    """
    SHL uses tick/cross SVG icons or 'Yes'/'No' text for boolean columns.
    Returns True if the cell indicates 'yes / supported'.
    """
    text = cell_html.lower()
    # SVG tick class names SHL has used
    positive_signals = ["tick", "check", "yes", "true", "&#10003;", "✓", "circle-filled"]
    return any(s in text for s in positive_signals)


def get_total_count(soup: BeautifulSoup) -> int:
    """Parse total result count from page text."""
    text = soup.get_text(separator=" ", strip=True)
    for pattern in [r"of\s+(\d+)\s+results", r"(\d+)\s+results"]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 0   # Return 0 instead of 999 so we can detect failure cleanly


# ── Page scraping with Playwright ─────────────────────────────────────────────

def fetch_page_html(page, start: int) -> str:
    """
    Load one catalog page in the already-open Playwright page object,
    wait for the table to render, return the full page HTML.
    """
    url = CATALOG_URL.format(start=start)
    logger.info(f"Navigating to: {url}")

    page.goto(url, wait_until="networkidle", timeout=60_000)

    # Wait for at least one of the known row selectors to appear
    for selector in ROW_SELECTORS:
        try:
            page.wait_for_selector(selector, timeout=10_000)
            logger.info(f"Table found with selector: {selector}")
            break
        except PWTimeout:
            continue
    else:
        # No selector matched — dump HTML for debugging
        logger.warning(f"No table rows found at start={start}. Saving debug HTML.")
        with open(f"debug_page_{start}.html", "w", encoding="utf-8") as f:
            f.write(page.content())

    return page.content()


def parse_items(html: str) -> list[dict]:
    """
    Parse assessment rows from rendered HTML.
    Returns a list of partial item dicts (no description yet).
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for selector in ROW_SELECTORS:
        rows = soup.select(selector)
        if rows:
            logger.info(f"Parsing {len(rows)} rows (selector: {selector})")
            break

    if not rows:
        return []

    items = []
    for row in rows:
        try:
            link = row.select_one("a[href]")
            if not link:
                continue

            name = extract_text(link).strip()
            href = link.get("href", "")
            url = (BASE_URL + href) if href.startswith("/") else href

            # Skip empty or nav links
            if not name or not href or "/product-catalog" not in href and "/products/" not in href:
                # Be lenient — only skip clearly wrong links
                if not href or href in ("#", "/"):
                    continue

            cells = row.select("td")

            # Remote Testing column: typically 2nd or 3rd <td>
            # SHL uses SVG icon OR text. We check each cell's inner HTML.
            remote_testing = False
            adaptive_irt = False

            # Try to detect by column position (SHL catalog has known column order:
            # Name | Remote Testing | Adaptive/IRT | Test Types)
            if len(cells) >= 2:
                remote_testing = parse_bool_icon(str(cells[1]))
            if len(cells) >= 3:
                adaptive_irt = parse_bool_icon(str(cells[2]))

            # Test types: look for specific label spans / data attributes
            test_types = []
            # SHL sometimes uses spans with class like 'catalogue__circle' or similar
            type_spans = row.select("span[class*='type'], span[class*='circle'], td:last-child span")
            for span in type_spans:
                t = extract_text(span).strip()
                if t and len(t) < 60 and t not in test_types:
                    test_types.append(t)

            # Fallback: grab any small labels not already in name
            if not test_types:
                for span in row.select("span, small"):
                    t = extract_text(span).strip()
                    if t and t != name and len(t) < 60 and t not in test_types:
                        test_types.append(t)

            items.append({
                "name": name,
                "url": url,
                "test_types": test_types,
                "remote_testing": remote_testing,
                "adaptive_irt": adaptive_irt,
                "description": "",
                "job_levels": "",
                "languages": "",
            })

        except Exception as e:
            logger.warning(f"Error parsing row: {e}")

    return items


# ── Detail page fetching (plain requests — no JS needed on detail pages) ──────

def fetch_detail(item: dict, session: requests.Session) -> dict:
    """
    Fetch the individual assessment page to get description, job levels,
    languages, and verify remote/adaptive flags from structured data.
    """
    try:
        resp = session.get(item["url"], headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Description
        description = ""
        for selector in DESCRIPTION_SELECTORS:
            tag = soup.select_one(selector)
            if tag:
                if tag.name == "meta":
                    description = tag.get("content", "").strip()
                else:
                    description = extract_text(tag)
                if description:
                    break
        item["description"] = description

        # Job levels (SHL detail pages often list these)
        jl_tag = soup.select_one("[class*='job-level'], [class*='jobLevel'], .job_levels")
        if jl_tag:
            item["job_levels"] = extract_text(jl_tag)

        # Languages
        lang_tag = soup.select_one("[class*='language'], .languages")
        if lang_tag:
            item["languages"] = extract_text(lang_tag)

        # Try to get test types from structured data on the detail page if missing
        if not item["test_types"]:
            for selector in ["[class*='test-type']", "[class*='testType']", ".type-label"]:
                tags = soup.select(selector)
                types = [extract_text(t) for t in tags if extract_text(t)]
                if types:
                    item["test_types"] = types
                    break

    except Exception as e:
        logger.warning(f"Could not fetch detail for {item['name']}: {e}")

    return item


# ── Main scraper orchestration ────────────────────────────────────────────────

def scrape_catalog() -> list[dict]:
    all_items: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=HEADERS["User-Agent"],
            locale="en-US",
        )
        page = context.new_page()

        # ── Pass 1: collect all listing rows ─────────────────────────────────
        start = 0
        while True:
            try:
                html = fetch_page_html(page, start)
            except Exception as e:
                logger.error(f"Failed to load page start={start}: {e}")
                break

            page_items = parse_items(html)

            if not page_items:
                logger.info(f"No items at start={start}. Scraping complete.")
                break

            all_items.extend(page_items)
            logger.info(f"Total collected: {len(all_items)}")

            # Stop condition: if we got fewer than PAGE_SIZE items, we're on the last page
            if len(page_items) < PAGE_SIZE:
                logger.info("Partial page — reached end of catalog.")
                break

            start += PAGE_SIZE
            time.sleep(1.5)   # be polite

        browser.close()

    logger.info(f"Scraping finished. {len(all_items)} items found.")

    if not all_items:
        logger.error(
            "No items scraped! The page structure may have changed.\n"
            "Check debug_page_0.html to inspect the rendered HTML.\n"
            "You may need to update ROW_SELECTORS."
        )
        return []

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique_items = []
    for item in all_items:
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            unique_items.append(item)
    logger.info(f"After dedup: {len(unique_items)} unique items")

    # ── Pass 2: enrich each item with description from its detail page ────────
    logger.info("Fetching individual detail pages...")
    session = requests.Session()
    for idx, item in enumerate(unique_items):
        unique_items[idx] = fetch_detail(item, session)
        if (idx + 1) % 10 == 0:
            logger.info(f"  Details fetched: {idx + 1}/{len(unique_items)}")
        time.sleep(0.8)

    return unique_items


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logger.info("=== SHL Catalog Scraper (Playwright edition) ===")
    items = scrape_catalog()

    if not items:
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Saved {len(items)} items to {OUTPUT_FILE}")
    logger.info("Sample item:")
    print(json.dumps(items[0], indent=2))


if __name__ == "__main__":
    main()