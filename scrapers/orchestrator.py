import csv
import os
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import requests

from scrapers.base import BaseScraper, Product, FIELD_NAMES, RAW_DIR
from scrapers.shopify import ShopifyScraper
from scrapers.woocommerce import WooCommerceScraper


SHOPIFY_PATTERN = re.compile(r"(?P<domain>[^.]+)\.myshopify\.com", re.IGNORECASE)
WC_API_PATTERN = re.compile(r"wp-json/wc/(store/v1|v[23])/products", re.IGNORECASE)


def detect_platform(url: str) -> str:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    if SHOPIFY_PATTERN.search(hostname):
        return "shopify"
    if WC_API_PATTERN.search(url):
        return "woocommerce"
    if hostname.endswith("myshopify.com") or "/products.json" in url:
        return "shopify"

    try:
        probe = requests.get(url.rstrip("/") + "/products.json?page=1&limit=1", timeout=5)
        if probe.status_code == 200 and "products" in probe.json():
            return "shopify"
    except Exception:
        pass

    for endpoint in ("/wp-json/wc/store/v1/products", "/wp-json/wc/v3/products"):
        try:
            probe = requests.get(url.rstrip("/") + endpoint + "?per_page=1", timeout=5)
            if probe.status_code in (200, 401):
                return "woocommerce"
        except Exception:
            continue

    return "generic"


def build_scraper(
    url: str,
    platform: Optional[str] = None,
    max_pages: Optional[int] = None,
    consumer_key: Optional[str] = None,
    consumer_secret: Optional[str] = None,
) -> BaseScraper:
    platform = platform or detect_platform(url)

    if platform == "shopify":
        m = SHOPIFY_PATTERN.search(url)
        if m:
            domain = m.group("domain") + ".myshopify.com"
        else:
            parsed = urlparse(url)
            domain = parsed.hostname or url.strip("/")
        return ShopifyScraper(store_domain=domain, max_pages=max_pages)

    if platform == "woocommerce":
        base = url
        if "/wp-json/" in url:
            base = url[: url.index("/wp-json/")]
        return WooCommerceScraper(
            base_url=base,
            consumer_key=consumer_key or "",
            consumer_secret=consumer_secret or "",
            max_pages=max_pages,
        )

    return GenericScraper(source_url=url, max_pages=max_pages)


class GenericScraper(BaseScraper):
    source_name = "generic"

    def __init__(self, source_url: str, max_pages: int | None = None):
        self.source_url = source_url.rstrip("/")
        self.max_pages = max_pages

    def fetch_page(self, page: int) -> list[dict]:
        for endpoint in ("/products.json", "/api/products", "/products"):
            url = f"{self.source_url}{endpoint}"
            try:
                resp = requests.get(url, params={"page": page, "limit": self.page_size}, timeout=10)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    for key in ("products", "data", "items", "results"):
                        if key in data and isinstance(data[key], list):
                            return data[key]
            except Exception:
                continue
        return []


def deduplicate(products: list[Product]) -> list[Product]:
    seen: set[str] = set()
    unique: list[Product] = []
    for p in products:
        key = p.product_id or p.url or p.title
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        unique.append(p)
    return unique


def scrape_urls(
    urls: list[str],
    max_pages: Optional[int] = None,
    consumer_key: Optional[str] = None,
    consumer_secret: Optional[str] = None,
    output: Optional[str] = None,
) -> list[Product]:
    all_products: list[Product] = []

    for url in urls:
        platform = detect_platform(url)
        print(f"[orchestrator] {url} → {platform}")
        scraper = build_scraper(
            url=url,
            platform=platform,
            max_pages=max_pages,
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
        )
        raw_products = []
        page = 1
        while True:
            items = scraper.fetch_page(page)
            if not items:
                break
            for item in items:
                norm_p = scraper.normalize(item)
                if scraper.is_valid_product(norm_p):
                    raw_products.append(norm_p)
            page += 1
            if max_pages and page > max_pages:
                break
        all_products.extend(raw_products)
        print(f"[orchestrator]   → {len(raw_products)} produits bruts")

    before = len(all_products)
    all_products = deduplicate(all_products)
    dupes = before - len(all_products)
    if dupes:
        print(f"[orchestrator] {dupes} doublons supprimés")

    os.makedirs(RAW_DIR, exist_ok=True)
    filepath = output or os.path.join(RAW_DIR, f"aggregated_{datetime.now():%Y%m%d_%H%M%S}.csv")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        writer.writeheader()
        for p in all_products:
            writer.writerow(p.to_dict())

    print(f"[orchestrator] {len(all_products)} produits totaux → {filepath}")
    return all_products


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not args:
        print("Usage: python -m scrapers.orchestrator URL1 [URL2 ...] [--max-pages N] [--ck KEY] [--cs SECRET]")
        sys.exit(1)

    urls = []
    max_pages = None
    ck = None
    cs = None

    it = iter(args)
    for arg in it:
        if arg == "--max-pages":
            max_pages = int(next(it, "5"))
        elif arg == "--ck":
            ck = next(it, None)
        elif arg == "--cs":
            cs = next(it, None)
        elif arg.startswith("http://") or arg.startswith("https://"):
            urls.append(arg)

    scrape_urls(urls, max_pages=max_pages, consumer_key=ck, consumer_secret=cs)
