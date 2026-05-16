import re
from html import unescape

import requests
from requests.auth import HTTPBasicAuth

from scrapers.base import BaseScraper

ENDPOINTS = [
    "wp-json/wc/store/v1/products",
    "wp-json/wp/v2/product",
    "wp-json/wc/v3/products",
]

STORE_API = 0
WP_V2_API = 1
V3_API = 2


class WooCommerceScraper(BaseScraper):
    source_name = "woocommerce"

    def __init__(
        self,
        base_url: str,
        consumer_key: str = "",
        consumer_secret: str = "",
        max_pages: int | None = None,
    ):
        self.base = base_url.rstrip("/")
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.max_pages = max_pages
        self._endpoint_idx: int | None = None
        self._auth = None
        if consumer_key and consumer_secret:
            self._auth = HTTPBasicAuth(consumer_key, consumer_secret)

    def _try_detect(self) -> int | None:
        for i, path in enumerate(ENDPOINTS):
            url = f"{self.base}/{path}?per_page=1"
            auth = self._auth if i == V3_API else None
            try:
                resp = requests.get(url, auth=auth, timeout=10)
                if resp.status_code == 200:
                    print(f"[WooCommerceScraper] Endpoint actif: /{path}")
                    return i
                if resp.status_code == 401 and i == V3_API:
                    print(f"[WooCommerceScraper] Endpoint nécessite auth: /{path}")
                    return i
            except Exception:
                continue
        return None

    def fetch_page(self, page: int) -> list[dict]:
        if self._endpoint_idx is None:
            self._endpoint_idx = self._try_detect()
            if self._endpoint_idx is None:
                print("[WooCommerceScraper] Aucun endpoint accessible")
                return []

        path = ENDPOINTS[self._endpoint_idx]
        auth = self._auth if self._endpoint_idx == V3_API else None
        url = f"{self.base}/{path}"

        params: dict = {"page": page, "per_page": self.page_size}
        if self._endpoint_idx == WP_V2_API:
            params.pop("per_page", None)
            params["per_page"] = self.page_size

        try:
            resp = requests.get(url, auth=auth, params=params, timeout=15)
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception:
            return []

        if self._endpoint_idx == STORE_API:
            return [self._store_api_to_v3(p) for p in data]
        if self._endpoint_idx == WP_V2_API:
            return [self._wp_v2_to_v3(p) for p in data]
        return data

    def _store_api_to_v3(self, p: dict) -> dict:
        prices = p.get("prices", {})
        description = re.sub(r"<[^>]+>", "", p.get("description", ""))
        description = unescape(description).strip()
        images = p.get("images", [])
        image_url = images[0].get("src", "") if images else ""
        return {
            "id": p.get("id"),
            "name": p.get("name", ""),
            "title": p.get("name", ""),
            "description": description,
            "price": str(prices.get("price", "")),
            "regular_price": str(prices.get("regular_price", "")),
            "sale_price": str(prices.get("sale_price", "")),
            "currency": prices.get("currency_code", ""),
            "image": image_url,
            "images": images,
            "categories": p.get("categories", []),
            "average_rating": p.get("average_rating", ""),
            "rating": p.get("average_rating", ""),
            "stock_status": p.get("stock_status", ""),
            "stock_quantity": p.get("quantity", ""),
            "type": p.get("type", ""),
            "url": p.get("permalink", p.get("url", "")),
            "short_description": p.get("short_description", ""),
            "sku": p.get("sku", ""),
            "tags": p.get("tags", []),
        }

    def _wp_v2_to_v3(self, p: dict) -> dict:
        title = p.get("title", {}).get("rendered", "")
        content = re.sub(r"<[^>]+>", "", p.get("content", {}).get("rendered", ""))
        content = unescape(content).strip()
        excerpt = re.sub(r"<[^>]+>", "", p.get("excerpt", {}).get("rendered", ""))
        excerpt = unescape(excerpt).strip()
        meta = p.get("meta", {})
        return {
            "id": p.get("id"),
            "name": title,
            "title": title,
            "description": content or excerpt,
            "price": str(meta.get("_price", meta.get("_regular_price", ""))),
            "regular_price": str(meta.get("_regular_price", "")),
            "sale_price": str(meta.get("_sale_price", "")),
            "sku": meta.get("_sku", ""),
            "stock_quantity": meta.get("_stock", ""),
            "stock_status": meta.get("_stock_status", ""),
            "image": p.get("featured_media_url", ""),
            "url": p.get("link", ""),
            "categories": p.get("_embedded", {}).get("wp:term", []),
        }

    def _extract_rating(self, raw: dict) -> str:
        return str(raw.get("average_rating", raw.get("rating", "")))

    def _extract_stock(self, raw: dict) -> str:
        return str(raw.get("stock_quantity") or raw.get("stock_status", ""))

    def _extract_category(self, raw: dict) -> str:
        cats = raw.get("categories", [])
        return ", ".join(c.get("name", "") for c in cats)

    def _extract_image(self, raw: dict) -> str:
        images = raw.get("images", [])
        if images:
            if isinstance(images[0], dict):
                return images[0].get("src", images[0].get("url", ""))
            return str(images[0])
        return raw.get("image", "")

    def _extract_price(self, raw: dict) -> str:
        return str(raw.get("price", ""))

    def _extract_price_promo(self, raw: dict) -> str:
        return str(raw.get("sale_price", ""))

    def _extract_price_old(self, raw: dict) -> str:
        return str(raw.get("regular_price", ""))


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else input("URL boutique WooCommerce : ")
    key = sys.argv[2] if len(sys.argv) > 2 else ""
    secret = sys.argv[3] if len(sys.argv) > 3 else ""
    scraper = WooCommerceScraper(base_url=url, consumer_key=key, consumer_secret=secret)
    scraper.scrape()
