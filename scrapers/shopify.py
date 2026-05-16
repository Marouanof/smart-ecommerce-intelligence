import re
import json
from urllib.parse import urljoin

import requests

from scrapers.base import BaseScraper

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


class ShopifyScraper(BaseScraper):
    source_name = "shopify"
    _use_playwright = False

    def __init__(self, store_domain: str, max_pages: int | None = None):
        self.store_domain = store_domain
        raw = store_domain.strip()
        if raw.startswith("http://") or raw.startswith("https://"):
            host = urlparse(raw).hostname or raw
        elif "." not in raw:
            host = f"{raw}.myshopify.com"
        else:
            host = raw
        self.store_url = f"https://{host}"
        self.api_url = f"{self.store_url}/products.json"
        self.base_url = self.store_url
        self.max_pages = max_pages
        print(f"[ShopifyScraper] URL boutique: {self.store_url}")
        print(f"[ShopifyScraper] API URL: {self.api_url}")

    def fetch_page(self, page: int) -> list[dict]:
        if not self._use_playwright:
            try:
                url = f"{self.api_url}?page={page}&limit={self.page_size}"
                print(f"[DEBUG] Requête GET {url}")
                resp = requests.get(
                    self.api_url,
                    params={"page": page, "limit": self.page_size},
                    timeout=30,
                )
                print(f"[DEBUG] Statut HTTP: {resp.status_code}")
                print(f"[DEBUG] URL finale: {resp.url}")
                if resp.status_code == 200:
                    text = resp.text[:2000]
                    print(f"[DEBUG] Réponse brute (début): {text[:300]}")
                    data = resp.json()
                    print(f"[DEBUG] Clés JSON: {list(data.keys())}")
                    print(f"[DEBUG] Type de 'products': {type(data.get('products'))}")
                    if isinstance(data.get("products"), list):
                        print(f"[DEBUG] Nombre de produits: {len(data['products'])}")
                        if data["products"]:
                            print(f"[DEBUG] 1er produit keys: {list(data['products'][0].keys())}")
                    else:
                        print(f"[DEBUG] Contenu JSON complet: {json.dumps(data, indent=2)[:1000]}")
                        return []
                    return data["products"]
                else:
                    print(f"[DEBUG] Corps réponse: {resp.text[:500]}")
            except Exception:
                pass
            self._use_playwright = True

        return self._fetch_with_playwright(page)

    def _fetch_with_playwright(self, page: int) -> list[dict]:
        if sync_playwright is None:
            return []
        if page > 1:
            return []

        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                ctx = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                )
                p = ctx.new_page()
                p.goto(self.store_url, timeout=30000)
                html = p.content()
                browser.close()
        except Exception:
            return []

        return self._parse_html(html)

    def _parse_html(self, html: str) -> list[dict]:
        if BeautifulSoup is None:
            return []

        soup = BeautifulSoup(html, "html.parser")
        products = []

        scripts = soup.select("script[type='application/ld+json'], script[type='application/json']")
        for script in scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get("@type") == "Product":
                    products.append(self._ld_to_dict(data))
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get("@type") == "Product":
                            products.append(self._ld_to_dict(item))
                elif isinstance(data, dict) and "product" in data:
                    p = data["product"]
                    if isinstance(p, dict):
                        products.append(self._shopify_json_to_dict(p))
            except Exception:
                continue

        if not products:
            cards = soup.select(
                ".product-card, .product-item, [class*='product'], "
                "[data-product], article.product, li.product"
            )
            for card in cards:
                products.append(self._card_to_dict(card))

        return products

    def _ld_to_dict(self, data: dict) -> dict:
        offers = data.get("offers", {})
        if isinstance(offers, list):
            offers = offers[0] if offers else {}
        price = offers.get("price", "")
        return {
            "id": data.get("sku", ""),
            "title": data.get("name", ""),
            "description": (data.get("description") or "").strip(),
            "price": str(price),
            "image": data.get("image", ""),
            "url": data.get("url", ""),
            "brand": data.get("brand", {}).get("name", "") if isinstance(data.get("brand"), dict) else str(data.get("brand", "")),
            "currency": offers.get("priceCurrency", ""),
            "availability": "in stock" if "InStock" in str(offers.get("availability", "")) else "",
        }

    def _shopify_json_to_dict(self, data: dict) -> dict:
        variants = data.get("variants", [{}])
        return {
            "id": str(data.get("id", "")),
            "title": data.get("title", ""),
            "description": (data.get("description") or "").strip(),
            "price": str(variants[0].get("price", "")),
            "price_promo": str(variants[0].get("compare_at_price", "")),
            "image": self._extract_image(data),
            "url": urljoin(self.store_url, data.get("url", "")),
            "vendor": data.get("vendor", ""),
            "published_at": data.get("published_at", ""),
        }

    def _card_to_dict(self, card) -> dict:
        a = card.select_one("a[href]")
        href = a.get("href", "") if a else ""
        img = card.select_one("img")
        price_el = card.select_one(".price, [class*='price'], .amount")
        title_el = card.select_one("h2, h3, h4, .title, .name, [class*='title']")

        return {
            "id": card.get("id", "").replace("product-", "").replace("ProductJson-", ""),
            "title": title_el.get_text(strip=True) if title_el else "",
            "description": "",
            "price": re.sub(r"[^\d.,]", "", price_el.get_text(strip=True)) if price_el else "",
            "image": img.get("src") or img.get("data-src", "") if img else "",
            "url": urljoin(self.store_url, href) if href else "",
        }

    def _extract_image(self, raw: dict) -> str:
        images = raw.get("images", [])
        if isinstance(images, list):
            for img in images:
                if isinstance(img, dict):
                    return img.get("src", img.get("url", ""))
                return str(img)
        return raw.get("image", "")


if __name__ == "__main__":
    import sys
    domain = sys.argv[1] if len(sys.argv) > 1 else input("Nom de domaine Shopify (ex: mon-boutique) : ")
    scraper = ShopifyScraper(store_domain=domain)
    scraper.scrape()
