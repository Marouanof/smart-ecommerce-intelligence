import csv
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import re

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


@dataclass
class Product:
    product_id: str = ""
    title: str = ""
    description: str = ""
    category: str = ""
    subcategory: str = ""
    brand: str = ""
    image_url: str = ""
    tags: str = ""
    url: str = ""
    platform: str = ""
    price: str = ""
    price_promo: str = ""
    price_old: str = ""
    discount_pct: str = ""
    currency: str = ""
    rating: str = ""
    review_count: str = ""
    category_rank: str = ""
    availability: str = ""
    stock_quantity: str = ""
    delivery_days: str = ""
    variant_count: str = ""
    colors: str = ""
    sizes: str = ""
    shop_name: str = ""
    shop_country: str = ""
    shop_product_count: str = ""
    vendor: str = ""
    related_products: str = ""
    published_at: str = ""
    scraped_at: str = ""
    customer_reviews: str = ""

    @property
    def is_on_sale(self) -> bool:
        return bool(self.price_promo) and self.price_promo != self.price

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


FIELD_NAMES = list(Product.__dataclass_fields__.keys())


class BaseScraper(ABC):
    source_name: str = "base"
    page_size: int = 50
    max_pages: Optional[int] = None

    @abstractmethod
    def fetch_page(self, page: int) -> list[dict]:
        ...

    def normalize(self, raw: dict) -> Product:
        return Product(
            product_id=str(raw.get("id", "")),
            title=raw.get("name") or raw.get("title", ""),
            description=(raw.get("description") or raw.get("body_html") or "").strip(),
            category=self._extract_category(raw),
            brand=raw.get("brand", ""),
            image_url=self._extract_image(raw),
            url=raw.get("url", raw.get("link", "")),
            platform=self.source_name,
            price=self._extract_price(raw),
            price_promo=self._extract_price_promo(raw),
            price_old=self._extract_price_old(raw),
            discount_pct=self._extract_discount_pct(raw),
            currency=raw.get("currency", ""),
            rating=self._extract_rating(raw),
            review_count=str(raw.get("review_count", raw.get("reviews_count", ""))),
            availability=self._extract_availability(raw),
            stock_quantity=self._extract_stock(raw),
            vendor=raw.get("vendor", raw.get("seller", "")),
            shop_name=self._extract_shop_name(raw),
            published_at=raw.get("published_at", raw.get("date_created", "")),
            scraped_at=datetime.now().isoformat(),
        )

    def _extract_price(self, raw: dict) -> str:
        variants = raw.get("variants")
        if variants:
            return str(variants[0].get("price", ""))
        return str(raw.get("price", ""))

    def _extract_price_promo(self, raw: dict) -> str:
        variants = raw.get("variants")
        if variants:
            return str(variants[0].get("compare_at_price", ""))
        return str(raw.get("sale_price", ""))

    def _extract_price_old(self, raw: dict) -> str:
        return str(raw.get("regular_price", raw.get("original_price", "")))

    def _extract_discount_pct(self, raw: dict) -> str:
        return str(raw.get("discount", raw.get("discount_percent", "")))

    def _extract_rating(self, raw: dict) -> str:
        return str(raw.get("rating", raw.get("average_rating", "")))

    def _extract_availability(self, raw: dict) -> str:
        return raw.get("availability", raw.get("stock_status", ""))

    def _extract_stock(self, raw: dict) -> str:
        variants = raw.get("variants")
        if variants:
            return str(variants[0].get("inventory_quantity", ""))
        return str(raw.get("stock_quantity", raw.get("stock", "")))

    def _extract_category(self, raw: dict) -> str:
        cats = raw.get("categories") or raw.get("product_type")
        if isinstance(cats, list):
            return ", ".join(c.get("name", "") for c in cats)
        return cats or ""

    def _extract_image(self, raw: dict) -> str:
        images = raw.get("images", [])
        if images:
            if isinstance(images[0], dict):
                return images[0].get("src", images[0].get("url", ""))
            return str(images[0])
        return raw.get("image", raw.get("thumbnail", ""))

    def _extract_shop_name(self, raw: dict) -> str:
        return raw.get("shop_name", raw.get("store_name", raw.get("vendor", "")))

    def is_valid_product(self, product: Product) -> bool:
        # Exclure les logiciels, plugins, extensions, services dématérialisés
        software_keywords = {
            "plugin", "extension", "software", "license", "licence", "theme", 
            "download", "téléchargement", "module", "abonnement", "subscription", 
            "addon", "add-on", "api", "support", "maintenance", "digital", "virtual",
            "pdf", "ebook", "course", "formation", "app", "integration"
        }
        
        # Tronquer la description pour éviter les faux positifs lointains
        desc = product.description[:500] if product.description else ""
        text_to_check = f"{product.title} {product.category} {product.tags} {desc}".lower()
        
        for kw in software_keywords:
            if re.search(rf'\b{kw}\b', text_to_check):
                return False
                
        # Filtrer les prix aberrants (souvent des licences entreprise)
        try:
            price_val = float(product.price.replace(',', '.'))
            if price_val > 1500:
                return False
        except (ValueError, TypeError):
            pass
            
        return True

    def scrape(self):
        os.makedirs(RAW_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DIR, f"{self.source_name}_{datetime.now():%Y%m%d_%H%M%S}.csv")
        all_products: list[Product] = []

        page = 1
        while True:
            items = self.fetch_page(page)
            if not items:
                break
            for p in items:
                norm_p = self.normalize(p)
                if self.is_valid_product(norm_p):
                    all_products.append(norm_p)
            page += 1
            if self.max_pages and page > self.max_pages:
                break

        if not all_products:
            print(f"[{self.source_name}] Aucun produit trouvé.")
            return

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
            writer.writeheader()
            for p in all_products:
                writer.writerow(p.to_dict())

        print(f"[{self.source_name}] {len(all_products)} produits sauvegardés → {filepath}")
