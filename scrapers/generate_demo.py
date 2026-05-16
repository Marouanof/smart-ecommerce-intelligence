import os
import sys
import csv
import requests
from datetime import datetime

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scrapers.base import Product, FIELD_NAMES, RAW_DIR

import os
import sys
import csv
import random
from datetime import datetime

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scrapers.base import Product, FIELD_NAMES, RAW_DIR

def main():
    print("Génération des données WooCommerce de démonstration (mode local)...")
    
    # Données simulées pour respecter le CDC et le dashboard WooCommerce
    categories = ["Clothing", "Accessories", "Home", "Electronics"]
    brands = ["WooBrand", "EcoWear", "TechStyle", "HomeLiving"]
    
    products = []
    for i in range(1, 101):
        cat = random.choice(categories)
        brand = random.choice(brands)
        base_price = random.randint(10, 150)
        
        p = Product(
            product_id=f"wc_prod_{i}",
            title=f"WooCommerce {cat} Item {i}",
            description=f"Un excellent produit de la catégorie {cat}. Qualité premium par {brand}.",
            category=cat,
            brand=brand,
            image_url=f"https://example.com/wc_image_{i}.jpg",
            url=f"https://woocommerce-demo.com/product/{i}",
            platform="woocommerce",
            price=str(base_price),
            price_promo=str(base_price) if random.random() > 0.3 else str(int(base_price * 0.8)),
            stock_quantity=str(random.randint(0, 100)),
            availability="in stock" if random.random() > 0.1 else "out of stock",
            published_at=datetime.now().isoformat(),
            scraped_at=datetime.now().isoformat()
        )
        products.append(p)
        
    os.makedirs(RAW_DIR, exist_ok=True)
    filepath = os.path.join(RAW_DIR, f"woocommerce_{datetime.now():%Y%m%d_%H%M%S}.csv")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        writer.writeheader()
        for p in products:
            writer.writerow(p.to_dict())
            
    print(f"Extraction WooCommerce simulée terminée. {len(products)} produits physiques récupérés.")
    print(f"Fichier sauvegardé dans : {filepath}")

if __name__ == "__main__":
    main()
