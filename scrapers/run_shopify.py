import os
import sys

# Ajouter le chemin du projet au PYTHONPATH si nécessaire
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scrapers.orchestrator import scrape_urls

def main():
    # Liste de boutiques Shopify connues pour vendre des produits physiques avec un grand catalogue
    urls = [
        "https://fashionnova.com",
        "https://colourpop.com",
        "https://gymshark.com",
        "https://kyliecosmetics.com",
        "https://mnml.la"
    ]
    
    print("Démarrage du scraping massif de produits physiques...")
    print(f"Boutiques cibles : {', '.join(urls)}")
    
    # On limite à 40 pages max par boutique (40 * 50 = 2000 produits max par boutique)
    # Pour 5 boutiques, ça donne environ 10000 produits bruts.
    # Avec les éventuels doublons et filtrages, on devrait atteindre facilement l'objectif de 3000-5000 produits.
    max_pages = 40
    
    products = scrape_urls(urls, max_pages=max_pages)
    
    print(f"\nScraping terminé. Nombre total de produits récupérés : {len(products)}")

if __name__ == "__main__":
    main()
