from llm.enrichment import enrichir_description, analyser_tendance, generer_rapport

produits = [
    {"name": "Écouteurs Bluetooth Pro", "description": "Casque sans fil avec réduction de bruit active, batterie 30h, étui de charge inclus.", "category": "Audio", "price": 79.99, "score": 4.8},
    {"name": "Montre Connectée Sport", "description": "Montre intelligente GPS, cardiofréquencemètre, 50mm étanche.", "category": "Wearables", "price": 149.99, "score": 4.5},
    {"name": "Lampe LED Bureau", "description": "Lampe de bureau LED avec gradation tactile, lumière naturelle sans scintillement.", "category": "Maison", "price": 34.99, "score": 4.2},
    {"name": "Chargeur Solaire 20W", "description": "Panneau solaire pliable double port USB pour smartphones et tablettes.", "category": "Accessoires", "price": 29.99, "score": 4.0},
    {"name": "Sac à Dos Urbain", "description": "Sac à dos 30L avec port USB intégré, anti-vol, imperméable.", "category": "Mode", "price": 54.99, "score": 4.6},
    {"name": "Enceinte Portable Bluetooth", "description": "Enceinte waterproof 360°, 20h autonomie, format poche.", "category": "Audio", "price": 39.99, "score": 4.3},
    {"name": "Kit Domotique Débutant", "description": "Pack 3 ampoules connectées + hub WiFi compatible Alexa/Google.", "category": "Maison", "price": 89.99, "score": 4.1},
    {"name": "Support Téléphone Voiture", "description": "Support magnétique rotatif pour tableau de bord, fixation universelle.", "category": "Accessoires", "price": 14.99, "score": 4.7},
    {"name": "Coussin Cervical Mémoire", "description": "Oreiller ergonomique en mousse à mémoire de forme avec housse lavable.", "category": "Maison", "price": 44.99, "score": 4.4},
    {"name": "Câble USB-C Tressé 2m", "description": "Câble rapide 100W charge et transfert, tressé nylon renforcé.", "category": "Accessoires", "price": 9.99, "score": 4.9},
]

if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1 : enrichir_description (1 produit)")
    print("=" * 60)
    res1 = enrichir_description(produits[0])
    print(f"Résultat : {res1}\n")

    print("=" * 60)
    print("TEST 2 : analyser_tendance (10 produits)")
    print("=" * 60)
    res2 = analyser_tendance(produits)
    for i, t in enumerate(res2, 1):
        print(f"  {i}. {t}")
    print()

    print("=" * 60)
    print("TEST 3 : generer_rapport (top 10)")
    print("=" * 60)
    res3 = generer_rapport(produits)
    print(res3)
    print()

    print("Tests terminés.")
