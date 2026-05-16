import os
import json
import time
import hashlib
from typing import Optional

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_URL = "https://api-inference.huggingface.co/models/{model}"
DEFAULT_MODEL = "microsoft/phi-2"
FALLBACK_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

_cache: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 300
_last_request_time: float = 0
_MIN_INTERVAL = 1.0


def _cache_key(text: str, prefix: str) -> str:
    return f"{prefix}:{hashlib.md5(text.encode()).hexdigest()}"


def _get_from_cache(key: str) -> Optional[str]:
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < _CACHE_TTL:
        return entry[1]
    if entry:
        del _cache[key]
    return None


def _set_cache(key: str, value: str):
    _cache[key] = (time.time(), value)


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


def _query_hf(prompt: str, model: str = DEFAULT_MODEL) -> Optional[str]:
    global _TOKEN
    ckey = _cache_key(prompt, f"hf:{model}")
    cached = _get_from_cache(ckey)
    if cached:
        return cached

    headers = {}
    if _TOKEN:
        headers["Authorization"] = f"Bearer {_TOKEN}"

    _rate_limit()
    try:
        resp = requests.post(
            API_URL.format(model=model),
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 200, "temperature": 0.3}},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            raw = ""
            if isinstance(data, list) and len(data) > 0:
                raw = data[0].get("generated_text", str(data[0]))
            elif isinstance(data, dict):
                raw = data.get("generated_text", str(data))
            else:
                raw = str(data)
            text = raw[len(prompt):].strip()
            _set_cache(ckey, text)
            return text
        elif resp.status_code == 503 and model == DEFAULT_MODEL:
            return _query_hf(prompt, model=FALLBACK_MODEL)
        return None
    except requests.RequestException:
        return None


def enrichir_description(produit: dict) -> str:
    name = produit.get("name", produit.get("title", "Produit inconnu"))
    desc = produit.get("description", "")
    categ = produit.get("category", "")

    prompt = (
        f"Résume ce produit en 1-2 phrases concises.\n"
        f"Raisonnement : identifie d'abord les caractéristiques clés, "
        f"puis synthétise-les en une phrase fluide.\n"
        f"Nom: {name}\n"
        f"Description: {desc}\n"
        f"Catégorie: {categ}\n"
        f"Résumé:"
    )
    result = _query_hf(prompt)
    if result:
        return result
    return _fallback_enrichir(produit)


def _fallback_enrichir(produit: dict) -> str:
    name = produit.get("name", produit.get("title", "Produit inconnu"))
    desc = produit.get("description", "")
    if desc:
        words = desc.split()[:20]
        short_desc = " ".join(words) + ("…" if len(desc.split()) > 20 else "")
        return f"{name} : {short_desc}"
    return name


def analyser_tendance(produits: list[dict]) -> list[str]:
    if not produits:
        return ["Aucune donnée disponible"]

    prompt_lines = [
        "Analyse les tendances produit suivantes étape par étape :",
        "1. Examine chaque produit et sa catégorie.",
        "2. Regroupe les produits par similarité.",
        "3. Identifie les 3 tendances principales avec justification.",
        "",
    ]
    for p in produits[:20]:
        name = p.get("name", p.get("title", ""))
        categ = p.get("category", "")
        prompt_lines.append(f"- {name} ({categ})")
    prompt_lines.append("\nTop 3 tendances:")
    prompt = "\n".join(prompt_lines)

    result = _query_hf(prompt)
    if result:
        lines = [line.strip("- •0123456789. ") for line in result.split("\n") if line.strip()]
        trends = [l for l in lines if len(l) > 5][:3]
        if trends:
            return trends
    return _fallback_tendance(produits)


def _fallback_tendance(produits: list[dict]) -> list[str]:
    from collections import Counter
    categories = Counter()
    keywords = Counter()
    for p in produits:
        cat = p.get("category", "Général")
        categories[cat] += 1
        for word in (p.get("name", "") + " " + p.get("description", "")).lower().split():
            if len(word) > 4:
                keywords[word] += 1
    top_cats = [c for c, _ in categories.most_common(2)]
    top_kw = keywords.most_common(1)
    trends = [f"Catégorie dominante : {top_cats[0]}" if top_cats else "N/A"]
    if len(top_cats) > 1:
        trends.append(f"Catégorie émergente : {top_cats[1]}")
    if top_kw:
        trends.append(f"Mot-clé récurrent : {top_kw[0][0]}")
    return trends[:3]


def generer_rapport(top10: list[dict]) -> str:
    if not top10:
        return "Aucun produit à rapporter."

    prompt_lines = [
        "Génère un rapport stratégique concis à partir de ce top 10 produits.\n"
        "Raisonnement : analyse d'abord les tendances de prix, notes et catégories, "
        "puis formule des recommandations justifiées.\n",
    ]
    for i, p in enumerate(top10[:10], 1):
        name = p.get("name", p.get("title", ""))
        price = p.get("price", "N/A")
        score = p.get("score", p.get("rating", "N/A"))
        prompt_lines.append(f"{i}. {name} - {price}€ - Score: {score}")
    prompt_lines.append(
        "\nRapport stratégique (paragraphe concis avec recommandations) :"
    )
    prompt = "\n".join(prompt_lines)

    result = _query_hf(prompt, model=FALLBACK_MODEL)
    if result:
        return result
    return _fallback_rapport(top10)


def _fallback_rapport(top10: list[dict]) -> str:
    total = len(top10)
    avg_price = 0.0
    best = None
    for p in top10:
        price = p.get("price")
        if price:
            try:
                avg_price += float(price)
            except (ValueError, TypeError):
                pass
        score = p.get("score", p.get("rating"))
        if score and (best is None or score > best[1]):
            best = (p.get("name", p.get("title", "")), score)
    avg_price = avg_price / total if total else 0
    lines = [f"Rapport stratégique - {total} produits analysés"]
    lines.append(f"Prix moyen : {avg_price:.2f}€")
    if best:
        lines.append(f"Meilleur score : {best[0]} ({best[1]})")
    lines.append("Recommandation : Concentrer les efforts marketing sur les produits les mieux notés.")
    return "\n".join(lines)
