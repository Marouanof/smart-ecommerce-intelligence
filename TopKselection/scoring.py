import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "price": -0.2,
    "rating": 0.4,
    "review_count": 0.3,
    "stock_quantity": 0.1,
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _detect_norm_cols(df: pd.DataFrame) -> dict[str, str]:
    mapping = {}
    for col in df.columns:
        if col.endswith("_norm"):
            base = col.replace("_norm", "")
            mapping[base] = col
    return mapping


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(abs(v) for v in weights.values())
    if total == 0:
        return {k: 0.0 for k in weights}
    return {k: v / total for k, v in weights.items()}


def calculate_composite_score(
    df: pd.DataFrame,
    weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    result = df.copy()
    norm_cols = _detect_norm_cols(result)

    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    w = _normalize_weights(w)

    score = pd.Series(0.0, index=result.index, dtype=float)
    components = {}

    for base, weight in w.items():
        col = norm_cols.get(base)
        if col is None:
            col = base
        if col not in result.columns:
            print(f"[scoring] colonne '{col}' introuvable, poids ignoré")
            continue
        values = pd.to_numeric(result[col], errors="coerce").fillna(0)
        contribution = values * weight
        score += contribution
        components[f"{base}_contrib"] = contribution

    result["composite_score"] = score
    for k, v in components.items():
        result[k] = v

    print(f"[scoring] poids normalisés: {w}")
    print(f"[scoring] score range: [{score.min():.4f}, {score.max():.4f}]")
    return result


def get_top_k_products(
    df: pd.DataFrame,
    k: int = 100,
    score_col: str = "composite_score",
) -> pd.DataFrame:
    if score_col not in df.columns:
        raise KeyError(f"Colonne '{score_col}' absente du DataFrame")
    sorted_df = df.sort_values(score_col, ascending=False).head(k).reset_index(drop=True)
    sorted_df["rank"] = range(1, len(sorted_df) + 1)
    print(f"[top_k] {len(sorted_df)} produits sélectionnés (k={k})")
    return sorted_df


def save_top_k(df_top: pd.DataFrame, output_path: Optional[str] = None) -> str:
    path = output_path or os.path.join(OUTPUT_DIR, "top_k_products.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_top.to_csv(path, index=False, encoding="utf-8")
    print(f"[save] Top-K sauvegardé -> {path}")
    return path


def analyze_score_distribution(df: pd.DataFrame, score_col: str = "composite_score") -> dict:
    if score_col not in df.columns:
        raise KeyError(f"Colonne '{score_col}' absente du DataFrame")

    s = pd.to_numeric(df[score_col], errors="coerce").dropna()
    stats = {
        "count": int(len(s)),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
        "missing": int(df[score_col].isna().sum()),
    }

    print(f"\n{'='*50}")
    print(f"Distribution du score '{score_col}'")
    print(f"{'='*50}")
    print(f"  Nombre          : {stats['count']}")
    print(f"  Manquants        : {stats['missing']}")
    print(f"  Min              : {stats['min']:.4f}")
    print(f"  Max              : {stats['max']:.4f}")
    print(f"  Moyenne          : {stats['mean']:.4f}")
    print(f"  Médiane          : {stats['median']:.4f}")
    print(f"  Écart-type       : {stats['std']:.4f}")
    print(f"  Q1 (25%)         : {stats['q25']:.4f}")
    print(f"  Q3 (75%)         : {stats['q75']:.4f}")
    print(f"{'='*50}\n")

    return stats


class ProductScorer:
    def __init__(self, weights: Optional[dict[str, float]] = None):
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    def calculate_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        norm_cols = _detect_norm_cols(result)
        w = _normalize_weights(self.weights)
        score = pd.Series(0.0, index=result.index, dtype=float)
        for base, weight in w.items():
            col = norm_cols.get(base, base)
            if col not in result.columns:
                print(f"[scoring] colonne '{col}' introuvable, poids ignoré")
                continue
            values = pd.to_numeric(result[col], errors="coerce").fillna(0)
            score += values * weight
        result["composite_score"] = score
        print(f"[scoring] score range: [{score.min():.4f}, {score.max():.4f}]")
        return result

    def get_top_k_products(self, df: pd.DataFrame, k: int = 100) -> pd.DataFrame:
        return get_top_k_products(df, k=k)

    def save_top_k(self, df_top: pd.DataFrame, output_path: Optional[str] = None) -> str:
        return save_top_k(df_top, output_path)

    def analyze_score_distribution(self, df: pd.DataFrame) -> dict:
        return analyze_score_distribution(df)


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data/processed/cleaned_products.csv")

    df = pd.read_csv(input_path)
    print(f"Chargé: {len(df)} lignes")

    scorer = ProductScorer()
    df = scorer.calculate_composite_score(df)
    df_top = scorer.get_top_k_products(df, k=100)
    scorer.save_top_k(df_top)
    scorer.analyze_score_distribution(df)

    output_full = os.path.join(base_dir, "data/processed/products_with_score.csv")
    df.to_csv(output_full, index=False)
    print(f"DataFrame complet avec score sauvegardé dans {output_full}")
