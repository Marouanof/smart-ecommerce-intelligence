import glob
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

COLUMN_ALIASES = {
    "price": ["price", "prix", "Prix", "PRICE", "unit_price"],
    "rating": ["rating", "note", "Note", "RATING", "score", "Score", "average_rating"],
    "review_count": ["review_count", "review_count", "reviews", "Reviews", "nb_reviews", "customer_reviews"],
    "reviews": ["review_count", "review_count", "reviews", "Reviews", "nb_reviews", "customer_reviews"],
    "stock_quantity": ["stock_quantity", "stock", "Stock", "inventory", "quantity", "inventory_quantity"],
    "availability": ["availability", "disponibilite", "stock_status", "disponible"],
    "title": ["title", "nom", "name", "Name", "Titre", "product_name", "product_title"],
    "category": ["category", "categorie", "Category", "catégorie"],
    "product_id": ["product_id", "id", "ID", "Id", "ProductId"],
    "traffic": ["traffic", "ventes", "sales", "visits"],
    "shop_country": ["shop_country", "geographie", "country", "pays"],
}

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def _resolve_col(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in cols_lower:
            return cols_lower[alias.lower()]
        for col in df.columns:
            if alias.lower() == col.lower():
                return col
    return None


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    pattern = path or os.path.join(RAW_DIR, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {pattern}")

    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=str, keep_default_na=True)
        df["_source_file"] = os.path.basename(f)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["product_id", "title"], keep="last")
    print(f"[load_data] {len(files)} fichiers, {len(combined)} produits chargés")
    return combined


def _to_numeric(col: pd.Series) -> pd.Series:
    out = col.replace("None", np.nan).replace("", np.nan)
    out = out.str.replace(",", ".", regex=False)
    out = pd.to_numeric(out, errors="coerce")
    return out


class DataPreprocessor:
    def __init__(self):
        self.column_mapping = COLUMN_ALIASES

    def _find_column(self, df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
        return _resolve_col(df, aliases)

    def clean_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        price_col = self._find_column(result, self.column_mapping["price"])
        rating_col = self._find_column(result, self.column_mapping["rating"])
        reviews_col = self._find_column(result, self.column_mapping["review_count"])
        stock_col = self._find_column(result, self.column_mapping["stock_quantity"])
        avail_col = self._find_column(result, self.column_mapping["availability"])

        if price_col:
            result[price_col] = _to_numeric(result[price_col])
            med = result[price_col].median()
            result[price_col] = result[price_col].fillna(med if pd.notna(med) else 0)

        if rating_col:
            result[rating_col] = _to_numeric(result[rating_col])
            if result[rating_col].isna().all() or (result[rating_col] == 0).all():
                print("[WARNING] Tous les ratings sont manquants. Simulation de notes réalistes.")
                np.random.seed(42)
                result[rating_col] = np.random.normal(loc=4.3, scale=0.5, size=len(result))
                result[rating_col] = result[rating_col].clip(2.5, 5.0).round(1)
            else:
                med = result[rating_col].median()
                result[rating_col] = result[rating_col].fillna(med if pd.notna(med) else 4.0)

        if reviews_col:
            result[reviews_col] = _to_numeric(result[reviews_col])
            if result[reviews_col].isna().all() or (result[reviews_col] <= 1).all():
                print("[WARNING] Tous les review_count sont manquants. Simulation réaliste log-normale.")
                np.random.seed(42)
                result[reviews_col] = np.random.lognormal(mean=4.0, sigma=1.5, size=len(result)).astype(int)
            else:
                result[reviews_col] = result[reviews_col].fillna(0)

        if stock_col:
            result[stock_col] = _to_numeric(result[stock_col])
            result[stock_col] = result[stock_col].fillna(0)

        if avail_col:
            result[avail_col] = result[avail_col].replace("None", np.nan).fillna("unknown")

        # Simulation Géographie et Trafic si manquants (CDC)
        country_col = self._find_column(result, self.column_mapping["shop_country"])
        if country_col:
            if result[country_col].isna().all() or (result[country_col] == "").all():
                def get_geo(url):
                    u = str(url).lower()
                    if "gymshark" in u: return "UK"
                    if "culturekings" in u: return "AU"
                    return "US"
                url_col = self._find_column(result, ["url", "link"])
                if url_col:
                    result[country_col] = result[url_col].apply(get_geo)
                else:
                    result[country_col] = "US"

        traffic_col = self._find_column(result, self.column_mapping["traffic"])
        if not traffic_col:
            traffic_col = "traffic"
            result[traffic_col] = np.nan
        if result[traffic_col].isna().all():
            np.random.seed(42)
            # Trafic proportionnel aux reviews
            result[traffic_col] = result[reviews_col] * np.random.randint(10, 100, size=len(result))

        for col in result.columns:
            if result[col].dtype == object:
                result[col] = result[col].replace("None", np.nan).fillna("")

        print(f"[clean] values imputed -> price={price_col}, rating={rating_col}, reviews={reviews_col}, stock={stock_col}")

        if price_col and rating_col:
            result = result[result[price_col] > 0]
            result = result[result[price_col] < 10000]
            result = result[result[rating_col] > 0]

        print(f"[filter] {len(result)} produits après filtrage (prix>0, prix<10000, rating>0)")
        return result

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return normalize_features(df)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return create_features(df)

    def load_data(self, path: Optional[str] = None) -> pd.DataFrame:
        return load_data(path)

    def run(
        self,
        input_pattern: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df = self.load_data(input_pattern)
        df = self.clean_missing_values(df)
        df = self.normalize_features(df)
        df = self.create_features(df)
        out = output_path or os.path.join(PROCESSED_DIR, "cleaned_products.csv")
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"[run] {len(df)} produits sauvegardes -> {out}")
        return df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return DataPreprocessor().clean_missing_values(df)


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    scaler = MinMaxScaler()
    normalized_cols = []

    for alias_key in ["price", "rating", "review_count", "stock_quantity"]:
        col = _resolve_col(result, COLUMN_ALIASES[alias_key])
        if col and col in result.columns:
            numeric = pd.to_numeric(result[col], errors="coerce").fillna(0)
            result[f"{col}_norm"] = scaler.fit_transform(numeric.values.reshape(-1, 1)).flatten()
            normalized_cols.append(f"{col}_norm")

    print(f"[normalize] colonnes normalisées: {normalized_cols}")
    return result


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    price_col = _resolve_col(result, COLUMN_ALIASES["price"])
    rating_col = _resolve_col(result, COLUMN_ALIASES["rating"])
    reviews_col = _resolve_col(result, COLUMN_ALIASES["review_count"])
    stock_col = _resolve_col(result, COLUMN_ALIASES["stock_quantity"])
    avail_col = _resolve_col(result, COLUMN_ALIASES["availability"])

    price = pd.to_numeric(result.get(price_col, pd.Series(0)), errors="coerce").fillna(0)
    rating = pd.to_numeric(result.get(rating_col, pd.Series(0)), errors="coerce").fillna(0)
    reviews = pd.to_numeric(result.get(reviews_col, pd.Series(0)), errors="coerce").fillna(0)
    stock = pd.to_numeric(result.get(stock_col, pd.Series(0)), errors="coerce").fillna(0)

    rating_safe = rating.replace(0, np.nan)
    result["prix_par_rating"] = np.where(rating_safe.notna(), price / rating_safe, price)
    result["prix_par_rating"] = result["prix_par_rating"].replace([np.inf, -np.inf], np.nan).fillna(0)

    result["engagement"] = reviews * rating

    if avail_col:
        avail_str = result[avail_col].astype(str).str.lower()
        result["disponible"] = np.where(
            (stock > 0) | (avail_str.str.contains("in.stock|true|yes|disponible", na=False)),
            1, 0,
        ).astype(int)
    else:
        result["disponible"] = (stock > 0).astype(int)

    result["log_reviews"] = np.log1p(reviews)

    print(f"[features] créées: prix_par_rating, engagement, disponible, log_reviews")
    return result


def run(
    input_pattern: Optional[str] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_data(input_pattern)
    df = clean_missing_values(df)
    df = normalize_features(df)
    df = create_features(df)

    out = output_path or os.path.join(PROCESSED_DIR, "cleaned_products.csv")
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"[run] {len(df)} produits sauvegardés -> {out}")
    return df


if __name__ == "__main__":
    run()
