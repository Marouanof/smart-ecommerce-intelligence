import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Optional


CLUSTER_FEATURES = [
    "price_norm",
    "rating_norm",
    "review_count_norm",
    "stock_quantity_norm",
    "engagement",
    "prix_par_rating",
    "log_reviews",
    "disponible",
]


class ClusteringAnalyzer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.metrics = {}

    def prepare_clustering_data(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in CLUSTER_FEATURES if c in df.columns]
        missing = [c for c in CLUSTER_FEATURES if c not in df.columns]
        if missing:
            print(f"[prepare] colonnes manquantes: {missing}")
        return df[available].copy()

    def kmeans_clustering(
        self,
        df: pd.DataFrame,
        n_clusters: int = 3,
        features: Optional[pd.DataFrame] = None,
    ) -> float:
        X = features if features is not None else self.prepare_clustering_data(df)
        model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
        labels = model.fit_predict(X)
        df["cluster_kmeans"] = labels
        score = silhouette_score(X, labels)
        self.models["kmeans"] = model
        self.metrics["kmeans"] = score
        print(f"[kmeans] n_clusters={n_clusters}, silhouette={score:.4f}")
        return score

    def dbscan_clustering(
        self,
        df: pd.DataFrame,
        eps: float = 0.5,
        min_samples: int = 5,
        features: Optional[pd.DataFrame] = None,
    ) -> float:
        X = features if features is not None else self.prepare_clustering_data(df)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        df["cluster_dbscan"] = labels
        n_clusters = len(set(labels) - {-1})
        n_noise = list(labels).count(-1)
        print(f"[dbscan] eps={eps}, min_samples={min_samples}, clusters={n_clusters}, bruit={n_noise}")
        if n_clusters > 1:
            mask = labels != -1
            score = silhouette_score(X[mask], labels[mask])
            self.models["dbscan"] = model
            self.metrics["dbscan"] = score
            print(f"[dbscan] silhouette={score:.4f}")
            return score
        self.metrics["dbscan"] = -1.0
        return -1.0

    def hierarchical_clustering(
        self,
        df: pd.DataFrame,
        n_clusters: int = 3,
        features: Optional[pd.DataFrame] = None,
    ) -> float:
        X = features if features is not None else self.prepare_clustering_data(df)
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        df["cluster_hierarchical"] = labels
        score = silhouette_score(X, labels)
        self.models["hierarchical"] = model
        self.metrics["hierarchical"] = score
        print(f"[hierarchical] n_clusters={n_clusters}, silhouette={score:.4f}")
        return score

    def pca_visualization(
        self,
        df: pd.DataFrame,
        cluster_col: str = "cluster_kmeans",
        features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        X = features if features is not None else self.prepare_clustering_data(df)
        pca = PCA(n_components=2, random_state=self.random_state)
        components = pca.fit_transform(X)
        result = pd.DataFrame(components, columns=["pca_1", "pca_2"])
        result[cluster_col] = df[cluster_col].values if cluster_col in df.columns else 0
        var_explained = pca.explained_variance_ratio_
        print(f"[pca] variance expliquée: {var_explained[0]:.2%}, {var_explained[1]:.2%}")
        return result

    def plot_clusters_pca(self, df_pca: pd.DataFrame, cluster_col: str = "cluster_kmeans"):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df_pca,
            x="pca_1",
            y="pca_2",
            hue=cluster_col,
            palette="Set2",
            s=50,
            alpha=0.8,
            edgecolor="k",
            linewidth=0.5,
        )
        plt.title(f"Visualisation PCA — {cluster_col}")
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.legend(title="Cluster")
        plt.tight_layout()
        plt.show()

    def analyze_segments(
        self,
        df: pd.DataFrame,
        cluster_col: str = "cluster_kmeans",
    ) -> pd.DataFrame:
        cols = ["price_norm", "rating_norm", "review_count_norm", "stock_quantity_norm",
                "engagement", "prix_par_rating", "log_reviews", "disponible", "composite_score"]
        available = [c for c in cols if c in df.columns]
        grouped = df.groupby(cluster_col)[available].mean().round(4)
        grouped["count"] = df.groupby(cluster_col).size()
        grouped["pct"] = (grouped["count"] / len(df) * 100).round(1)
        print(f"\n[segments] profil moyen par {cluster_col}")
        print(grouped.to_string())
        return grouped


def run_clustering_pipeline(df: pd.DataFrame) -> dict:
    analyzer = ClusteringAnalyzer(random_state=42)

    print("\n" + "=" * 50)
    print("Pipeline de clustering")
    print("=" * 50)

    features = analyzer.prepare_clustering_data(df)
    print(f"[pipeline] {features.shape[1]} features, {features.shape[0]} produits")

    best_kmeans_score = -1
    best_kmeans_k = 3
    for k in [3, 4, 5]:
        df_temp = df.copy()
        score = analyzer.kmeans_clustering(df_temp, n_clusters=k, features=features)
        if score > best_kmeans_score:
            best_kmeans_score = score
            best_kmeans_k = k
            df["cluster_kmeans"] = df_temp["cluster_kmeans"]

    print(f"\n>> Meilleur KMeans: k={best_kmeans_k}, silhouette={best_kmeans_score:.4f}")

    for eps in [0.3, 0.5, 0.7]:
        df_temp = df.copy()
        score = analyzer.dbscan_clustering(df_temp, eps=eps, features=features)
        if score > analyzer.metrics.get("dbscan", -1):
            df["cluster_dbscan"] = df_temp["cluster_dbscan"]

    analyzer.hierarchical_clustering(df, n_clusters=3, features=features)

    df_pca = analyzer.pca_visualization(df, cluster_col="cluster_kmeans", features=features)

    print("\n" + "=" * 50)
    print("Profils des clusters (KMeans)")
    print("=" * 50)
    profiles = analyzer.analyze_segments(df, cluster_col="cluster_kmeans")

    interpretations = _interpret_clusters(profiles)
    for cluster_id, label in interpretations.items():
        print(f"  cluster {cluster_id} : {label}")

    return {
        "analyzer": analyzer,
        "df_pca": df_pca,
        "interpretations": interpretations,
        "features": features,
    }


def _interpret_clusters(profiles: pd.DataFrame) -> dict:
    interpretations = {}
    if profiles.empty:
        return interpretations

    price_col = "price_norm" if "price_norm" in profiles.columns else None
    rating_col = "rating_norm" if "rating_norm" in profiles.columns else None
    stock_col = "stock_quantity_norm" if "stock_quantity_norm" in profiles.columns else None

    for cluster_id in profiles.index:
        row = profiles.loc[cluster_id]
        tags = []
        if price_col is not None:
            if row[price_col] > 0.7:
                tags.append("premium")
            elif row[price_col] < 0.3:
                tags.append("discount")
            else:
                tags.append("mid-range")
        if rating_col is not None and row[rating_col] > 0.7:
            tags.append("top-rated")
        if stock_col is not None and row[stock_col] > 0.7:
            tags.append("high-stock")
        interpretations[int(cluster_id)] = ", ".join(tags) if tags else "standard"
    return interpretations


if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data/processed/products_with_score.csv")

    df = pd.read_csv(input_path)
    print(f"Chargé: {len(df)} lignes depuis {input_path}")

    results = run_clustering_pipeline(df)

    output_path = os.path.join(base_dir, "data/processed/products_with_clusters.csv")
    df.to_csv(output_path, index=False)
    print(f"\nDataFrame enrichi sauvegarde -> {output_path}")

    print("\n" + "=" * 50)
    print("Résumé des scores de silhouette")
    print("=" * 50)
    for method, score in results["analyzer"].metrics.items():
        if score > 0:
            print(f"  {method:15s}: {score:.4f}")
        else:
            print(f"  {method:15s}: N/A")

    print("\nInterprétations:")
    for cluster_id, label in results["interpretations"].items():
        print(f"  cluster {cluster_id} : {label}")
