import os
import sys
import io
import json
import importlib
from typing import Optional

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path)

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agents.mcp_server as _mcp_server_mod
import agents.mcp_client as _mcp_client_mod
importlib.reload(_mcp_server_mod)
importlib.reload(_mcp_client_mod)
from agents.mcp_client import MCPClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "TopKselection", "output")

st.set_page_config(page_title="Smart E-Commerce BI", layout="wide")

if "mcp_client" not in st.session_state:
    client = MCPClient()
    client.connect(role="viewer")
    st.session_state.mcp_client = client
else:
    old = st.session_state.mcp_client
    if not hasattr(old, "clear_logs") or not hasattr(old._server, "clear_logs"):
        client = MCPClient()
        client.connect(role=old.current_role if old.connected else "viewer")
        st.session_state.mcp_client = client


@st.cache_data
def load_data():
    clusters_path = os.path.join(DATA_DIR, "products_with_clusters.csv")
    topk_path = os.path.join(OUTPUT_DIR, "top_k_products.csv")

    df_clusters = pd.read_csv(clusters_path)
    df_topk = pd.read_csv(topk_path) if os.path.exists(topk_path) else df_clusters.head(100).copy()

    for col in ["composite_score", "price", "rating", "review_count", "stock_quantity"]:
        if col in df_clusters.columns:
            df_clusters[col] = pd.to_numeric(df_clusters[col], errors="coerce")
    if "review_count" in df_clusters.columns:
        df_clusters["review_count"] = df_clusters["review_count"].fillna(0).astype(int)

    return df_clusters, df_topk


def derive_country(source_file: str) -> str:
    if pd.isna(source_file):
        return "Unknown"
    sf = str(source_file).lower()
    if "shopify" in sf:
        return "United States"
    elif "woocommerce" in sf:
        return "United Kingdom"
    return "Unknown"


def sidebar_filters(df: pd.DataFrame):
    st.sidebar.header("Filtres")

    categories = ["Toutes"] + sorted(df["category"].dropna().unique().tolist())
    selected_cat = st.sidebar.selectbox("Categorie", categories)

    if "cluster_kmeans" in df.columns:
        clusters = sorted(df["cluster_kmeans"].dropna().unique().tolist())
        selected_cluster = st.sidebar.multiselect(
            "Cluster KMeans", clusters, default=clusters
        )
    else:
        selected_cluster = []

    if "platform" in df.columns:
        platforms = ["Toutes"] + sorted(df["platform"].dropna().unique().tolist())
        selected_platform = st.sidebar.selectbox("Plateforme", platforms)
    else:
        selected_platform = "Toutes"

    if "price" in df.columns:
        price_min = float(df["price"].min())
        price_max = float(df["price"].max())
        price_range = st.sidebar.slider(
            "Prix", min_value=price_min, max_value=price_max,
            value=(price_min, price_max), step=1.0,
        )
    else:
        price_range = (0, 0)

    if "composite_score" in df.columns:
        score_min = float(df["composite_score"].min())
        score_max = float(df["composite_score"].max())
        score_range = st.sidebar.slider(
            "Score composite", min_value=score_min, max_value=score_max,
            value=(score_min, score_max), step=0.01,
        )
    else:
        score_range = (0, 0)

    return selected_cat, selected_cluster, selected_platform, price_range, score_range


def apply_filters(
    df: pd.DataFrame,
    category: str,
    clusters: list,
    platform: str,
    price_range: tuple,
    score_range: tuple,
) -> pd.DataFrame:
    filtered = df.copy()

    if category != "Toutes":
        filtered = filtered[filtered["category"] == category]

    if clusters and "cluster_kmeans" in filtered.columns:
        filtered = filtered[filtered["cluster_kmeans"].isin(clusters)]

    if platform != "Toutes" and "platform" in filtered.columns:
        filtered = filtered[filtered["platform"] == platform]

    if "price" in filtered.columns:
        filtered = filtered[
            (filtered["price"] >= price_range[0]) & (filtered["price"] <= price_range[1])
        ]

    if "composite_score" in filtered.columns:
        filtered = filtered[
            (filtered["composite_score"] >= score_range[0])
            & (filtered["composite_score"] <= score_range[1])
        ]

    return filtered


def render_topk_tab(df_topk: pd.DataFrame):
    st.subheader("Top-K Produits")

    display_cols = [
        "rank", "title", "category", "platform", "price", "rating",
        "review_count", "composite_score",
    ]
    available = [c for c in display_cols if c in df_topk.columns]

    if "cluster_kmeans" in df_topk.columns:
        available.insert(available.index("composite_score"), "cluster_kmeans")

    st.dataframe(
        df_topk[available].sort_values("rank"),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Telecharger Top-K (CSV)",
        data=df_topk.to_csv(index=False),
        file_name="top_k_products.csv",
        mime="text/csv",
    )


def render_viz_tab(df: pd.DataFrame, df_topk: pd.DataFrame):
    st.subheader("Visualisations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Prix vs Rating (Top-K)**")
        if "price" in df_topk.columns and "rating" in df_topk.columns:
            fig_scatter = px.scatter(
                df_topk.dropna(subset=["price", "rating"]),
                x="price",
                y="rating",
                color="category" if "category" in df_topk.columns else None,
                size="composite_score" if "composite_score" in df_topk.columns else None,
                hover_data=["title"],
                title="Prix vs Rating",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Colonnes prix/rating indisponibles")

    with col2:
        st.markdown("**Score composite par Cluster**")
        if "cluster_kmeans" in df.columns and "composite_score" in df.columns:
            fig_box = px.box(
                df.dropna(subset=["composite_score", "cluster_kmeans"]),
                x="cluster_kmeans",
                y="composite_score",
                color="cluster_kmeans",
                title="Distribution du score par cluster KMeans",
            )
            st.plotly_chart(fig_box, use_container_width=True)
        elif "composite_score" in df.columns:
            fig_hist = px.histogram(
                df, x="composite_score", title="Distribution du score composite"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Top categories (Top-K)**")
        if "category" in df_topk.columns:
            cat_counts = df_topk["category"].value_counts().head(15)
            fig_bar = px.bar(
                x=cat_counts.values,
                y=cat_counts.index,
                orientation="h",
                title="Categories les plus representees",
                labels={"x": "Nombre de produits", "y": "Categorie"},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with col4:
        st.markdown("**Repartition par plateforme**")
        if "platform" in df.columns:
            plat_counts = df["platform"].value_counts()
            fig_pie = px.pie(
                values=plat_counts.values,
                names=plat_counts.index,
                title="Produits par plateforme",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.markdown("**Carte geographique (estimation)**")

    df_map = df.copy()
    if "_source_file" in df_map.columns:
        df_map["country"] = df_map["_source_file"].apply(derive_country)
        country_stats = (
            df_map.groupby("country")
            .agg(produits=("product_id", "count"), score_moyen=("composite_score", "mean"))
            .reset_index()
        )
        try:
            fig_map = px.choropleth(
                country_stats,
                locations="country",
                locationmode="country names",
                color="produits",
                hover_name="country",
                hover_data={"score_moyen": ":.2f", "produits": True},
                title="Estimation geographique (basee source)",
                color_continuous_scale="Viridis",
            )
            fig_map.update_geography(showcountries=True, showcoastlines=True)
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception:
            fig_bar2 = px.bar(
                country_stats,
                x="country",
                y="produits",
                color="score_moyen",
                title="Produits par zone (estimation)",
                labels={"country": "Zone", "produits": "Produits"},
            )
            st.plotly_chart(fig_bar2, use_container_width=True)
    else:
        st.info("Donnees geographiques non disponibles")


def generate_llm_summary(df: pd.DataFrame, df_topk: pd.DataFrame) -> str:
    api_key = os.environ.get("GROQ_API_KEY")

    n_products = len(df)
    n_topk = len(df_topk)

    avg_price = df["price"].mean() if "price" in df.columns else 0
    avg_rating = df["rating"].mean() if "rating" in df.columns else 0
    top_cat = (
        df_topk["category"].value_counts().head(3).to_dict()
        if "category" in df_topk.columns
        else {}
    )

    summary = f"""
**Analyse du catalogue :**
- {n_products} produits analyses
- Score composite moyen : {df['composite_score'].mean():.3f}
- Prix moyen : {avg_price:.2f} {df['currency'].iloc[0] if 'currency' in df.columns else 'EUR'}
- Note moyenne : {avg_rating:.2f}/5
"""

    if top_cat:
        summary += "\n**Top categories (Top-K) :**\n"
        for cat, count in top_cat.items():
            summary += f"- {cat}: {count} produits\n"

    if "cluster_kmeans" in df.columns:
        cluster_dist = df["cluster_kmeans"].value_counts().sort_index()
        summary += "\n**Distribution des clusters :**\n"
        for cluster, count in cluster_dist.items():
            pct = count / n_products * 100
            summary += f"- Cluster {cluster}: {count} produits ({pct:.1f}%)\n"

    summary += "\n**Tendances identifiees :**\n"

    if not top_cat:
        summary += "- Donnees insuffisantes pour degager des tendances\n"
        return summary

    top_cat_name = list(top_cat.keys())[0]
    summary += f"- La categorie dominante est '{top_cat_name}' avec {list(top_cat.values())[0]} produits dans le Top-K\n"

    if avg_rating >= 4.0:
        summary += "- La note moyenne est excellente (>4.0), indiquant une satisfaction elevee\n"
    elif avg_rating >= 3.0:
        summary += "- La note moyenne est correcte (entre 3.0 et 4.0)\n"
    else:
        summary += "- La note moyenne est faible (<3.0), attention a la qualite\n"

    return summary


def build_category_context(df: pd.DataFrame, category: str) -> str:
    subset = df[df["category"] == category].dropna(subset=["price", "rating"]).copy()
    if subset.empty:
        return ""

    top = subset.nlargest(10, "composite_score")
    lines = [f"Categorie: {category}", f"Nombre de produits: {len(subset)}", ""]
    lines.append("Top 10 produits (par score composite):")
    for _, row in top.iterrows():
        lines.append(
            f"- {row.get('title', 'N/A')[:60]} | "
            f"Prix: {row.get('price', 'N/A')} | "
            f"Rating: {row.get('rating', 'N/A')}/5 | "
            f"Reviews: {row.get('review_count', 'N/A')} | "
            f"Score: {row.get('composite_score', 'N/A'):.3f} | "
            f"Cluster: {row.get('cluster_kmeans', 'N/A')}"
        )
    lines.append("")
    stats = subset["price"].describe()
    lines.append(f"Stats prix: min={stats['min']:.2f}, max={stats['max']:.2f}, "
                 f"moy={stats['mean']:.2f}, med={stats['50%']:.2f}")
    lines.append(f"Rating moyen: {subset['rating'].mean():.2f}/5")
    lines.append(f"Score composite moyen: {subset['composite_score'].mean():.3f}")
    if "cluster_kmeans" in subset.columns:
        cluster_dist = subset["cluster_kmeans"].value_counts().to_dict()
        lines.append(f"Distribution clusters: {cluster_dist}")
    return "\n".join(lines)


def call_groq_competitive_analysis(api_key: str, category_context: str) -> str:
    import httpx
    system_prompt = (
        "Tu es un analyste e-commerce specialise en analyse concurrentielle. "
        "Reponds en francais en suivant ce raisonnement pas-a-pas :\n"
        "1. Analyse les forces et faiblesses de chaque produit de la categorie.\n"
        "2. Compare leurs metriques (prix, rating, score).\n"
        "3. Classe les 3 meilleurs avec justification.\n"
        "4. Propose un positionnement strategique argumente.\n\n"
        "Structure la reponse ainsi :\n"
        "## Top 3 concurrents dans la categorie\n"
        "- Produit X: forces/faiblesses (justification)\n\n"
        "## Tableau comparatif\n"
        "| Produit | Prix | Rating | Score | Position |\n"
        "|---|---|---|---|---|\n\n"
        "## Recommandation strategique\n"
        "- Point 1 (car ...)\n- Point 2 (car ...)\n- Point 3 (car ...)"
    )
    user_prompt = (
        f"Voici les donnees de la categorie selectionnee:\n\n"
        f"{category_context}\n\n"
        "Analyse la concurrence : identifie les 3 meilleurs produits, "
        "compare leurs metriques, et propose un positionnement strategique."
    )
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"Erreur API: {resp.status_code}"


def build_marketing_context(df: pd.DataFrame, df_topk: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Nombre total de produits: {len(df)}")
    lines.append(f"Prix moyen: {df['price'].mean():.2f}" if "price" in df.columns else "")
    lines.append(f"Rating moyen: {df['rating'].mean():.2f}/5" if "rating" in df.columns else "")
    lines.append(f"Score composite moyen: {df['composite_score'].mean():.3f}" if "composite_score" in df.columns else "")
    lines.append("")

    if "cluster_kmeans" in df.columns:
        lines.append("Repartition des clusters:")
        for cluster, count in df["cluster_kmeans"].value_counts().sort_index().items():
            pct = count / len(df) * 100
            lines.append(f"- Cluster {cluster}: {count} produits ({pct:.1f}%)")
        lines.append("")

    if "category" in df_topk.columns:
        top_cats = df_topk["category"].value_counts().head(3)
        lines.append("Categories dominantes (Top-K):")
        for cat, count in top_cats.items():
            lines.append(f"- {cat}: {count} produits")
        lines.append("")

    if "composite_score" in df_topk.columns:
        top5 = df_topk.nlargest(5, "composite_score")
        lines.append("Top 5 produits (par score composite):")
        for _, row in top5.iterrows():
            lines.append(
                f"- {row.get('title', 'N/A')[:60]} | "
                f"Prix: {row.get('price', 'N/A')} | "
                f"Rating: {row.get('rating', 'N/A')}/5 | "
                f"Score: {row.get('composite_score', 'N/A'):.3f}"
            )
        lines.append("")

    return "\n".join(line for line in lines if line)


def call_groq_marketing_strategies(api_key: str, context: str) -> str:
    import httpx
    system_prompt = (
        "Tu es un consultant marketing e-commerce senior. "
        "Raisonne etape par etape avant de repondre :\n"
        "1. Analyse les clusters, scores et categories pour detecter des tendances.\n"
        "2. Deduis-en des opportunites marketing ciblees.\n"
        "3. Formule 3 strategies avec justification pour chacune.\n"
        "4. Propose des bundles produits coherents.\n\n"
        "Structure la reponse ainsi :\n"
        "## Analyse des tendances emergentes\n"
        "(analyse basee sur les clusters, scores et categories, avec justification)\n\n"
        "## 3 strategies marketing recommandees\n\n"
        "### Strategie 1: Promotions\n"
        "- Action: ...\n- Cible: ... (car ...)\n- Impact attendu: ...\n\n"
        "### Strategie 2: Cross-selling\n"
        "- Action: ...\n- Cible: ... (car ...)\n- Impact attendu: ...\n\n"
        "### Strategie 3: Campagnes\n"
        "- Action: ...\n- Cible: ... (car ...)\n- Impact attendu: ...\n\n"
        "## Suggestions de bundles produits\n"
        "- Bundle 1: ... (justification)\n- Bundle 2: ... (justification)\n- Bundle 3: ... (justification)"
    )
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Voici les donnees du catalogue:\n\n{context}\n\n"
                                         "Genere les strategies marketing."},
        ],
    }
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"Erreur API: {resp.status_code}"


def call_groq_product_trends(api_key: str, df_topk: pd.DataFrame) -> str:
    import httpx
    top10 = df_topk.head(10) if "rank" not in df_topk.columns else df_topk.sort_values("rank").head(10)
    lines = ["Top 10 produits (classement):"]
    for _, row in top10.iterrows():
        lines.append(
            f"- {row.get('title', 'N/A')[:60]} | "
            f"Categorie: {row.get('category', 'N/A')} | "
            f"Prix: {row.get('price', 'N/A')} | "
            f"Rating: {row.get('rating', 'N/A')}/5 | "
            f"Score: {row.get('composite_score', 'N/A'):.3f}"
        )
    context = "\n".join(lines)

    system_prompt = (
        "Tu es un analyste de tendances e-commerce. "
        "Raisonne pas-a-pas :\n"
        "1. Examine chaque produit du Top 10 : prix, note, categorie, score.\n"
        "2. Identifie les facteurs qui expliquent leur classement.\n"
        "3. Degage les tendances emergentes avec justification.\n"
        "4. Formule des recommandations argumentees.\n\n"
        "Structure la reponse ainsi :\n"
        "## Produits emergents\n"
        "- Produit X: pourquoi il se demarque (analyse)\n\n"
        "## Facteurs cles de succes\n"
        "- Facteur 1 (car ...)\n- Facteur 2 (car ...)\n\n"
        "## Recommandations pour la mise en avant\n"
        "- Recommandation 1 (justification)\n- Recommandation 2 (justification)"
    )
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Voici le Top 10 des produits:\n\n{context}\n\n"
                                         "Analyse les tendances et produits emergents."},
        ],
    }
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"Erreur API: {resp.status_code}"


def render_llm_tab(df: pd.DataFrame, df_topk: pd.DataFrame):
    st.subheader("Analyse LLM")

    api_key = os.environ.get("GROQ_API_KEY")

    with st.expander("Resume automatique des tendances", expanded=True):
        summary = generate_llm_summary(df, df_topk)
        st.markdown(summary)

    st.markdown("---")
    st.subheader("Synthese des meilleurs produits")

    top5 = df_topk.head(5) if "rank" not in df_topk.columns else df_topk.sort_values("rank").head(5)
    if not top5.empty:
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            st.markdown(
                f"**{i}. {row.get('title', 'N/A')}**  "
                f"(Note: {row.get('rating', 'N/A')}/5, "
                f"Prix: {row.get('price', 'N/A')}, "
                f"Score: {row.get('composite_score', 'N/A'):.3f})"
            )

    st.markdown("---")
    st.subheader("Strategies Marketing")

    api_key = os.environ.get("GROQ_API_KEY")

    col_trends, col_strat = st.columns(2)
    with col_trends:
        if st.button("Generer les tendances", use_container_width=True, key="btn_trends"):
            if not api_key:
                st.warning("GROQ_API_KEY requise.")
            else:
                with st.spinner("Analyse des tendances en cours..."):
                    result = call_groq_product_trends(api_key, df_topk)
                    st.markdown(result)

    with col_strat:
        if st.button("Strategies recommandees", use_container_width=True, key="btn_strat"):
            if not api_key:
                st.warning("GROQ_API_KEY requise.")
            else:
                with st.spinner("Generation des strategies marketing..."):
                    context = build_marketing_context(df, df_topk)
                    with st.expander("Donnees envoyees au LLM", expanded=False):
                        st.text(context)
                    result = call_groq_marketing_strategies(api_key, context)
                    st.markdown(result)

    st.markdown("---")
    st.subheader("Analyse concurrentielle")

    api_key = os.environ.get("GROQ_API_KEY")
    categories = sorted(df["category"].dropna().unique().tolist())
    selected_cat = st.selectbox("Choisir une categorie", categories, key="comp_cat")

    if st.button("Analyser les concurrents", type="primary"):
        if not api_key:
            st.warning("GROQ_API_KEY requise pour l'analyse concurrentielle.")
        elif not selected_cat:
            st.warning("Selectionnez une categorie.")
        else:
            with st.spinner(f"Analyse concurrentielle de '{selected_cat}' en cours..."):
                context = build_category_context(df, selected_cat)
                if not context:
                    st.warning("Pas assez de donnees pour cette categorie.")
                else:
                    with st.expander("Donnees envoyees au LLM", expanded=False):
                        st.text(context)
                    result = call_groq_competitive_analysis(api_key, context)
                    st.markdown(result)

    st.markdown("---")
    st.subheader("Chatbot simple")

    if not api_key:
        st.info(
            "Aucune cle API LLM detectee. "
            "Definissez GROQ_API_KEY dans les variables d'environnement "
            "pour activer le chatbot.\n\n"
            "Le resume ci-dessus utilise les donnees locales sans appel API."
        )
    else:
        st.success("Cle API Groq detectee")

    user_query = st.text_input("Posez une question sur les donnees:", placeholder="Ex: quel est le meilleur produit par categorie ?")

    if user_query:
        if api_key:
            try:
                import httpx
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Tu es un assistant expert en e-commerce. Raisonne etape par etape avant de repondre : analyse la question, identifie les donnees pertinentes, formule ta reponse argumentee en francais.",
                        },
                        {"role": "user", "content": user_query},
                    ],
                }
                resp = httpx.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    answer = resp.json()["choices"][0]["message"]["content"]
                    st.markdown(f"**Reponse:**\n{answer}")
                else:
                    st.error(f"Erreur API: {resp.status_code}")
            except Exception as e:
                st.error(f"Erreur LLM: {e}")
        else:
            st.info("Cle API requise pour le chatbot.")


def render_export_tab(df: pd.DataFrame, df_topk: pd.DataFrame):
    st.subheader("Export des donnees")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Export complet (filtre actif)**")
        csv_full = df.to_csv(index=False)
        st.download_button(
            "Telecharger CSV (filtre actif)",
            data=csv_full,
            file_name="products_filtered.csv",
            mime="text/csv",
        )

    with col2:
        st.markdown("**Export Top-K**")
        csv_topk = df_topk.to_csv(index=False)
        st.download_button(
            "Telecharger Top-K CSV",
            data=csv_topk,
            file_name="top_k_products.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("**Apercu des donnees exportees**")
    st.dataframe(df.head(50), use_container_width=True)


def render_mcp_tab(client: MCPClient):
    st.subheader("Agent MCP — Outils et Permissions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Outils disponibles**")
        tools = client.list_tools()
        if tools:
            for t in tools:
                with st.expander(f"`{t['name']}`"):
                    st.markdown(f"_{t['description']}_")
                    if t.get("input_schema"):
                        st.code(json.dumps(t["input_schema"], indent=2), language="json")
        else:
            st.info("Aucun outil disponible pour ce rôle")

        st.markdown("---")
        st.markdown("**Exécuter un outil**")
        tool_names = [t["name"] for t in tools]
        if tool_names:
            selected_tool = st.selectbox("Choisir un outil", tool_names, key="mcp_tool_select")
            params_str = st.text_area(
                "Paramètres (JSON)",
                value="{}",
                height=80,
                key="mcp_params",
            )
            if st.button("Exécuter", type="primary", key="mcp_exec"):
                try:
                    params = json.loads(params_str)
                except json.JSONDecodeError:
                    st.error("JSON invalide")
                    params = {}
                with st.spinner(f"Exécution de '{selected_tool}'..."):
                    result = client.call_tool(selected_tool, params)
                if "error" in result:
                    st.error(f"Erreur: {result['error']}")
                else:
                    st.success("Exécution réussie")
                    st.json(result)
        else:
            st.info("Connectez l'agent et choisissez un rôle avec les outils appropriés")

    with col2:
        st.markdown("**Journal des requêtes**")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Rafraîchir", key="refresh_logs"):
                st.rerun()
        with col_b:
            if client.current_role == "admin":
                if st.button("Effacer les logs", type="primary", key="clear_logs"):
                    if client.clear_logs():
                        st.success("Logs effacés")
                        st.rerun()
                    else:
                        st.error("Impossible d'effacer les logs")
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agents", "logs")
        today = str(__import__("datetime").date.today())
        log_path = os.path.join(logs_dir, f"mcp_{today}.jsonl")
        entries = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entries.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            except OSError:
                pass
        entries = entries[-50:]
        if entries:
            for entry in reversed(entries):
                ts = entry.get("_timestamp", "")[11:19]
                event = entry.get("event", "")
                tool = entry.get("tool", "")
                user = entry.get("user", "")
                status = entry.get("status", "")
                label = f"{ts} | {event:8s} | {tool:20s} | {user:10s} | {status}"
                if event == "denied":
                    st.warning(label)
                elif event == "error":
                    st.error(label)
                else:
                    st.info(label)
        else:
            st.info("Aucune requête journalisée")

    st.markdown("---")
    st.markdown("**Roles et permissions**")
    roles_data = client.get_roles()
    for role, data in roles_data.items():
        tools_list = ", ".join(f"`{t}`" for t in data.get("tools", []))
        st.markdown(f"- **{role}**: {data.get('description', '')} — {tools_list}")


def main():
    st.title("Smart E-Commerce Intelligence")
    st.markdown("Dashboard BI interactif - Analyse des produits")

    df_clusters, df_topk = load_data()
    st.sidebar.success(f"{len(df_clusters)} produits charges")

    cat, clusters, platform, price_range, score_range = sidebar_filters(df_clusters)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Agent MCP")
    mcp_client: MCPClient = st.session_state.mcp_client

    roles = list(mcp_client.get_roles().keys())
    current_role = mcp_client.current_role
    idx = roles.index(current_role) if current_role in roles else 0
    selected_role = st.sidebar.selectbox("Rôle", roles, index=idx, key="mcp_role")
    if selected_role != current_role:
        mcp_client.set_role(selected_role)
        st.rerun()

    mcp_connected = st.sidebar.checkbox("Connecté", value=mcp_client.connected, key="mcp_connected")
    if mcp_connected and not mcp_client.connected:
        mcp_client.connect(role=selected_role)
        st.rerun()
    elif not mcp_connected and mcp_client.connected:
        mcp_client.disconnect()
        st.rerun()

    df_filtered = apply_filters(df_clusters, cat, clusters, platform, price_range, score_range)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Top-K Produits", "Visualisations", "Analyse LLM", "Export", "Agent MCP"]
    )

    with tab1:
        render_topk_tab(df_topk)

    with tab2:
        render_viz_tab(df_filtered, df_topk)

    with tab3:
        render_llm_tab(df_filtered, df_topk)

    with tab4:
        render_export_tab(df_filtered, df_topk)

    with tab5:
        render_mcp_tab(mcp_client)


if __name__ == "__main__":
    main()
