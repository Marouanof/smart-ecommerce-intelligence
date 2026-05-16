# Smart E-Commerce Intelligence

**Pipeline ML et Dashboard BI pour la sélection intelligente de produits e-commerce**

Projet dans le cadre du module **Data Mining** — Cycle d'ingenieur LSI.

---

## Table des matières

- [Contexte et objectif](#contexte-et-objectif)
- [Architecture du projet](#architecture-du-projet)
- [Pipeline Data Mining](#pipeline-data-mining)
- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Technologies utilisées](#technologies-utilisées)
- [Auteurs](#auteurs)

---

## Contexte et objectif

Ce projet applique les techniques de **Data Mining** à la sélection de produits e-commerce. L'objectif est de construire un pipeline automatisé capable de :

- Collecter des données produits depuis différentes sources (Shopify, WooCommerce)
- Nettoyer, normaliser et enrichir les données
- Calculer un **score composite** pour classer les produits
- Appliquer des algorithmes de **clustering** pour segmenter le catalogue
- Entraîner des modèles **supervisés** pour prédire les produits à fort potentiel
- Générer des **règles d'association** pour le cross-selling
- Visualiser les résultats via un **dashboard interactif**

---

## Architecture du projet

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Scrapers    │────>│  Pipeline ML │────>│  Modèles     │────>│  Dashboard   │
│  (collecte)  │     │  (5 étapes)  │     │  .pkl + CSV  │     │  Streamlit   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                            │
                            v
                    ┌──────────────┐
                    │  MCP Server  │
                    │  (Agent IA)  │
                    └──────────────┘
```

Le projet suit une architecture **modulaire** où chaque étape est indépendante et peut être exécutée séparément, localement ou dans un environnement conteneurisé (Docker, Kubeflow).

---

## Pipeline Data Mining

Le pipeline ML se décompose en **5 étapes** conformes à la méthodologie CRISP-DM :

### 1. Prétraitement (`TopKselection/preprocessing.py`)
- Chargement et déduplication des données brutes
- Imputation des valeurs manquantes (médiane, simulation réaliste)
- Filtrage des anomalies (prix, notes)
- Normalisation MinMax des features numériques
- Feature engineering : `prix_par_rating`, `engagement`, `disponible`, `log_reviews`

### 2. Scoring (`TopKselection/scoring.py`)
- Calcul d'un **score composite** pondéré :
  - Prix (−0.2), Rating (0.4), Review count (0.3), Stock (0.1)
- Sélection automatique du **Top-K** (100 meilleurs produits)
- Analyse de la distribution des scores

### 3. Apprentissage supervisé (`TopKselection/supervised.py`)
- Variable cible binaire : `is_top_product` (top 20%)
- **Random Forest** (100 arbres, class_weight='balanced')
- **XGBoost** (scale_pos_weight adapté)
- Évaluation : Accuracy, Precision, Recall, F1, Matrice de confusion
- Sauvegarde des modèles au format `.pkl`

### 4. Clustering (`TopKselection/clustering.py`)
- **KMeans** (k=3,4,5 — meilleur silhouette score)
- **DBSCAN** (eps=0.3, 0.5, 0.7)
- **Classification hiérarchique** (agglomérative, k=3)
- Visualisation PCA 2D
- Interprétation automatique des clusters (premium, discount, top-rated, etc.)

### 5. Règles d'association (`TopKselection/association_rules.py`)
- Algorithme **Apriori** (support ≥ 0.01, confiance ≥ 0.5, lift ≥ 1.0)
- Analyse cross-canal (catégorie + marque + cluster)
- Interprétation de la force des règles (lift > 3 = fort)

---

## Fonctionnalités

### Dashboard BI (Streamlit)
- **Top-K Produits** : tableau interactif des 100 meilleurs produits, export CSV
- **Visualisations** : scatter plot, box plot, bar chart, pie chart, carte choroplèthe
- **Analyse LLM** : génération de résumés, tendances, stratégies marketing et analyse concurrentielle via **Groq API** (Llama 3.3 70B)
- **Chatbot e-commerce** : question-réponse sur les données
- **Filtres avancés** : catégorie, cluster, plateforme, prix, score

### Agent MCP (Model Context Protocol)
- Serveur d'outils avec **contrôle d'accès par rôles** (viewer, analyst, admin)
- 8 outils disponibles : recherche produits, top-k, clusters, statistiques, export, pipeline
- Journalisation complète des requêtes
- Isolation des accès par niveau de permission

### MLOps
- **Docker Compose** : orchestration locale des 6 services
- **Kubeflow Pipelines** : définition KFP v2 pour déploiement Kubernetes
- **GitHub Actions** : CI/CD avec compilation et validation du pipeline

---

## Prérequis

- Python 3.10+
- Virtualenv (recommandé)

---

## Installation

```powershell
# Cloner le projet
git clone <url-du-projet>
cd smart-ecommerce-intelligence

# Créer et activer l'environnement virtuel
python -m venv venv
.\venv\Scripts\Activate.ps1

# Installer les dépendances
pip install pandas numpy scikit-learn xgboost joblib mlxtend
pip install streamlit plotly matplotlib seaborn httpx python-dotenv
pip install requests beautifulsoup4 playwright
```

### Variables d'environnement

Créer un fichier `.env` à la racine :

```ini
GROQ_API_KEY=votre_cle_groq
HUGGINGFACE_TOKEN=votre_token_huggingface
```

---

## Utilisation

### Générer les données brutes (optionnel)

```powershell
# Scraper des boutiques Shopify
python -m scrapers.run_shopify

# Générer des données WooCommerce de démonstration
python -m scrapers.generate_demo
```

Les fichiers CSV produits sont stockés dans `data/raw/`.

### Exécuter le pipeline ML complet

```powershell
python -m TopKselection.pipeline
```

Étapes exécutées automatiquement :
1. Preprocessing → `data/processed/cleaned_products.csv`
2. Scoring → `data/processed/products_with_score.csv` + `TopKselection/output/top_k_products.csv`
3. Supervised → `TopKselection/models/random_forest.pkl`, `xgboost.pkl`
4. Clustering → `data/processed/products_with_clusters.csv`
5. Association → `TopKselection/output/association_rules.csv`

Un rapport de synthèse est généré dans `TopKselection/output/pipeline_report.txt`.

### Lancer la dashboard

```powershell
streamlit run dashboard/app.py
```

Accès : [http://localhost:8501](http://localhost:8501)

### Orchestration Docker

```powershell
docker-compose -f docker/docker-compose.yml up --build
```

### Orchestration Kubeflow

```powershell
pip install kfp
python kfp_pipeline.py
```

Le fichier `kfp_pipeline.yaml` est généré et peut être uploadé sur un cluster Kubeflow.

---

## Structure du projet

```
smart-ecommerce-intelligence/
│
├── TopKselection/               # Pipeline ML (5 étapes)
│   ├── preprocessing.py         # Étape 1 : Prétraitement
│   ├── scoring.py               # Étape 2 : Scoring + Top-K
│   ├── supervised.py            # Étape 3 : Random Forest + XGBoost
│   ├── clustering.py            # Étape 4 : KMeans, DBSCAN, Hiérarchique
│   ├── association_rules.py     # Étape 5 : Apriori
│   ├── pipeline.py              # Orchestrateur local
│   ├── models/                  # Modèles entraînés (.pkl)
│   └── output/                  # Rapports et exports
│
├── scrapers/                    # Collecte de données
│   ├── base.py                  # Classe Product, BaseScraper
│   ├── shopify.py               # Scraper Shopify
│   ├── woocommerce.py           # Scraper WooCommerce
│   ├── orchestrator.py          # Détection plateforme + orchestration
│   ├── run_shopify.py           # Lanceur scraping Shopify
│   └── generate_demo.py         # Générateur données démo
│
├── dashboard/
│   └── app.py                   # Dashboard Streamlit (BI + LLM + MCP)
│
├── agents/                      # MCP (Model Context Protocol)
│   ├── mcp_server.py            # Serveur d'outils
│   ├── mcp_client.py            # Client MCP
│   └── permissions.json         # Règles d'accès par rôle
│
├── llm/
│   └── enrichment.py            # Enrichissement LLM (Hugging Face)
│
├── data/
│   ├── raw/                     # Données brutes (CSV)
│   └── processed/               # Données transformées (CSV)
│
├── docker/                      # Conteneurisation
│   ├── docker-compose.yml       # Orchestration 6 services
│   ├── preprocessing/Dockerfile
│   ├── scoring/Dockerfile
│   ├── supervised/Dockerfile
│   ├── clustering/Dockerfile
│   ├── association/Dockerfile
│   └── dashboard/Dockerfile
│
├── .github/workflows/           # CI/CD GitHub Actions
│   ├── docker-build.yml         # Build des images Docker
│   └── kubeflow-pipeline.yml    # Compilation et validation KFP
│
├── kfp_pipeline.py              # Définition Kubeflow Pipeline
├── kfp_pipeline.yaml            # Pipeline compilé
├── requirements.txt             # Dépendances scraping
├── .env                         # Variables d'environnement
└── README.md
```

---

## Technologies utilisées

| Domaine | Technologies |
|---|---|
| **Langage** | Python 3.10 |
| **Data Mining** | scikit-learn, XGBoost, mlxtend (Apriori) |
| **Visualisation** | Streamlit, Plotly, Matplotlib, Seaborn |
| **LLM** | Groq API (Llama 3.3 70B), Hugging Face Inference API |
| **Conteneurisation** | Docker, Docker Compose |
| **Orchestration ML** | Kubeflow Pipelines (KFP v2) |
| **CI/CD** | GitHub Actions |
| **MCP** | Architecture serveur/client avec contrôle d'accès |
| **Scraping** | BeautifulSoup, Playwright, HTTPx |

---

## Auteurs

Projet réalisé dans le cadre du module **Data Mining** — Cycle d'ingenieur LSI des Données.

---

*Dernière mise à jour : Mai 2026*
