import os
import sys
import json
import datetime
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np

from TopKselection.preprocessing import load_data, clean_missing_values, normalize_features, create_features
from TopKselection.scoring import ProductScorer, get_top_k_products, save_top_k, analyze_score_distribution
from TopKselection.supervised import SupervisedML
from TopKselection.clustering import run_clustering_pipeline
from TopKselection.association_rules import run_association_pipeline

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DATA_PROCESSED = "data/processed"


class MainPipeline:
    def __init__(
        self,
        data_raw_path: str = "data/raw/",
        data_processed_path: str = "data/processed/",
    ):
        self.data_raw_path = data_raw_path
        self.data_processed_path = data_processed_path
        self.results = {}

    def run_preprocessing(self) -> pd.DataFrame:
        print("\n" + "=" * 50)
        print("ETAPE 1: Preprocessing")
        print("=" * 50)
        try:
            df = load_data(os.path.join(self.data_raw_path, "*.csv"))
            df = clean_missing_values(df)
            df = normalize_features(df)
            df = create_features(df)
            out_path = os.path.join(self.data_processed_path, "cleaned_products.csv")
            os.makedirs(self.data_processed_path, exist_ok=True)
            df.to_csv(out_path, index=False, encoding="utf-8")
            print(f"[pipeline] preprocessing termine -> {out_path}")
            self.results["preprocessing"] = {"shape": df.shape, "path": out_path}
            return df
        except Exception as e:
            print(f"[pipeline] ERREUR preprocessing: {e}")
            raise

    def run_scoring(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        print("\n" + "=" * 50)
        print("ETAPE 2: Scoring")
        print("=" * 50)
        try:
            scorer = ProductScorer()
            df = scorer.calculate_composite_score(df)
            score_stats = analyze_score_distribution(df)
            df_top = get_top_k_products(df, k=100)
            save_top_k(df_top)
            out_full_path = os.path.join(self.data_processed_path, "products_with_score.csv")
            df.to_csv(out_full_path, index=False, encoding="utf-8")
            print(f"[pipeline] scoring termine -> {out_full_path}")
            self.results["scoring"] = {
                "score_range": {"min": score_stats["min"], "max": score_stats["max"]},
                "top_k_path": os.path.join(OUTPUT_DIR, "top_k_products.csv"),
                "full_path": out_full_path,
            }
            return df, df_top
        except Exception as e:
            print(f"[pipeline] ERREUR scoring: {e}")
            raise

    def run_supervised(self, df: pd.DataFrame) -> dict:
        print("\n" + "=" * 50)
        print("ETAPE 3: Supervised ML")
        print("=" * 50)
        try:
            ml = SupervisedML()
            results = ml.run_supervised_pipeline(df)
            self.results["supervised"] = {
                "rf_f1": results["rf"]["f1"],
                "rf_accuracy": results["rf"]["accuracy"],
                "xgb_f1": results.get("xgb", {}).get("f1"),
                "xgb_accuracy": results.get("xgb", {}).get("accuracy"),
                "feature_cols": results["feature_cols"],
            }
            print("[pipeline] supervised termine")
            return results
        except Exception as e:
            print(f"[pipeline] ERREUR supervised: {e}")
            raise

    def run_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n" + "=" * 50)
        print("ETAPE 4: Clustering")
        print("=" * 50)
        try:
            cluster_results = run_clustering_pipeline(df)
            out_path = os.path.join(self.data_processed_path, "products_with_clusters.csv")
            df.to_csv(out_path, index=False, encoding="utf-8")
            print(f"[pipeline] clustering termine -> {out_path}")
            self.results["clustering"] = {
                "silhouette_scores": {
                    name: score for name, score in cluster_results["analyzer"].metrics.items()
                },
                "interpretations": cluster_results["interpretations"],
                "path": out_path,
            }
            return df
        except Exception as e:
            print(f"[pipeline] ERREUR clustering: {e}")
            raise

    def run_association(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n" + "=" * 50)
        print("ETAPE 5: Regles d'association")
        print("=" * 50)
        try:
            rules = run_association_pipeline(df)
            n_rules = len(rules)
            print(f"[pipeline] association terminee ({n_rules} regles)")
            self.results["association"] = {
                "n_rules": n_rules,
                "path": os.path.join(OUTPUT_DIR, "association_rules.csv"),
            }
            return rules
        except Exception as e:
            print(f"[pipeline] ERREUR association: {e}")
            raise

    def run_full_pipeline(
        self,
        skip_preprocessing: bool = False,
    ) -> dict:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLET TopKselection")
        print("=" * 60)

        self.results["start_time"] = datetime.datetime.now().isoformat()

        if skip_preprocessing:
            path = os.path.join(self.data_processed_path, "cleaned_products.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"[pipeline] skip preprocessing, charge {path} ({len(df)} lignes)")
            else:
                print(f"[pipeline] {path} introuvable, execution preprocessing")
                df = self.run_preprocessing()
        else:
            df = self.run_preprocessing()

        scoring_path = os.path.join(self.data_processed_path, "products_with_score.csv")
        if skip_preprocessing and os.path.exists(scoring_path):
            df = pd.read_csv(scoring_path)
            print(f"[pipeline] skip scoring, charge {scoring_path} ({len(df)} lignes)")
        else:
            df, df_top = self.run_scoring(df)

        self.run_supervised(df)

        clusters_path = os.path.join(self.data_processed_path, "products_with_clusters.csv")
        if skip_preprocessing and os.path.exists(clusters_path):
            df = pd.read_csv(clusters_path)
            print(f"[pipeline] skip clustering, charge {clusters_path} ({len(df)} lignes)")
        else:
            df = self.run_clustering(df)

        self.run_association(df)

        self.results["end_time"] = datetime.datetime.now().isoformat()
        self.results["status"] = "completed"

        print("\n" + "=" * 60)
        print("PIPELINE TERMINE AVEC SUCCES")
        print("=" * 60)

        return self.results


def save_pipeline_report(results_dict: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append("RAPPORT DU PIPELINE TopKselection")
    lines.append("=" * 60)
    lines.append(f"Demarrage : {results_dict.get('start_time', 'N/A')}")
    lines.append(f"Fin       : {results_dict.get('end_time', 'N/A')}")
    lines.append(f"Statut    : {results_dict.get('status', 'N/A')}")
    lines.append("")

    if "preprocessing" in results_dict:
        pp = results_dict["preprocessing"]
        lines.append("-" * 40)
        lines.append("1. Preprocessing")
        lines.append("-" * 40)
        lines.append(f"   Shape finale: {pp.get('shape')}")
        lines.append("")

    if "scoring" in results_dict:
        sc = results_dict["scoring"]
        lines.append("-" * 40)
        lines.append("2. Scoring")
        lines.append("-" * 40)
        score_range = sc.get("score_range", {})
        lines.append(f"   Score range : [{score_range.get('min', 'N/A')}, {score_range.get('max', 'N/A')}]")
        lines.append("")

    if "supervised" in results_dict:
        sv = results_dict["supervised"]
        lines.append("-" * 40)
        lines.append("3. Supervised ML")
        lines.append("-" * 40)
        lines.append(f"   Random Forest F1      : {sv.get('rf_f1', 'N/A')}")
        lines.append(f"   Random Forest Accuracy : {sv.get('rf_accuracy', 'N/A')}")
        if sv.get("xgb_f1") is not None:
            lines.append(f"   XGBoost F1             : {sv['xgb_f1']}")
            lines.append(f"   XGBoost Accuracy       : {sv.get('xgb_accuracy', 'N/A')}")
        else:
            lines.append("   XGBoost : non installe")
        lines.append(f"   Features              : {sv.get('feature_cols', [])}")
        lines.append("")

    if "clustering" in results_dict:
        cl = results_dict["clustering"]
        lines.append("-" * 40)
        lines.append("4. Clustering")
        lines.append("-" * 40)
        for method, score in cl.get("silhouette_scores", {}).items():
            if score and score > 0:
                lines.append(f"   {method:15s}: {score:.4f}")
            else:
                lines.append(f"   {method:15s}: N/A")
        lines.append("   Interpretations:")
        for cluster_id, label in cl.get("interpretations", {}).items():
            lines.append(f"      cluster {cluster_id} : {label}")
        lines.append("")

    if "association" in results_dict:
        ar = results_dict["association"]
        lines.append("-" * 40)
        lines.append("5. Regles d'association")
        lines.append("-" * 40)
        lines.append(f"   Regles generees: {ar.get('n_rules', 0)}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("FIN DU RAPPORT")
    lines.append("=" * 60)

    report = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[rapport] sauvegarde -> {output_path}")


def print_summary(results_dict: dict):
    print("\n" + "=" * 50)
    print("RESUME DU PIPELINE")
    print("=" * 50)

    if "supervised" in results_dict:
        sv = results_dict["supervised"]
        print(f"  RF  F1: {sv.get('rf_f1', 'N/A')}")
        if sv.get("xgb_f1") is not None:
            print(f"  XGB F1: {sv['xgb_f1']}")

    if "clustering" in results_dict:
        cl = results_dict["clustering"]
        print(f"  Clustering silhouettes:")
        for method, score in cl.get("silhouette_scores", {}).items():
            if score and score > 0:
                print(f"    {method}: {score:.4f}")

    if "association" in results_dict:
        ar = results_dict["association"]
        print(f"  Regles d'association: {ar.get('n_rules', 0)}")

    print("=" * 50)


if __name__ == "__main__":
    pipeline = MainPipeline(
        data_raw_path="data/raw/",
        data_processed_path="data/processed/",
    )

    try:
        results = pipeline.run_full_pipeline(skip_preprocessing=False)
    except Exception as e:
        print(f"\n[pipeline] ERREUR: {e}")
        results = pipeline.results
        results["status"] = "failed"

    report_path = os.path.join(OUTPUT_DIR, "pipeline_report.txt")
    save_pipeline_report(results, report_path)
    print_summary(results)
