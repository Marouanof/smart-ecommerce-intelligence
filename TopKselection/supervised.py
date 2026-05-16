import os
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

try:
    import joblib
except ImportError:
    joblib = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

warnings.filterwarnings("ignore", category=UserWarning)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

FEATURE_BASES = [
    "engagement",
    "prix_par_rating",
    "log_reviews",
    "disponible",
]

OPTIONAL_FEATURES = [
    "discount_pct",
    "delivery_days",
    "variant_count",
]


class SupervisedML:
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.rf_model: Optional[RandomForestClassifier] = None
        self.xgb_model: Optional["XGBClassifier"] = None
        self.feature_cols: list[str] = []
        self.results: dict = {}

    def _find_score_column(self, df: pd.DataFrame) -> Optional[str]:
        for candidate in ["composite_score", "score", "rating_norm"]:
            if candidate in df.columns:
                return candidate
        return None

    def create_target_variable(self, df: pd.DataFrame, top_percentile: int = 20) -> pd.DataFrame:
        score_col = self._find_score_column(df)
        print(f"[DEBUG] Colonne score utilisée: {score_col}")
        print(f"[DEBUG] Min score: {df[score_col].min()}")
        print(f"[DEBUG] Max score: {df[score_col].max()}")
        print(f"[DEBUG] Moyenne: {df[score_col].mean()}")
        print(f"[DEBUG] Percentiles: {df[score_col].quantile([0.5, 0.8, 0.9, 0.95, 0.99]).to_dict()}")

        threshold = df[score_col].quantile(1 - top_percentile / 100.0)
        print(f"[DEBUG] Seuil calculé (top {top_percentile}%): {threshold}")

        df["is_top_product"] = (df[score_col] >= threshold).astype(int)
        print(f"[target] {df['is_top_product'].mean() * 100:.1f}% positifs")
        return df

    @staticmethod
    def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        available = []
        for col in FEATURE_BASES:
            if col in df.columns:
                available.append(col)
        for col in OPTIONAL_FEATURES:
            if col in df.columns:
                available.append(col)

        missing = [c for c in FEATURE_BASES if c not in df.columns]
        if missing:
            print(f"[features] colonnes dérivées non trouvées: {missing}")
        found_opt = [c for c in OPTIONAL_FEATURES if c in df.columns]
        if found_opt:
            print(f"[features] colonnes optionnelles trouvées: {found_opt}")

        X = df[available].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        print(f"[features] {len(available)} colonnes sélectionnées: {available}")
        return X, available

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        if self.cv_folds > 1:
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring="f1")
            print(f"[RF] CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        self.rf_model = model
        return model

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> Optional["XGBClassifier"]:
        if XGBClassifier is None:
            print("[XGBoost] non installé, skip")
            return None
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = XGBClassifier(
            n_estimators=100,
            random_state=self.random_state,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        if self.cv_folds > 1:
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring="f1")
            print(f"[XGB] CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        self.xgb_model = model
        return model

    @staticmethod
    def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model") -> dict:
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{'='*50}")
        print(f"Évaluation {model_name}")
        print(f"{'='*50}")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-score : {metrics['f1']:.4f}")
        print(f"\nMatrice de confusion:")
        print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
        print(f"{'='*50}\n")

        metrics["confusion_matrix"] = cm.tolist()
        metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
        return metrics

    def save_models(self) -> dict[str, str]:
        os.makedirs(MODELS_DIR, exist_ok=True)
        paths = {}
        if joblib is None:
            print("[save] joblib non installé, modèles non sauvegardés")
            return paths
        if self.rf_model:
            p = os.path.join(MODELS_DIR, "random_forest.pkl")
            joblib.dump(self.rf_model, p)
            paths["rf"] = p
            print(f"[save] RF -> {p}")
        if self.xgb_model:
            p = os.path.join(MODELS_DIR, "xgboost.pkl")
            joblib.dump(self.xgb_model, p)
            paths["xgb"] = p
            print(f"[save] XGB -> {p}")
        return paths

    def run_supervised_pipeline(
        self,
        df: pd.DataFrame,
        test_size: float = 0.3,
        top_percentile: int = 20,
        save: bool = True,
    ) -> dict:
        print(f"\n{'#'*60}")
        print("# Pipeline Supervisé")
        print(f"{'#'*60}\n")

        # === CONVERSION DES COLONNES _norm EN NUMÉRIQUE ===
        norm_cols = [col for col in df.columns if col.endswith('_norm')]
        for col in norm_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"[DEBUG] Conversion {col}: {df[col].dtype} -> {df[col].dtype}")

        df = self.create_target_variable(df, top_percentile=top_percentile)
        X, self.feature_cols = self.prepare_features(df)
        y = df["is_top_product"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y,
        )
        print(f"[split] Train: {len(X_train)} | Test: {len(X_test)}")

        print("\n--- Random Forest ---")
        rf = self.train_random_forest(X_train, y_train)
        rf_metrics = self.evaluate_model(rf, X_test, y_test, "RandomForest")

        print("\n--- XGBoost ---")
        xgb = self.train_xgboost(X_train, y_train)
        xgb_metrics = None
        if xgb is not None:
            xgb_metrics = self.evaluate_model(xgb, X_test, y_test, "XGBoost")

        self.results = {
            "rf": rf_metrics,
            "xgb": xgb_metrics,
            "feature_cols": self.feature_cols,
            "test_size": test_size,
            "top_percentile": top_percentile,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        self._print_comparison(rf_metrics, xgb_metrics)

        if save:
            self.save_models()

        return self.results

    def _print_comparison(self, rf_metrics: dict, xgb_metrics: Optional[dict]):
        print(f"\n{'='*50}")
        print("Comparaison RF vs XGBoost")
        print(f"{'='*50}")
        print(f"{'Métrique':<12} {'RF':<10} {'XGBoost':<10}")
        print(f"{'-'*34}")
        print(f"{'Accuracy':<12} {rf_metrics['accuracy']:<10.4f} {xgb_metrics['accuracy'] if xgb_metrics else 'N/A':<10}")
        print(f"{'Precision':<12} {rf_metrics['precision']:<10.4f} {xgb_metrics['precision'] if xgb_metrics else 'N/A':<10}")
        print(f"{'Recall':<12} {rf_metrics['recall']:<10.4f} {xgb_metrics['recall'] if xgb_metrics else 'N/A':<10}")
        print(f"{'F1-score':<12} {rf_metrics['f1']:<10.4f} {xgb_metrics['f1'] if xgb_metrics else 'N/A':<10}")
        print(f"{'='*50}\n")


def run_supervised_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.3,
    top_percentile: int = 20,
    cv_folds: int = 5,
) -> dict:
    ml = SupervisedML(cv_folds=cv_folds)
    return ml.run_supervised_pipeline(df, test_size=test_size, top_percentile=top_percentile)


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/products_with_score.csv"
    if not os.path.exists(input_path):
        print(f"Fichier introuvable: {input_path}")
        print("Exécute d'abord: python -m TopKselection.preprocessing puis scoring")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Données chargées: {len(df)} lignes")
    run_supervised_pipeline(df)
