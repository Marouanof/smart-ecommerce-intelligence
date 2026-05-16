import os
import numpy as np
import pandas as pd
from typing import Optional

try:
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError:
    apriori = None
    association_rules = None


class AssociationRulesAnalyzer:
    def __init__(self, min_support=0.01, min_confidence=0.5, min_lift=1.0):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift

    def prepare_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        cat = result.get("category", pd.Series("unknown", index=result.index))
        brand = result.get("brand", pd.Series("unknown", index=result.index))
        cat = cat.fillna("unknown").astype(str).str.strip()
        brand = brand.fillna("unknown").astype(str).str.strip()
        result["basket"] = cat + "_" + brand

        result["transaction_id"] = result.index.astype(str)

        for col in ["cluster_kmeans", "cluster_hierarchical", "cluster_dbscan"]:
            if col in result.columns:
                result[col] = result[col].fillna(-1).astype(int)
                result[col] = col + "_" + result[col].astype(str)

        print(f"[prepare] {result['basket'].nunique()} paniers uniques, "
              f"{len(result)} transactions (une par produit)")
        return result

    def generate_association_rules(self, df_transactions: pd.DataFrame) -> pd.DataFrame:
        if apriori is None or association_rules is None:
            raise ImportError("mlxtend est requis pour les regles d'association")

        item_cols = ["basket"]
        for col in ["cluster_kmeans", "cluster_hierarchical", "cluster_dbscan"]:
            if col in df_transactions.columns:
                item_cols.append(col)

        melted = df_transactions[["transaction_id"] + item_cols].melt(
            id_vars="transaction_id", value_vars=item_cols, value_name="item"
        )
        melted = melted.dropna(subset=["item"])
        melted = melted.drop_duplicates(subset=["transaction_id", "item"])

        one_hot = (
            melted
            .pivot_table(index="transaction_id", columns="item", aggfunc="size", fill_value=0)
        )
        one_hot = one_hot.astype(bool)
        n_transactions = one_hot.shape[0]

        effective_support = max(self.min_support, 5.0 / max(n_transactions, 1))
        if effective_support > self.min_support:
            print(f"[apriori] min_support ajuste a {effective_support:.4f} "
                  f"({n_transactions} transactions)")

        frequent = apriori(one_hot, min_support=effective_support, use_colnames=True, max_len=2)
        print(f"[apriori] {len(frequent)} itemsets frequents (support>={effective_support:.4f})")

        if len(frequent) < 2:
            print("[apriori] pas assez d'itemsets pour generer des regles")
            return pd.DataFrame()

        rules = association_rules(frequent, metric="confidence", min_threshold=self.min_confidence)
        rules = rules[rules["lift"] >= self.min_lift].copy()
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

        for col in ["antecedents", "consequents"]:
            rules[col] = rules[col].apply(lambda x: ", ".join(sorted(x)))

        print(f"[rules] {len(rules)} regles generees (confidence>={self.min_confidence}, lift>={self.min_lift})")
        return rules

    def get_top_rules(
        self,
        df_rules: pd.DataFrame,
        metric: str = "lift",
        n: int = 10,
    ) -> pd.DataFrame:
        if df_rules.empty:
            return df_rules
        if metric not in df_rules.columns:
            print(f"[top_rules] metrique '{metric}' introuvable, utilisation de 'lift'")
            metric = "lift"
        top = df_rules.sort_values(metric, ascending=False).head(n).reset_index(drop=True)
        print(f"[top_rules] {len(top)} meilleures regles (par {metric})")
        return top

    def interpret_rules(self, df_rules: pd.DataFrame):
        if df_rules.empty:
            print("[interpret] aucune regle a interpreter")
            return

        print("\n" + "=" * 60)
        print("Interpretation business des regles d'association")
        print("=" * 60)

        for _, rule in df_rules.iterrows():
            ant = rule["antecedents"]
            con = rule["consequents"]
            support = rule["support"]
            confidence = rule["confidence"]
            lift = rule["lift"]

            print(f"\n  Si '{ant}' alors '{con}'")
            print(f"    Support   : {support:.2%} des transactions")
            print(f"    Confiance : {confidence:.2%}")
            print(f"    Lift      : {lift:.2f}")

            if lift > 3:
                print(f"    -> Association FORTE (lift > 3)")
            elif lift > 1.5:
                print(f"    -> Association MODEREE (lift entre 1.5 et 3)")
            elif lift > 1:
                print(f"    -> Association FAIBLE (lift > 1)")
            else:
                print(f"    -> Association NON SIGNIFICATIVE (lift <= 1)")


def run_association_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 50)
    print("Pipeline de regles d'association")
    print("=" * 50)

    analyzer = AssociationRulesAnalyzer(min_support=0.01, min_confidence=0.5, min_lift=1.0)

    df_transactions = analyzer.prepare_transaction_data(df)

    rules = analyzer.generate_association_rules(df_transactions)

    if not rules.empty:
        top = analyzer.get_top_rules(rules, metric="lift", n=10)
        analyzer.interpret_rules(top)

        out_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "association_rules.csv")
        rules.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\nRegles sauvegardees -> {out_path}")
    else:
        print("[pipeline] aucune regle generee")

    return rules


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data/processed/products_with_clusters.csv")

    df = pd.read_csv(input_path)
    print(f"Charge: {len(df)} lignes depuis {input_path}")

    rules = run_association_pipeline(df)

    if not rules.empty:
        print("\n" + "=" * 50)
        print("Meilleures regles (Top 5 par lift)")
        print("=" * 50)
        top5 = rules.sort_values("lift", ascending=False).head(5)
        for i, (_, r) in enumerate(top5.iterrows(), 1):
            print(f"  {i}. {r['antecedents']} -> {r['consequents']}  "
                  f"(lift={r['lift']:.2f}, conf={r['confidence']:.2%})")
