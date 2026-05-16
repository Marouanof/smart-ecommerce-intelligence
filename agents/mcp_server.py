import json
import os
import time
import datetime
import csv
import io
from typing import Any, Optional, Callable
from pathlib import Path

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PERMISSIONS_PATH = BASE_DIR / "agents" / "permissions.json"
LOGS_DIR = BASE_DIR / "agents" / "logs"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "TopKselection" / "output"


class MCPError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class MCPLogger:
    def __init__(self):
        os.makedirs(LOGS_DIR, exist_ok=True)

    def _log_file(self) -> Path:
        date = datetime.date.today().isoformat()
        return LOGS_DIR / f"mcp_{date}.jsonl"

    def log(self, entry: dict):
        entry["_timestamp"] = datetime.datetime.now().isoformat()
        try:
            with open(self._log_file(), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass

    def get_recent(self, n: int = 50) -> list[dict]:
        path = self._log_file()
        if not path.exists():
            return []
        entries = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass
        return entries[-n:]

    def clear(self) -> bool:
        path = self._log_file()
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.truncate(0)
            return True
        except OSError:
            return False


class MCPPermissionManager:
    def __init__(self, permissions_path: str | Path = PERMISSIONS_PATH):
        self.permissions_path = Path(permissions_path)
        self._roles: dict = {}
        self._tools: dict = {}
        self._load()

    def _load(self):
        if self.permissions_path.exists():
            with open(self.permissions_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._roles = data.get("roles", {})
            self._tools = data.get("tools", {})

    def reload(self):
        self._load()

    def get_role_tools(self, role: str) -> list[str]:
        role_data = self._roles.get(role, {})
        return role_data.get("tools", [])

    def get_tool_def(self, tool_name: str) -> dict:
        return self._tools.get(tool_name, {})

    def check_permission(self, role: str, tool_name: str) -> bool:
        allowed = self.get_role_tools(role)
        return tool_name in allowed

    def get_all_tools(self) -> dict:
        return self._tools

    def get_all_roles(self) -> dict:
        return self._roles


class MCPTool:
    def __init__(self, name: str, handler: Callable, perm_manager: MCPPermissionManager):
        self.name = name
        self.handler = handler
        self.definition = perm_manager.get_tool_def(name)

    @property
    def description(self) -> str:
        return self.definition.get("description", "")

    @property
    def input_schema(self) -> dict:
        return self.definition.get("input_schema", {})

    def validate_params(self, params: dict) -> dict:
        schema = self.input_schema
        validated = {}
        for key, ptype in schema.items():
            if key in params:
                val = params[key]
                if ptype == "number":
                    validated[key] = float(val)
                elif ptype == "boolean":
                    validated[key] = bool(val)
                elif ptype == "string":
                    validated[key] = str(val)
                else:
                    validated[key] = val
        return validated

    def execute(self, params: dict, context: dict) -> dict:
        validated = self.validate_params(params)
        result = self.handler(validated, context)
        return result


class MCPServer:
    def __init__(self, role: str = "viewer"):
        self.role = role
        self.logger = MCPLogger()
        self.perm_manager = MCPPermissionManager()
        self._tools: dict[str, MCPTool] = {}
        self._register_tools()

    def _register_tools(self):
        self._tools["search_products"] = MCPTool("search_products", self._handle_search_products, self.perm_manager)
        self._tools["get_top_k"] = MCPTool("get_top_k", self._handle_get_top_k, self.perm_manager)
        self._tools["get_cluster_analysis"] = MCPTool("get_cluster_analysis", self._handle_get_cluster_analysis, self.perm_manager)
        self._tools["get_product_stats"] = MCPTool("get_product_stats", self._handle_get_product_stats, self.perm_manager)
        self._tools["get_association_rules"] = MCPTool("get_association_rules", self._handle_get_association_rules, self.perm_manager)
        self._tools["export_data"] = MCPTool("export_data", self._handle_export_data, self.perm_manager)
        self._tools["run_pipeline"] = MCPTool("run_pipeline", self._handle_run_pipeline, self.perm_manager)
        self._tools["manage_permissions"] = MCPTool("manage_permissions", self._handle_manage_permissions, self.perm_manager)

    def set_role(self, role: str):
        if role in self.perm_manager.get_all_roles():
            self.role = role

    def list_tools(self) -> list[dict]:
        allowed_names = self.perm_manager.get_role_tools(self.role)
        tools = []
        for name in allowed_names:
            tool = self._tools.get(name)
            if tool:
                tools.append({
                    "name": name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                })
        return tools

    def call_tool(self, tool_name: str, params: dict, user: str = "anonymous") -> dict:
        if tool_name not in self._tools:
            raise MCPError("TOOL_NOT_FOUND", f"Outil '{tool_name}' inconnu")

        if not self.perm_manager.check_permission(self.role, tool_name):
            self.logger.log({
                "event": "denied",
                "user": user,
                "role": self.role,
                "tool": tool_name,
                "params": params,
                "reason": "permission_denied",
            })
            raise MCPError("PERMISSION_DENIED", f"Le rôle '{self.role}' n'a pas accès à '{tool_name}'")

        tool = self._tools[tool_name]
        context = {"user": user, "role": self.role}

        start = time.time()
        try:
            result = tool.execute(params, context)
            status = "success"
        except MCPError as mcp_err:
            tool_def = self.perm_manager.get_tool_def(tool_name)
            if tool_def.get("requires_logging", False):
                self.logger.log({
                    "event": "error",
                    "user": user,
                    "role": self.role,
                    "tool": tool_name,
                    "params": params,
                    "error": mcp_err.message,
                })
            raise
        except Exception as e:
            status = "error"
            self.logger.log({
                "event": "error",
                "user": user,
                "role": self.role,
                "tool": tool_name,
                "params": params,
                "error": str(e),
            })
            raise MCPError("EXECUTION_ERROR", f"Erreur lors de l'execution de '{tool_name}': {e}")

        elapsed = time.time() - start

        tool_def = self.perm_manager.get_tool_def(tool_name)
        if tool_def.get("requires_logging", False):
            self.logger.log({
                "event": "execute",
                "user": user,
                "role": self.role,
                "tool": tool_name,
                "params": params,
                "elapsed_s": round(elapsed, 3),
                "status": status,
            })

        return result

    def get_logs(self, n: int = 50) -> list[dict]:
        all_logs = self.logger.get_recent(n)
        allowed_tools = self.perm_manager.get_role_tools(self.role)
        return [entry for entry in all_logs if entry.get("tool") in allowed_tools]

    def clear_logs(self, user: str = "anonymous") -> bool:
        if "manage_permissions" not in self.perm_manager.get_role_tools(self.role):
            return False
        ok = self.logger.clear()
        if ok:
            self.logger.log({
                "event": "clear_logs",
                "user": user,
                "role": self.role,
                "tool": "manage_permissions",
            })
        return ok

    # --- Tool handlers ---

    def _load_clusters_df(self) -> pd.DataFrame:
        path = DATA_PROCESSED / "products_with_clusters.csv"
        if path.exists():
            return pd.read_csv(path)
        path2 = DATA_PROCESSED / "products_with_score.csv"
        if path2.exists():
            return pd.read_csv(path2)
        raise MCPError("DATA_NOT_FOUND", "Aucune donnee traitee trouvee dans data/processed/")

    def _load_topk_df(self) -> pd.DataFrame:
        path = OUTPUT_DIR / "top_k_products.csv"
        if path.exists():
            return pd.read_csv(path)
        df = self._load_clusters_df()
        if "composite_score" in df.columns:
            return df.nlargest(100, "composite_score")
        return df.head(100)

    def _handle_search_products(self, params: dict, context: dict) -> dict:
        df = self._load_clusters_df()
        query = params.get("query", "").lower().strip()
        category = params.get("category", "").strip()
        price_min = params.get("price_min")
        price_max = params.get("price_max")
        limit = int(params.get("limit", 50))

        if query:
            mask = df.select_dtypes(include="object").apply(
                lambda col: col.astype(str).str.lower().str.contains(query, na=False)
            ).any(axis=1)
            df = df[mask]
        if category:
            df = df[df["category"].astype(str).str.lower() == category.lower()]
        if price_min is not None and "price" in df.columns:
            df = df[df["price"] >= price_min]
        if price_max is not None and "price" in df.columns:
            df = df[df["price"] <= price_max]

        total = len(df)
        df = df.head(limit)

        columns = ["title", "category", "price", "rating", "composite_score"]
        available = [c for c in columns if c in df.columns]
        records = df[available].to_dict(orient="records")

        return {
            "total": total,
            "returned": len(records),
            "products": records,
        }

    def _handle_get_top_k(self, params: dict, context: dict) -> dict:
        df = self._load_topk_df()
        k = int(params.get("k", 100))
        df = df.head(k)

        columns = ["rank", "title", "category", "platform", "price", "rating", "review_count", "composite_score"]
        available = [c for c in columns if c in df.columns]
        records = df[available].to_dict(orient="records")

        return {
            "k": k,
            "returned": len(records),
            "products": records,
        }

    def _handle_get_cluster_analysis(self, params: dict, context: dict) -> dict:
        df = self._load_clusters_df()
        if "cluster_kmeans" not in df.columns:
            return {"error": "Aucun cluster disponible", "clusters": []}

        clusters = []
        for cluster_id in sorted(df["cluster_kmeans"].dropna().unique()):
            subset = df[df["cluster_kmeans"] == cluster_id]
            clusters.append({
                "cluster_id": int(cluster_id),
                "count": len(subset),
                "avg_price": float(subset["price"].mean()) if "price" in subset.columns else 0,
                "avg_rating": float(subset["rating"].mean()) if "rating" in subset.columns else 0,
                "avg_score": float(subset["composite_score"].mean()) if "composite_score" in subset.columns else 0,
                "top_categories": (
                    subset["category"].value_counts().head(3).to_dict()
                    if "category" in subset.columns else {}
                ),
            })

        return {
            "n_clusters": len(clusters),
            "method": "KMeans",
            "clusters": clusters,
        }

    def _handle_get_product_stats(self, params: dict, context: dict) -> dict:
        df = self._load_clusters_df()
        total = len(df)

        stats = {"total_products": total}

        for col in ["price", "rating", "composite_score", "review_count", "stock_quantity"]:
            if col in df.columns:
                stats[col] = {
                    "mean": round(float(df[col].mean()), 2),
                    "min": round(float(df[col].min()), 2),
                    "max": round(float(df[col].max()), 2),
                    "median": round(float(df[col].median()), 2),
                    "std": round(float(df[col].std()), 2),
                }

        if "category" in df.columns:
            stats["top_categories"] = df["category"].value_counts().head(10).to_dict()

        if "platform" in df.columns:
            stats["platforms"] = df["platform"].value_counts().to_dict()

        if "cluster_kmeans" in df.columns:
            stats["clusters"] = df["cluster_kmeans"].value_counts().sort_index().to_dict()

        return stats

    def _handle_get_association_rules(self, params: dict, context: dict) -> dict:
        path = OUTPUT_DIR / "association_rules.csv"
        if not path.exists():
            return {"rules": [], "message": "Aucune regle d'association disponible. Lancez d'abord le pipeline."}
        df = pd.read_csv(path)
        min_lift = params.get("min_lift")
        if min_lift is not None and "lift" in df.columns:
            df = df[df["lift"] >= min_lift]
        records = df.head(50).to_dict(orient="records")
        return {
            "total_rules": len(df),
            "returned": len(records),
            "rules": records,
        }

    def _handle_export_data(self, params: dict, context: dict) -> dict:
        fmt = params.get("format", "csv")
        df = self._load_clusters_df()
        output = io.StringIO()
        if fmt == "csv":
            df.to_csv(output, index=False)
            return {
                "format": "csv",
                "data": output.getvalue(),
                "rows": len(df),
            }
        raise MCPError("INVALID_FORMAT", f"Format '{fmt}' non supporte")

    def _handle_run_pipeline(self, params: dict, context: dict) -> dict:
        skip = params.get("skip_preprocessing", False)
        try:
            from TopKselection.pipeline import MainPipeline, save_pipeline_report
            pipeline = MainPipeline()
            results = pipeline.run_full_pipeline(skip_preprocessing=skip)
            report_path = OUTPUT_DIR / "pipeline_report.txt"
            save_pipeline_report(results, str(report_path))
            return {
                "status": results.get("status", "completed"),
                "start_time": results.get("start_time", ""),
                "end_time": results.get("end_time", ""),
                "preprocessing_shape": results.get("preprocessing", {}).get("shape"),
                "rf_f1": results.get("supervised", {}).get("rf_f1"),
                "xgb_f1": results.get("supervised", {}).get("xgb_f1"),
                "n_rules": results.get("association", {}).get("n_rules", 0),
            }
        except Exception as e:
            raise MCPError("PIPELINE_ERROR", f"Erreur pipeline: {e}")

    def _handle_manage_permissions(self, params: dict, context: dict) -> dict:
        action = params.get("action", "")
        role = params.get("role", "")
        tool = params.get("tool", "")
        user = context.get("user", "anonymous")

        if context.get("role") != "admin":
            raise MCPError("PERMISSION_DENIED", "Seul l'admin peut gerer les permissions")

        self.logger.log({
            "event": "permission_change",
            "user": user,
            "action": action,
            "role": role,
            "tool": tool,
        })

        return {
            "action": action,
            "role": role,
            "tool": tool,
            "status": "logged_only",
            "message": "Changement de permissions enregistre (version statique). Modifiez permissions.json pour appliquer.",
        }
