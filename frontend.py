import os
import json
import requests
import chardet
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO, StringIO
from typing import Optional, List, Dict, Any
from model_card_generator import generate_model_card_pdf
from backend import build_cluster_profile
BACKEND_URL = "http://127.0.0.1:5000"
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")   # use small model for dev
OLLAMA_TIMEOUT = 600  # seconds, first-call warmup can be slow on Windows
from counterfactual_explanations import interactive_top_features_panel
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from audience_explanation_engine import (
    generate_audience_explanation,
    generate_clustering_audience_explanation,
    generate_dashboard_recommendation,
)
from metaknowledge_collector import load_baseline 
from tradeoff_explorer import (
    create_tradeoff_chart,
    recommend_model,
    create_comparison_table,
    calculate_pareto_frontier,
    calculate_co2_savings
)
from counterfactual_explanations import (
    get_feature_ranges,
    find_minimal_changes,
    generate_counterfactual_explanation,
    create_interactive_counterfactual,
     create_counterfactual_comparison_table
)
from fairness_detection import (
    detect_demographic_columns,
    calculate_fairness_metrics,
    detect_bias,
    generate_fairness_report,
    create_fairness_comparison_df,
    get_severity_color,
    test_fairness_significance,
    calculate_clustering_fairness,
    detect_clustering_bias,
    create_clustering_fairness_df,
)
# If a corporate proxy is set, disable for local calls
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
    os.environ.pop(k, None)
st.set_page_config(page_title="SustainaML AutoML", layout="wide")                       
st.markdown(
    """
    <style>

    /* ---------------------------------------------------------
       SustainaML visual cards
       --------------------------------------------------------- */

    .sml-card {
        background: linear-gradient(
            145deg,
            #ffffff 0%,
            #f8fafc 100%
        );
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 18px 20px;
        min-height: 118px;
        box-shadow:
            0 4px 14px rgba(15, 23, 42, 0.06);
    }

    .sml-blue {
        border-top: 4px solid #3b82f6;
    }

    .sml-green {
        border-top: 4px solid #10b981;
    }

    .sml-amber {
        border-top: 4px solid #f59e0b;
    }

    .sml-red {
        border-top: 4px solid #ef4444;
    }

    .sml-purple {
        border-top: 4px solid #8b5cf6;
    }

    .sml-card-title {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .sml-card-value {
        color: #0f172a;
        font-size: 1.35rem;
        font-weight: 750;
        line-height: 1.2;
    }

    .sml-card-subtitle {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 7px;
    }

    .sml-status-card {
        border-radius: 16px;
        padding: 22px 24px;
        min-height: 135px;
        box-shadow:
            0 4px 14px rgba(15, 23, 42, 0.06);
    }

    .sml-status-label {
        font-size: 0.80rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .sml-status-value {
        font-size: 1.65rem;
        font-weight: 750;
        margin-top: 8px;
    }

    .sml-status-good {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-left: 6px solid #10b981;
    }

    .sml-status-neutral {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 6px solid #3b82f6;
    }

    .sml-status-warning {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 6px solid #f59e0b;
    }

    .sml-status-bad {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 6px solid #ef4444;
    }

    </style>
    """,
    unsafe_allow_html=True
)
def _validate_dataset(df: pd.DataFrame, task: str) -> list[str]:
    issues = []
    if df is None or df.empty:
        issues.append("The file is empty.")
        return issues

    # Generic checks
    if df.shape[0] < 30:
        issues.append(f"Very few rows ({df.shape[0]}). Results may be unstable.")
    # Task-specific
    if task == "classification":
        if "target" not in df.columns:
            issues.append("Missing required column: 'target'.")
        else:
            n_unique = df["target"].nunique(dropna=False)
            if n_unique < 2:
                issues.append("Target has < 2 classes.")
    elif task == "clustering":
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) == 0:
            issues.append("No numeric columns for clustering.")
    return issues
def _render_clustering_result(out: dict, df_clu: pd.DataFrame):
    # ---- Results UI ----
    st.subheader(f"Best configuration • {out.get('framework','')} • k = {out.get('best_k')}")
    st.write(f"**Algorithm:** {out.get('algo','?')}  |  **DR:** {out.get('dr','-')}")
    # Metrics
    st.write("#### Internal metrics")
    st.dataframe(pd.DataFrame(out.get("metrics", {}), index=[0]))
    # Sustainability snapshot
    co2 = out.get("CO2 Emission")
    en  = out.get("Energy Consumption")
    if any(v is not None for v in (co2, en)):
        st.write("#### Sustainability")
        st.write(
            f"- **CO₂ (µkg):** {co2:.0f}  \n"
            f"- **Energy (µWh):** {en:.0f}"
        )
    # Scatter plot (PCA 2D)
    st.write("#### PCA scatter (colored by cluster)")
    try:
        emb = pd.DataFrame(out.get("embedding_2d", []), columns=["pc1","pc2"])
        emb["cluster"] = out.get("labels", [])
        fig = px.scatter(emb, x="pc1", y="pc2", color=emb["cluster"].astype(str), hover_data=emb.columns)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.caption("Plotly not available — showing table instead.")
        st.dataframe(
            pd.DataFrame(out.get("embedding_2d", []), columns=["pc1","pc2"]).assign(cluster=out.get("labels", []))
        )
    # Sizes
    st.write("#### Cluster sizes")
    if out.get("labels") is not None:
        sizes = pd.Series(out["labels"]).value_counts().sort_index()
        st.bar_chart(sizes)
    # Profiles (numeric means)
    st.write("#### Cluster profiles (numeric means)")
    num_cols = df_clu.select_dtypes(include="number").columns
    if len(num_cols):
        prof = df_clu[num_cols].assign(cluster=out.get("labels", [])).groupby("cluster").mean().round(3)
        st.dataframe(prof)
    else:
        st.info("No numeric columns for profiling.")
    # Download labeled CSV
    labeled = df_clu.copy()
    if out.get("labels") is not None:
        labeled["cluster"] = out["labels"]
    st.download_button(
        "Download labeled CSV",
        labeled.to_csv(index=False).encode("utf-8"),
        file_name="clusters.csv",
        mime="text/csv"
    )
def _summarize_interpretability_ctx(ctx: dict) -> dict:
    if not isinstance(ctx, dict):
        return {}
    techniques = ctx.get("techniques", []) or []
    compact = {}
    best_name = None
    best_iem = None
    if techniques:
        ranked = sorted(
            techniques,
            key=lambda t: float(t.get("iem", -1)),
            reverse=True
        )
        best = ranked[0]
        best_name = best.get("name")
        best_iem = best.get("iem")
        for t in ranked:
            name = str(t.get("name", "unknown")).lower()
            compact[name] = {
                "iem": t.get("iem"),
                "dimensions": t.get("dimensions", {}),
                "top_features": (t.get("artifacts", {}) or {}).get("top_features", [])
            }
        ranking = []
        for name, info in compact.items():
            ranking.append({
                "name": name,
                "iem": info.get("iem"),
                "dimensions": info.get("dimensions", {})
            })
        ranking.sort(key=lambda x: float(x.get("iem", -1)), reverse=True)
        return {
            "framework": ctx.get("framework"),
            "algorithm": ctx.get("algorithm"),
            "meta": ctx.get("meta", {}),
            "best_technique": best_name,
            "best_iem": best_iem,
            "ranking": ranking,
            "techniques": compact  }
def _direct_answer_from_interp_ctx(ctx: dict, question: str, verbose: bool = False):
    if not isinstance(ctx, dict):
        return None
    q = (question or "").lower()
    techniques = ctx.get("techniques", {}) or {}

    dims = ["fidelity", "stability", "compactness", "correctness", "simplicity", "completeness"]
    def _best_for_dim(dim_name: str):
        best_name = None
        best_val = None
        for tech_name, tech_info in techniques.items():
            val = (tech_info.get("dimensions") or {}).get(dim_name)
            if val is None:
                continue
            val = float(val)
            if best_val is None or val > best_val:
                best_val = val
                best_name = tech_name
        return best_name, best_val
    # 1) Dimension-specific comparison should come BEFORE generic "which technique"
    for dim in dims:
        if dim in q:
            best_name, best_val = _best_for_dim(dim)
            if best_name is None:
                return f"{dim} not found."

            if verbose or "why" in q or "compare" in q or "more" in q or "highest" in q:
                ranked = []
                for tech_name, tech_info in techniques.items():
                    val = (tech_info.get("dimensions") or {}).get(dim)
                    if val is not None:
                        ranked.append((tech_name, float(val)))
                ranked.sort(key=lambda x: x[1], reverse=True)

                parts = [f"{name}={val:.4f}" for name, val in ranked]
                return (
                    f"{best_name} has the highest {dim} ({best_val:.4f}). "
                    f"Ranking: " + ", ".join(parts) + "."
                )

            return f"best_{dim}={best_name} ({best_val:.4f})"

    # 2) Generic best-technique question = IEM
    if (
        "best technique" in q
        or "best explanation" in q
        or ("which technique" in q and "iem" in q)
        or ("which technique" in q and "overall" in q)
        or ("which technique" in q and "best" in q)
    ):
        if ctx.get("best_technique") is not None:
            if verbose:
                return (
                    f"{ctx['best_technique']} is the top overall technique by IEM "
                    f"with score {float(ctx.get('best_iem')):.4f}."
                )
            return f"best_technique={ctx['best_technique']} (iem={float(ctx.get('best_iem')):.4f})"
        return "best_technique not found."

    # 3) Explicit technique lookup
    for tech in ["shap", "permutation", "local_surrogate"]:
        if tech.replace("_", " ") in q or tech in q:
            info = techniques.get(tech)
            if not info:
                return f"{tech} not found."
            if verbose:
                dims_txt = ", ".join(
                    f"{k}={float(v):.4f}" for k, v in (info.get("dimensions") or {}).items()
                )
                return f"{tech}: iem={float(info.get('iem')):.4f}. {dims_txt}."
            return json.dumps(info)

    # 4) IEM lookup
    if "iem" in q and ("best" not in q):
        pieces = []
        for tech_name, tech_info in techniques.items():
            val = tech_info.get("iem")
            if val is not None:
                pieces.append(f"{tech_name}={float(val):.4f}")
        if pieces:
            return ", ".join(pieces)
        return "iem not found."

    return None
def grounded_llm_answer_interp(question: str, verbose: bool = False):
    try:
        ctx = requests.get(
            f"{BACKEND_URL}/last_runs",
            params={"task": "interpretability"},
            timeout=15
        ).json()
    except Exception as e:
        return f"Could not fetch interpretability context from backend: {e}"
    if isinstance(ctx, list) and ctx:
        last = ctx[-1]
        res = last.get("result", {}) if isinstance(last, dict) else {}
        ctx = _summarize_interpretability_ctx(res)
    else:
        return "No interpretability results found. Please run the interpretability evaluation first."

    direct = _direct_answer_from_interp_ctx(ctx, question, verbose=verbose)
    if direct is not None:
        return direct
    system = (
        "You are a copilot for an AutoML interpretability dashboard. "
        "Answer ONLY using the provided JSON context. "
        "Do not invent missing values. "
        "When the question asks about a dimension such as fidelity, stability, compactness, "
        "correctness, simplicity, or completeness, compare techniques using that dimension only, "
        "not the overall IEM. "
        "If the answer is not in the context, say 'not found'."
    )

    if verbose:
        user = (
            f"Question: {question}\n\n"
            f"Context (JSON):\n{json.dumps(ctx)[:40000]}\n\n"
            "Give a clear answer in 2 to 5 sentences. "
            "If the question asks 'why', explain using the exact numeric values from the relevant dimension."
        )
    else:
        user = (
            f"Context:\n{json.dumps(ctx)}\n\n"
            f"Question: {question}\n"
            "Answer in 2 to 5 sentences only."
        )

    if os.getenv("OLLAMA_BASE_URL"):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": f"{system}\n\n{user}",
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 180}
                },
                timeout=OLLAMA_TIMEOUT
            )
            r.raise_for_status()
            text = (r.json().get("response") or "").strip()
            return text or "No response from local model."
        except Exception as e:
            return f"Ollama call failed: {e}"
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            chat = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=0.2,
            )
            return chat.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI call failed: {e}"
    return "No LLM provider configured."
def _best_model_from_results(results: dict):
    if not isinstance(results, dict) or not results:
        return None, None
    valid = []
    for algo_name, metrics in results.items():
        if isinstance(metrics, dict) and "Accuracy" in metrics and metrics.get("Accuracy") is not None:
            valid.append((algo_name, metrics))
    if not valid:
        return None, None
    best_algo, best_metrics = max(valid, key=lambda x: float(x[1].get("Accuracy", -1)))
    return best_algo, best_metrics
def _top_feature_list(feature_importance: dict, top_k: int = 5):
    if not isinstance(feature_importance, dict) or not feature_importance:
        return []
    pairs = []
    for k, v in feature_importance.items():
        try:
            pairs.append((str(k), float(v)))
        except Exception:
            continue
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": k, "importance": v} for k, v in pairs[:top_k]]
def _latest_interpretability_summary():
    # Prefer the most recent frontend-cached interpretability result
    raw = st.session_state.get("latest_interpretability_result")
    if isinstance(raw, dict) and raw.get("techniques"):
        return _summarize_interpretability_ctx(raw)
    # Fallback to backend log
    try:
        ctx = requests.get(
            f"{BACKEND_URL}/last_runs",
            params={"task": "interpretability"},
            timeout=15
        ).json()
        if isinstance(ctx, list) and ctx:
            last = ctx[-1]
            res = last.get("result", {}) if isinstance(last, dict) else {}
            if isinstance(res, dict) and res.get("techniques"):
                return _summarize_interpretability_ctx(res)
    except Exception:
        pass

    return {}
def _build_audience_context():
    results = st.session_state.get("automl_results", {}) or {}
    best_algo, best_metrics = _best_model_from_results(results)

    interp_ctx = _latest_interpretability_summary()

    audience_ctx = {
        "best_model": best_algo,
        "best_model_metrics": {
            "accuracy": None,
            "f1": None,
            "co2": None,
            "energy": None,
        },
        "top_features": [],
        "interpretability": interp_ctx,
    }

    if isinstance(best_metrics, dict):
        audience_ctx["best_model_metrics"] = {
            "accuracy": best_metrics.get("Accuracy"),
            "f1": best_metrics.get("F1 Score"),
            "co2": best_metrics.get("CO2 Emission"),
            "energy": best_metrics.get("Energy Consumption"),
        }
        audience_ctx["top_features"] = _top_feature_list(
            best_metrics.get("feature_importance", {}),
            top_k=5
        )
    return audience_ctx

# ============================================================================
# AUDIENCE RECOMMENDATION ACTION SPACE
# Only controls that actually exist in the dashboard should be exposed
# to the recommendation system.
# ============================================================================

def _split_model_key(model_key: str):
    """
    Split result keys such as:
        FLAML_XGBoost
        H2O_GBM
        MLJAR_RF

    Returns:
        (framework, algorithm)
    """
    if not model_key or "_" not in model_key:
        return None, None

    framework, algorithm = model_key.split("_", 1)
    return framework, algorithm


def _get_classification_action_space():
    """
    Return ONLY classification controls that the current dashboard
    allows the user to modify.

    IMPORTANT:
    Random seed is intentionally excluded for now because the frontend
    sends it but /run_automl currently does not consistently use it.
    """

    results = st.session_state.get("automl_results", {}) or {}
    best_model, best_metrics = _best_model_from_results(results)

    framework, algorithm = _split_model_key(best_model)

    if not framework or not algorithm:
        return {
            "task": "classification",
            "available": False,
            "reason": "No valid best model found."
        }

    # These ranges correspond to sensible values that can be represented
    # by the existing dashboard controls.
    parameter_specs = {
        "FLAML": {
            "RF": {
                "n_estimators": {"type": "int", "min": 10, "max": 500},
                "max_depth": {"type": "int", "min": 1, "max": 50},
                "min_samples_split": {"type": "int", "min": 2, "max": 20},
                "min_samples_leaf": {"type": "int", "min": 1, "max": 20},
            },
            "XGBoost": {
                "learning_rate": {"type": "float", "min": 0.01, "max": 1.0},
                "n_estimators": {"type": "int", "min": 10, "max": 500},
                "max_depth": {"type": "int", "min": 1, "max": 20},
                "subsample": {"type": "float", "min": 0.1, "max": 1.0},
                "colsample_bytree": {"type": "float", "min": 0.1, "max": 1.0},
            },
            "LightGBM": {
                "num_leaves": {"type": "int", "min": 2, "max": 256},
                "learning_rate": {"type": "float", "min": 0.01, "max": 1.0},
                "n_estimators": {"type": "int", "min": 10, "max": 500},
                "min_child_samples": {"type": "int", "min": 1, "max": 100},
            },
            "Extra Trees": {
                "n_estimators": {"type": "int", "min": 10, "max": 500},
                "max_depth": {"type": "int", "min": 1, "max": 50},
                "min_samples_split": {"type": "int", "min": 2, "max": 20},
                "min_samples_leaf": {"type": "int", "min": 1, "max": 20},
            },
            "KNN": {
                "n_neighbors": {"type": "int", "min": 1, "max": 50},
                "weights": {
                    "type": "categorical",
                    "options": ["uniform", "distance"]
                },
            },
            "Logistic Regression": {
                "C": {"type": "float", "min": 0.001, "max": 100.0},
            },
        },

        "H2O": {
            "GLM": {
                        "alpha": {
                            "type": "float",
                            "min": 0.0,
                            "max": 1.0
                        },
                        "lambda_": {
                            "type": "float",
                            "min": 0.0001,
                            "max": 10.0
                        },
                    },
            "GBM": {
                "ntrees": {"type": "int", "min": 10, "max": 500},
                "learn_rate": {"type": "float", "min": 0.01, "max": 1.0},
                "max_depth": {"type": "int", "min": 1, "max": 30},
            },
            "Deep Learning": {
                "epochs": {"type": "int", "min": 1, "max": 200},
            },
            "Distributed RF": {
                "ntrees": {"type": "int", "min": 10, "max": 500},
                "max_depth": {"type": "int", "min": 1, "max": 30},
            },
        },

        "MLJAR": {
            "Baseline": {
                "C": {"type": "float", "min": 0.001, "max": 100.0},
            },
            "Decision Tree": {
                "max_depth": {"type": "int", "min": 1, "max": 50},
                "min_samples_split": {"type": "int", "min": 2, "max": 20},
                "min_samples_leaf": {"type": "int", "min": 1, "max": 20},
            },
            "RF": {
                "n_estimators": {"type": "int", "min": 10, "max": 500},
                "max_depth": {"type": "int", "min": 1, "max": 50},
                "min_samples_split": {"type": "int", "min": 2, "max": 20},
                "min_samples_leaf": {"type": "int", "min": 1, "max": 20},
            },
            "XGBoost": {
                "learning_rate": {"type": "float", "min": 0.01, "max": 1.0},
                "n_estimators": {"type": "int", "min": 10, "max": 500},
                "max_depth": {"type": "int", "min": 1, "max": 20},
                "subsample": {"type": "float", "min": 0.1, "max": 1.0},
                "colsample_bytree": {"type": "float", "min": 0.1, "max": 1.0},
            },
            "Extra Trees": {
                "n_estimators": {"type": "int", "min": 10, "max": 500},
                "max_depth": {"type": "int", "min": 1, "max": 50},
                "min_samples_split": {"type": "int", "min": 2, "max": 20},
                "min_samples_leaf": {"type": "int", "min": 1, "max": 20},
            },
            "LightGBM": {
                "num_leaves": {"type": "int", "min": 2, "max": 256},
                "learning_rate": {"type": "float", "min": 0.01, "max": 1.0},
                "n_estimators": {"type": "int", "min": 10, "max": 500},
                "min_child_samples": {"type": "int", "min": 1, "max": 100},
            },
            "SVM": {
                "kernel": {
                    "type": "categorical",
                    "options": ["linear", "rbf", "poly", "sigmoid"]
                },
                "C": {"type": "float", "min": 0.001, "max": 100.0},
            },
            "KNN": {
                "n_neighbors": {"type": "int", "min": 1, "max": 50},
                "weights": {
                    "type": "categorical",
                    "options": ["uniform", "distance"]
                },
            },
            "Neural Network": {
                "alpha": {
                    "type": "float",
                    "min": 0.00001,
                    "max": 0.1
                },
                "max_iter": {
                    "type": "int",
                    "min": 100,
                    "max": 1000
                },
            },
        },
    }

    hyperparameter_space = (
        parameter_specs
        .get(framework, {})
        .get(algorithm, {})
    )

    current_modified = (
        st.session_state
        .get("modified_hyperparams", {})
        .get(framework, {})
        .get(algorithm, {})
    )

    # Existing defaults from the classification dashboard.
    defaults = {
        "FLAML": {
            "RF": {"n_estimators": 100, "max_depth": 6, "min_samples_split": 2, "min_samples_leaf": 1},
            "XGBoost": {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 6, "subsample": 1.0, "colsample_bytree": 1.0},
            "LightGBM": {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100, "min_child_samples": 20},
            "Extra Trees": {"n_estimators": 100, "max_depth": 6, "min_samples_split": 2, "min_samples_leaf": 1},
            "KNN": {"n_neighbors": 5, "weights": "uniform"},
            "Logistic Regression": { "C": 1.0},
        },
        "H2O": {
            "GLM": {"alpha": 0.5, "lambda_": 0.1},
            "GBM": {"ntrees": 100, "learn_rate": 0.05, "max_depth": 6},
            "Deep Learning": {"epochs": 10},
            "Distributed RF": {"ntrees": 200, "max_depth": 6},
        },
        "MLJAR": {
            "Baseline": {"C": 1.0},
            "Decision Tree": {"max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1},
            "RF": {"n_estimators": 100, "max_depth": 6, "min_samples_split": 2, "min_samples_leaf": 1},
            "XGBoost": {"learning_rate": 0.1, "n_estimators": 100, "max_depth": 6, "subsample": 1.0, "colsample_bytree": 1.0},
            "Extra Trees": {"n_estimators": 100, "max_depth": 6, "min_samples_split": 2, "min_samples_leaf": 1},
            "LightGBM": {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100, "min_child_samples": 20},
            "SVM": {"kernel": "rbf", "C": 1.0},
            "KNN": {"n_neighbors": 5, "weights": "uniform"},
            "Neural Network": {
                "alpha": 0.0001,
                "max_iter": 500,
            },
        },
    }

    default_values = (
        defaults
        .get(framework, {})
        .get(algorithm, {})
    )

    # Actual parameter values reported by the backend for the model
    # that produced the current result.
    actual_run_params = {}

    if isinstance(best_metrics, dict):
        actual_run_params = (
            best_metrics.get("hyperparameters", {})
            or {}
        )

    current_values = {}

    for parameter in hyperparameter_space.keys():

        # Highest priority: what backend says was actually used.
        if parameter in actual_run_params:
            current_values[parameter] = actual_run_params.get(
                parameter
            )

        # Fallback: explicitly confirmed dashboard value.
        elif parameter in current_modified:
            current_values[parameter] = current_modified.get(
                parameter
            )

        # Final fallback: frontend display default.
        else:
            current_values[parameter] = default_values.get(
                parameter
            )

    return {
        "task": "classification",
        "available": bool(hyperparameter_space),
        "model": best_model,
        "framework": framework,
        "algorithm": algorithm,
        "current_values": current_values,
        "allowed_hyperparameters": hyperparameter_space,

        # Time budget is already an actual dashboard control.
        "time_budget": {
            "current": int(st.session_state.get("time_budget", 30)),
            "allowed_values": [10, 30, 60, 120],
        },
        "recommendation_rule": "change_exactly_one_control",
    }
def _get_clustering_action_space():
    """
    Return ONLY controls currently available in the clustering dashboard.
    """

    return {
        "task": "clustering",
        "available": True,

        "k_min": {
            "current": int(st.session_state.get("clu_kmin", 2)),
            "min": 2,
            "max": 50,
            "type": "int",
        },

        "k_max": {
            "current": int(st.session_state.get("clu_kmax", 8)),
            "min": 3,
            "max": 100,
            "type": "int",
        },

        "optimization_metric": {
            "current": st.session_state.get(
                "clu_metric",
                "silhouette"
            ),
            "allowed_values": [
                "silhouette",
                "calinski_harabasz",
                "davies_bouldin",
            ],
        },

        "time_budget_sec": {
            "current": int(st.session_state.get("clu_time", 60)),
            "allowed_values": [15, 30, 60, 120],
        },

        "n_trials": {
            "current": int(st.session_state.get("clu_trials", 30)),
            "min": 5,
            "max": 200,
            "step": 5,
            "type": "int",
        },

        "random_seed": {
            "current": int(st.session_state.get("clu_seed", 42)),
            "min": 0,
            "max": 2_147_483_647,
            "type": "int",
        },

        "recommendation_rule": "change_exactly_one_control",
    }
def _extract_model_info_for_explanation():
    """Extract model info from AutoML results for explanation engine."""
    results = st.session_state.get("automl_results", {}) or {}
    best_algo, best_metrics = _best_model_from_results(results)
    
    model_info = {
        "name": best_algo or "Unknown Model",
        "framework": best_algo.split("_")[0] if best_algo and "_" in best_algo else "Unknown",
        "hyperparameters": str(best_metrics.get("hyperparameters", "Default")) if best_metrics else ""
    }
    return model_info
def _extract_features_for_explanation(top_k: int = 5):
    """Extract top features from AutoML results."""
    results = st.session_state.get("automl_results", {}) or {}
    best_algo, best_metrics = _best_model_from_results(results)
    if not best_metrics:
        return []
    feature_importance = best_metrics.get("feature_importance", {})
    if not feature_importance:
        return []
    # Sort by importance and get top K
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(float(x[1])) if isinstance(x[1], (int, float)) else 0,
        reverse=True
    ) 
    features_info = [
        {
            "name": name,
            "importance": float(importance) if isinstance(importance, (int, float)) else 0
        }
        for name, importance in sorted_features[:top_k]
    ]
    return features_info
def _extract_metrics_for_explanation():
    """Extract metrics from AutoML results."""
    results = st.session_state.get("automl_results", {}) or {}
    best_algo, best_metrics = _best_model_from_results(results)
    if not best_metrics:
        return {}
    metrics_info = {
    "accuracy": float(best_metrics.get("Accuracy", 0)),
    "f1_score": float(best_metrics.get("F1 Score", 0)),
    "precision": float(best_metrics.get("Precision", 0)) if best_metrics.get("Precision") else None,
    "recall": float(best_metrics.get("Recall", 0)) if best_metrics.get("Recall") else None,
    "false_negative_rate": (
    float(
        best_metrics.get(
            "false_negative_rate"
        )
    )
    if best_metrics.get(
        "false_negative_rate"
    ) is not None
    else None
),
    "roc_auc": float(best_metrics.get("ROC-AUC", 0)) if best_metrics.get("ROC-AUC") else None,
    "co2_emission": float(best_metrics.get("CO2 Emission")) if best_metrics.get("CO2 Emission") is not None else None,
    "energy_consumption": float(best_metrics.get("Energy Consumption")) if best_metrics.get("Energy Consumption") is not None else None,  
    "target_description": "target outcome",
    "sustainability_note": "Model was trained with sustainability tracking enabled",
    "all_algorithms_metrics": [
        {
            "algorithm": algo,
            "co2_emission": float(m.get("CO2 Emission")) if m.get("CO2 Emission") is not None else None,
            "energy_consumption": float(m.get("Energy Consumption")) if m.get("Energy Consumption") is not None else None,
            "accuracy": float(m.get("Accuracy", 0)),
        }
        for algo, m in results.items()
    ]
} 
    return metrics_info
def _compare_classification_validation(
    baseline: dict,
    validation: dict
) -> dict:
    """
    Compare one recommendation rerun against the original baseline.

    This is a SINGLE-RUN comparison, not statistical proof that the
    recommendation generalizes.
    """

    def _num(d, key):
        try:
            value = d.get(key)

            return (
                float(value)
                if value is not None
                else None
            )
        except (TypeError, ValueError):
            return None

    metrics = {
        "Accuracy": {
            "baseline": _num(baseline, "Accuracy"),
            "validation": _num(validation, "Accuracy"),
            "direction": "higher",
        },
        "F1 Score": {
            "baseline": _num(baseline, "F1 Score"),
            "validation": _num(validation, "F1 Score"),
            "direction": "higher",
        },
        "ROC-AUC": {
            "baseline": _num(baseline, "ROC-AUC"),
            "validation": _num(validation, "ROC-AUC"),
            "direction": "higher",
        },

        "Log Loss": {
            "baseline": _num(baseline, "Log Loss"),
            "validation": _num(validation, "Log Loss"),
            "direction": "lower",
        },
        "CO2 Emission": {
            "baseline": _num(baseline, "CO2 Emission"),
            "validation": _num(validation, "CO2 Emission"),
            "direction": "lower",
        },
        "Energy Consumption": {
            "baseline": _num(
                baseline,
                "Energy Consumption"
            ),
            "validation": _num(
                validation,
                "Energy Consumption"
            ),
            "direction": "lower",
        },
    }

    for _, values in metrics.items():
        b = values["baseline"]
        v = values["validation"]

        values["delta"] = (
            v - b
            if b is not None and v is not None
            else None
        )

        values["percent_change"] = (
            ((v - b) / abs(b)) * 100
            if (
                b is not None
                and v is not None
                and b != 0
            )
            else None
        )

    accuracy_delta = (
        metrics["Accuracy"]["delta"]
        if metrics["Accuracy"]["delta"] is not None
        else 0.0
    )

    f1_delta = (
        metrics["F1 Score"]["delta"]
        if metrics["F1 Score"]["delta"] is not None
        else 0.0
    )

    energy_pct = (
        metrics["Energy Consumption"]["percent_change"]
    )

    co2_pct = (
        metrics["CO2 Emission"]["percent_change"]
    )

    # Small tolerances prevent tiny numerical changes from being
    # described as meaningful quality improvements.
    quality_tol = 0.001

    quality_improved = (
        accuracy_delta > quality_tol
        or f1_delta > quality_tol
    )

    quality_worsened = (
        accuracy_delta < -quality_tol
        or f1_delta < -quality_tol
    )

    sustainability_improved = (
        (
            energy_pct is not None
            and energy_pct < -5.0
        )
        or
        (
            co2_pct is not None
            and co2_pct < -5.0
        )
    )

    sustainability_worsened = (
        (
            energy_pct is not None
            and energy_pct > 5.0
        )
        or
        (
            co2_pct is not None
            and co2_pct > 5.0
        )
    )

    if quality_improved and not sustainability_worsened:
        outcome = "quality_improved"

    elif (
        sustainability_improved
        and not quality_worsened
    ):
        outcome = "sustainability_improved"

    elif (
        quality_improved
        and sustainability_worsened
    ) or (
        quality_worsened
        and sustainability_improved
    ):
        outcome = "tradeoff"

    elif quality_worsened and not sustainability_improved:
        outcome = "worse"

    else:
        outcome = "no_clear_change"

    return {
        "metrics": metrics,
        "outcome": outcome,
        "quality_tolerance": quality_tol,
        "sustainability_tolerance_percent": 5.0,
        "interpretation_scope": "single_run",
    }
def _run_classification_validation_request(
    dataset_json,
    framework,
    algorithm,
    hyperparams,
    time_budget,
    seed
):
    payload = {
        "data": dataset_json,
        "framework": framework,
        "algorithm": algorithm,
        "hyperparams": hyperparams,
        "time_budget": int(time_budget),
        "seed": int(seed),
    }

    response = requests.post(
        f"{BACKEND_URL}/validate_classification_recommendation",
        json=payload,
        timeout=600
    )

    response.raise_for_status()

    body = response.json()

    if body.get("status") != "success":
        raise RuntimeError(
            body.get(
                "message",
                "Classification validation failed."
            )
        )

    return body.get("result", {})
def _summarize_repeated_classification_validation(
    baseline_runs: list,
    recommended_runs: list
) -> dict:
    """
    Summarize paired repeated validation runs.

    Each baseline run and recommended run at index i must use
    the same experimental seed.

    In addition to mean/SD, this function computes robust statistics:
    median, IQR, paired median delta, and IQR-based outlier flags.
    """

    metric_keys = [
        "Accuracy",
        "F1 Score",
        "ROC-AUC",
        "Log Loss",
        "CO2 Emission",
        "Energy Consumption",
    ]

    summary = {}

    def _iqr_stats(values):
        """
        Return median, Q1, Q3, IQR and IQR-based outlier information.
        """
        arr = np.asarray(
            values,
            dtype=float
        )

        if len(arr) == 0:
            return {
                "median": None,
                "q1": None,
                "q3": None,
                "iqr": None,
                "lower_fence": None,
                "upper_fence": None,
                "outlier_count": 0,
                "outlier_indices": [],
                "outlier_values": [],
            }

        q1 = float(
            np.percentile(arr, 25)
        )

        q3 = float(
            np.percentile(arr, 75)
        )

        iqr = q3 - q1

        lower_fence = (
            q1 - 1.5 * iqr
        )

        upper_fence = (
            q3 + 1.5 * iqr
        )

        outlier_indices = [
            int(i)
            for i, value in enumerate(arr)
            if (
                value < lower_fence
                or value > upper_fence
            )
        ]

        outlier_values = [
            float(arr[i])
            for i in outlier_indices
        ]

        return {
            "median": float(
                np.median(arr)
            ),
            "q1": q1,
            "q3": q3,
            "iqr": float(iqr),
            "lower_fence": float(
                lower_fence
            ),
            "upper_fence": float(
                upper_fence
            ),
            "outlier_count": len(
                outlier_indices
            ),
            "outlier_indices": (
                outlier_indices
            ),
            "outlier_values": (
                outlier_values
            ),
        }

    for metric in metric_keys:

        baseline_values = [
            float(run[metric])
            for run in baseline_runs
            if run.get(metric) is not None
        ]

        recommended_values = [
            float(run[metric])
            for run in recommended_runs
            if run.get(metric) is not None
        ]

        if (
            not baseline_values
            or not recommended_values
            or len(baseline_values)
            != len(recommended_values)
        ):
            summary[metric] = {
                "baseline_mean": None,
                "baseline_std": None,
                "baseline_median": None,
                "baseline_iqr": None,
                "recommended_mean": None,
                "recommended_std": None,
                "recommended_median": None,
                "recommended_iqr": None,
                "mean_delta": None,
                "delta_std": None,
                "median_delta": None,
                "delta_iqr": None,
                "baseline_outlier_count": 0,
                "recommended_outlier_count": 0,
                "paired_delta_values": [],
            }
            continue

        baseline_arr = np.asarray(
            baseline_values,
            dtype=float
        )

        recommended_arr = np.asarray(
            recommended_values,
            dtype=float
        )

        paired_delta = (
            recommended_arr
            - baseline_arr
        )
        # Pair-level consistency:
        # how often the recommended configuration was better,
        # worse, or effectively equal across matched seeds.
        if metric in [
            "Accuracy",
            "F1 Score",
            "ROC-AUC",
        ]:
            improved_mask = paired_delta > 0
            worsened_mask = paired_delta < 0

        elif metric in [
            "Log Loss",
            "CO2 Emission",
            "Energy Consumption",
        ]:
            improved_mask = paired_delta < 0
            worsened_mask = paired_delta > 0

        else:
            improved_mask = np.zeros(
                len(paired_delta),
                dtype=bool
            )
            worsened_mask = np.zeros(
                len(paired_delta),
                dtype=bool
            )

        improved_count = int(
            np.sum(improved_mask)
        )

        worsened_count = int(
            np.sum(worsened_mask)
        )

        unchanged_count = int(
            len(paired_delta)
            - improved_count
            - worsened_count
        )
        baseline_robust = (
            _iqr_stats(
                baseline_arr
            )
        )

        recommended_robust = (
            _iqr_stats(
                recommended_arr
            )
        )

        delta_robust = (
            _iqr_stats(
                paired_delta
            )
        )

        baseline_mean = float(
            np.mean(baseline_arr)
        )

        recommended_mean = float(
            np.mean(recommended_arr)
        )

        baseline_std = (
            float(
                np.std(
                    baseline_arr,
                    ddof=1
                )
            )
            if len(baseline_arr) > 1
            else 0.0
        )

        recommended_std = (
            float(
                np.std(
                    recommended_arr,
                    ddof=1
                )
            )
            if len(recommended_arr) > 1
            else 0.0
        )

        summary[metric] = {
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,

            "baseline_median": (
                baseline_robust[
                    "median"
                ]
            ),

            "baseline_iqr": (
                baseline_robust[
                    "iqr"
                ]
            ),

            "recommended_mean": (
                recommended_mean
            ),

            "recommended_std": (
                recommended_std
            ),

            "recommended_median": (
                recommended_robust[
                    "median"
                ]
            ),

            "recommended_iqr": (
                recommended_robust[
                    "iqr"
                ]
            ),

            "mean_delta": float(
                np.mean(
                    paired_delta
                )
            ),

            "delta_std": (
                float(
                    np.std(
                        paired_delta,
                        ddof=1
                    )
                )
                if len(paired_delta) > 1
                else 0.0
            ),

            "median_delta": (
                delta_robust[
                    "median"
                ]
            ),

            "delta_iqr": (
                delta_robust[
                    "iqr"
                ]
            ),

            "baseline_outlier_count": (
                baseline_robust[
                    "outlier_count"
                ]
            ),

            "baseline_outlier_indices": (
                baseline_robust[
                    "outlier_indices"
                ]
            ),

            "baseline_outlier_values": (
                baseline_robust[
                    "outlier_values"
                ]
            ),

            "recommended_outlier_count": (
                recommended_robust[
                    "outlier_count"
                ]
            ),

            "recommended_outlier_indices": (
                recommended_robust[
                    "outlier_indices"
                ]
            ),

            "recommended_outlier_values": (
                recommended_robust[
                    "outlier_values"
                ]
            ),
            "improved_count": improved_count,
            "worsened_count": worsened_count,
            "unchanged_count": unchanged_count,
            "total_pairs": int(
                len(paired_delta)
            ),
            "paired_delta_values": (
                paired_delta.tolist()
            ),
        }

    return {
        "n_repeats": len(
            baseline_runs
        ),
        "summary": summary,
        "baseline_runs": baseline_runs,
        "recommended_runs": recommended_runs,
        "design": "paired_seed_validation",
    }
def _interpret_repeated_classification_validation(
    repeated_bundle: dict
) -> dict:
    """
    Produce a deterministic empirical interpretation.

    IMPORTANT:
    - This is not an LLM judgment.
    - Thresholds are practical tolerances, not statistical significance.
    - Sustainability is marked uncertain when measurements show
      substantial instability/outliers.
    """

    summary = repeated_bundle.get(
        "summary",
        {}
    )

    # --------------------------------------------------------------
    # Practical tolerances
    # --------------------------------------------------------------

    accuracy_tol = 0.005
    f1_tol = 0.005
    roc_auc_tol = 0.005
    log_loss_tol = 0.01

    sustainability_tol_percent = 5.0

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------

    def _value(
        metric,
        field,
        default=None
    ):
        return (
            summary
            .get(metric, {})
            .get(field, default)
        )

    # --------------------------------------------------------------
    # PREDICTIVE EFFECT
    #
    # Use paired MEDIAN deltas to reduce sensitivity to unusual runs.
    # --------------------------------------------------------------

    accuracy_delta = _value(
        "Accuracy",
        "median_delta"
    )

    f1_delta = _value(
        "F1 Score",
        "median_delta"
    )

    roc_auc_delta = _value(
        "ROC-AUC",
        "median_delta"
    )

    log_loss_delta = _value(
        "Log Loss",
        "median_delta"
    )

    predictive_positive_votes = 0
    predictive_negative_votes = 0

    if accuracy_delta is not None:
        if accuracy_delta > accuracy_tol:
            predictive_positive_votes += 1
        elif accuracy_delta < -accuracy_tol:
            predictive_negative_votes += 1

    if f1_delta is not None:
        if f1_delta > f1_tol:
            predictive_positive_votes += 1
        elif f1_delta < -f1_tol:
            predictive_negative_votes += 1

    if roc_auc_delta is not None:
        if roc_auc_delta > roc_auc_tol:
            predictive_positive_votes += 1
        elif roc_auc_delta < -roc_auc_tol:
            predictive_negative_votes += 1

    # Lower Log Loss is better.
    if log_loss_delta is not None:
        if log_loss_delta < -log_loss_tol:
            predictive_positive_votes += 1
        elif log_loss_delta > log_loss_tol:
            predictive_negative_votes += 1

    if (
        predictive_positive_votes > 0
        and predictive_negative_votes == 0
    ):
        predictive_status = "improved"

    elif (
        predictive_negative_votes > 0
        and predictive_positive_votes == 0
    ):
        predictive_status = "worsened"

    elif (
        predictive_positive_votes > 0
        and predictive_negative_votes > 0
    ):
        predictive_status = "mixed"

    else:
        predictive_status = "no_clear_change"

    # --------------------------------------------------------------
    # SUSTAINABILITY VARIABILITY CHECK
    # --------------------------------------------------------------

    sustainability_metrics = [
        "CO2 Emission",
        "Energy Consumption",
    ]

    sustainability_unstable = False
    variability_reasons = []

    for metric in sustainability_metrics:

        metric_info = summary.get(
            metric,
            {}
        )

        recommended_mean = (
            metric_info.get(
                "recommended_mean"
            )
        )

        recommended_std = (
            metric_info.get(
                "recommended_std"
            )
        )

        recommended_outliers = (
            metric_info.get(
                "recommended_outlier_count",
                0
            )
        )

        # Coefficient of variation.
        cv = None

        if (
            recommended_mean is not None
            and recommended_std is not None
            and recommended_mean != 0
        ):
            cv = abs(
                recommended_std
                / recommended_mean
            )

        if recommended_outliers > 0:
            sustainability_unstable = True

            variability_reasons.append(
                f"{metric}: "
                f"{recommended_outliers} IQR-based outlier(s)"
            )

        if (
            cv is not None
            and cv > 0.50
        ):
            sustainability_unstable = True

            variability_reasons.append(
                f"{metric}: high relative variability "
                f"(SD/mean={cv:.2f})"
            )

    # --------------------------------------------------------------
    # SUSTAINABILITY DIRECTION
    #
    # Use medians instead of means.
    # --------------------------------------------------------------

    sustainability_votes_improved = 0
    sustainability_votes_worsened = 0

    for metric in sustainability_metrics:

        metric_info = summary.get(
            metric,
            {}
        )

        baseline_median = (
            metric_info.get(
                "baseline_median"
            )
        )

        recommended_median = (
            metric_info.get(
                "recommended_median"
            )
        )

        if (
            baseline_median is None
            or recommended_median is None
            or baseline_median == 0
        ):
            continue

        pct_change = (
            (
                recommended_median
                - baseline_median
            )
            / abs(
                baseline_median
            )
        ) * 100.0

        if (
            pct_change
            < -sustainability_tol_percent
        ):
            sustainability_votes_improved += 1

        elif (
            pct_change
            > sustainability_tol_percent
        ):
            sustainability_votes_worsened += 1

    if sustainability_unstable:
        sustainability_status = (
            "uncertain_due_to_variability"
        )

    elif (
        sustainability_votes_improved > 0
        and sustainability_votes_worsened == 0
    ):
        sustainability_status = "improved"

    elif (
        sustainability_votes_worsened > 0
        and sustainability_votes_improved == 0
    ):
        sustainability_status = "worsened"

    elif (
        sustainability_votes_improved > 0
        and sustainability_votes_worsened > 0
    ):
        sustainability_status = "mixed"

    else:
        sustainability_status = "no_clear_change"

    # --------------------------------------------------------------
    # OVERALL EMPIRICAL OUTCOME
    #
    # Do not call this "LLM correctness".
    # --------------------------------------------------------------

    if (
        predictive_status == "improved"
        and sustainability_status == "improved"
    ):
        overall_status = (
            "supported_on_both_objectives"
        )

    elif (
        predictive_status == "improved"
        and sustainability_status
        == "no_clear_change"
    ):
        overall_status = (
            "predictive_improvement_no_clear_sustainability_change"
        )

    elif (
        predictive_status == "improved"
        and sustainability_status
        == "worsened"
    ):
        overall_status = "tradeoff"

    elif (
        predictive_status == "improved"
        and sustainability_status
        == "uncertain_due_to_variability"
    ):
        overall_status = (
            "predictive_improvement_sustainability_uncertain"
        )

    elif (
        predictive_status == "worsened"
        and sustainability_status
        != "improved"
    ):
        overall_status = "not_supported"

    else:
        overall_status = "inconclusive"

    return {
        "predictive_status": (
            predictive_status
        ),

        "sustainability_status": (
            sustainability_status
        ),

        "overall_status": (
            overall_status
        ),

        "predictive_votes": {
            "positive": (
                predictive_positive_votes
            ),
            "negative": (
                predictive_negative_votes
            ),
        },

        "sustainability_variability": {
            "unstable": (
                sustainability_unstable
            ),
            "reasons": (
                variability_reasons
            ),
        },

        "practical_tolerances": {
            "accuracy": accuracy_tol,
            "f1": f1_tol,
            "roc_auc": roc_auc_tol,
            "log_loss": log_loss_tol,
            "sustainability_percent": (
                sustainability_tol_percent
            ),
        },

        "interpretation_note": (
            "This is a deterministic practical-effect interpretation "
            "of repeated paired runs. The thresholds are not statistical "
            "significance tests."
        ),
    }
def _format_repeated_validation_for_audience(
    repeated_bundle: dict,
    audience_type: str
) -> str:
    """
    Convert the deterministic repeated-validation result into
    audience-appropriate wording.

    No LLM is used here.
    """

    empirical = repeated_bundle.get(
        "empirical_interpretation",
        {}
    )

    recommendation = repeated_bundle.get(
        "recommendation",
        {}
    )

    predictive_status = empirical.get(
        "predictive_status",
        "unknown"
    )

    sustainability_status = empirical.get(
        "sustainability_status",
        "unknown"
    )

    control = recommendation.get(
        "control",
        "the recommended control"
    )

    current_value = recommendation.get(
        "current_value"
    )

    suggested_value = recommendation.get(
        "suggested_value"
    )

    n_repeats = repeated_bundle.get(
        "n_repeats",
        0
    )

    change_text = (
        f"{control}: {current_value} → {suggested_value}"
    )

    # --------------------------------------------------------------
    # NORMAL USER
    # --------------------------------------------------------------
    if audience_type == "Normal User":

        predictive_text = {
            "improved": (
                "the model performed better overall"
            ),
            "worsened": (
                "the model performed worse overall"
            ),
            "mixed": (
                "the model results were mixed"
            ),
            "no_clear_change": (
                "there was no clear change in model performance"
            ),
        }.get(
            predictive_status,
            "the performance result was unclear"
        )

        sustainability_text = {
            "improved": (
                "energy and environmental measurements also improved"
            ),
            "worsened": (
                "energy and environmental measurements became worse"
            ),
            "no_clear_change": (
                "there was no clear sustainability change"
            ),
            "uncertain_due_to_variability": (
                "the sustainability measurements varied too much "
                "to reach a reliable conclusion"
            ),
            "mixed": (
                "the sustainability results were mixed"
            ),
        }.get(
            sustainability_status,
            "the sustainability result was unclear"
        )

        return (
            f"We tested **{change_text}** across "
            f"**{n_repeats} matched runs**. "
            f"Across those tests, {predictive_text}. "
            f"For sustainability, {sustainability_text}. "
            f"This repeated result is more informative than relying "
            f"on a single rerun."
        )

    # --------------------------------------------------------------
    # DOMAIN EXPERT
    # --------------------------------------------------------------
    if audience_type == "Domain Expert":

        return (
            f"The proposed dashboard change **{change_text}** was "
            f"evaluated across **{n_repeats} matched validation runs**. "
            f"The repeated analysis classified the predictive effect as "
            f"**{predictive_status.replace('_', ' ')}** and the "
            f"sustainability effect as "
            f"**{sustainability_status.replace('_', ' ')}**. "
            f"The sustainability conclusion incorporates observed "
            f"run-to-run variability rather than relying on a single "
            f"measurement."
        )

    # --------------------------------------------------------------
    # DATA SCIENTIST
    # --------------------------------------------------------------
    return (
        f"Repeated paired validation of **{change_text}** "
        f"across **{n_repeats} matched seeds** produced a deterministic "
        f"predictive classification of "
        f"**{predictive_status.replace('_', ' ')}** and a sustainability "
        f"classification of "
        f"**{sustainability_status.replace('_', ' ')}**. "
        f"The verdict is based on paired repeated measurements with "
        f"mean/SD, median/IQR, pair-level consistency, and variability "
        f"diagnostics; it is not generated by the LLM."
    )
def _extract_xai_signals_for_explanation():
    """
    Return only XAI/trustworthiness indicators actually produced
    by the current AutoML result.

    Missing indicators remain unavailable rather than receiving
    synthetic fallback scores.
    """

    results = (
        st.session_state.get(
            "automl_results",
            {}
        )
        or {}
    )

    _, best_metrics = _best_model_from_results(
        results
    )

    empty_signals = {
        "interpretability": None,
        "stability": None,
        "fairness": None,
        "calibration": None,
    }

    if not best_metrics:
        return empty_signals

    xai_signals = (
        best_metrics.get(
            "xai_signals",
            {}
        )
        or {}
    )

    signals = {
        "interpretability": xai_signals.get(
            "interpretability"
        ),
        "stability": xai_signals.get(
            "stability"
        ),
        "fairness": xai_signals.get(
            "fairness"
        ),
        "calibration": xai_signals.get(
            "calibration"
        ),
    }

    return {
        key: value
        for key, value in signals.items()
        if value is not None
    }
def auto_detect_domain(df):
    cols = " ".join(df.columns.astype(str)).lower()
    if any(x in cols for x in ["patient", "disease", "cholesterol", "blood", "heart", "diagnosis"]):
        return "Medical"
    if any(x in cols for x in ["loan", "credit", "income", "bank", "mortgage"]):
        return "Finance"
    if any(x in cols for x in ["student", "grade", "exam", "school"]):
        return "Education"
    if any(x in cols for x in ["customer", "sales", "revenue", "product"]):
        return "Business"
    return "General"
def grounded_llm_answer_audience(
    audience: str,
    focus: str,
    followup_question: str = "",
    verbose: bool = False,
    conversation_history: Optional[List[Dict[str, Any]]] = None
):
    ctx = _build_audience_context()

    if not ctx.get("best_model"):
        return "No AutoML results found. Please run AutoML first."

    audience = (audience or "Executive").strip()
    focus = (focus or "overall summary").strip()
    followup_question = (followup_question or "").strip()
    conversation_history = conversation_history or []
    system = (
        "You are a copilot for an AutoML dashboard with persistent audience-aware explanations. "
        "Answer ONLY using the provided JSON context and the prior audience conversation. "
        "Do not invent facts. "
        "Keep the same audience style across turns. "
        "If the user asks a follow-up, answer it consistently with the previously selected audience and focus. "
        "If the context is missing something, say so clearly."
    )
    if verbose:
        length_rule = "Write 2 to 4 short paragraphs."
    else:
        length_rule = "Write 4 to 7 concise sentences."
    history_block = ""
    if conversation_history:
        trimmed = conversation_history[-6:]
        hist_lines = []
        for turn in trimmed:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            hist_lines.append(f"{role.upper()}: {content}")
        history_block = "\n".join(hist_lines)
    user = (
        f"Audience: {audience}\n"
        f"Focus: {focus}\n"
        f"Current user message: {followup_question if followup_question else 'Provide an audience-specific summary.'}\n\n"
        f"Prior audience conversation:\n{history_block if history_block else 'None'}\n\n"
        f"Context JSON:\n{json.dumps(ctx)[:40000]}\n\n"
        f"Task:\n"
        f"1. Stay consistent with the selected audience.\n"
        f"2. Answer the current message using the prior audience conversation when relevant.\n"
        f"3. Mention performance in audience-appropriate language.\n"
        f"4. Mention interpretability findings if available.\n"
        f"5. Mention sustainability metrics briefly if relevant.\n"
        f"6. {length_rule}"
    )
    if os.getenv("OLLAMA_BASE_URL"):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": f"{system}\n\n{user}",
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 350}
                },
                timeout=OLLAMA_TIMEOUT
            )
            r.raise_for_status()
            text = (r.json().get("response") or "").strip()
            return text or "No response from local model."
        except Exception as e:
            return f"Ollama call failed: {e}"

    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            chat = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=0.2,
            )
            return chat.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI call failed: {e}"

    return "No LLM provider configured."
if "audience_selected_type" not in st.session_state:
    st.session_state["audience_selected_type"] = "Normal User"

if "selected_domain" not in st.session_state:
    st.session_state["selected_domain"] = "Medical"

if "explanation_results" not in st.session_state:
    st.session_state["explanation_results"] = {}
if "classification_recommendation" not in st.session_state:
    st.session_state["classification_recommendation"] = None

if "classification_validation_result" not in st.session_state:
    st.session_state["classification_validation_result"] = None

if "classification_validation_baseline" not in st.session_state:
    st.session_state["classification_validation_baseline"] = None

if "classification_validation_error" not in st.session_state:
    st.session_state["classification_validation_error"] = None

if "classification_recommendation_context" not in st.session_state:
    st.session_state["classification_recommendation_context"] = None
if "classification_repeated_validation" not in st.session_state:
    st.session_state["classification_repeated_validation"] = None

if "classification_repeated_validation_error" not in st.session_state:
    st.session_state["classification_repeated_validation_error"] = None

if "clustering_recommendation" not in st.session_state:
    st.session_state["clustering_recommendation"] = None

if "clustering_recommendation_context" not in st.session_state:
    st.session_state["clustering_recommendation_context"] = None
if "generating_explanation" not in st.session_state:
    st.session_state["generating_explanation"] = False
if "show_tradeoff_panel" not in st.session_state:
    st.session_state["show_tradeoff_panel"] = False
 
if "sustainability_weight" not in st.session_state:
    st.session_state["sustainability_weight"] = 0.5
if "show_counterfactual_panel" not in st.session_state:
    st.session_state["show_counterfactual_panel"] = False
if "show_fairness_panel" not in st.session_state:
    st.session_state["show_fairness_panel"] = False
# --- 1) Task chooser (top of app) ---
st.write("")
task = st.radio(
    "What would you like to do?",
    ["Classification", "Clustering "],
    index=0,
    horizontal=True,
    help="Pick your task first. Classification uses your existing H2O / MLJAR / FLAML UI. Clustering opens the new AutoML panel."
)
# small helper for robust CSV reading if you already have a util; otherwise fallback
def _read_csv_any(file):
    try:
        return pd.read_csv(file)
    except Exception:
        file.seek(0)
        return pd.read_excel(file)

if task.startswith("Clustering"):
    st.sidebar.header("Dataset (Unlabeled)")
    clu_file = st.sidebar.file_uploader("Upload CSV (no target column)", type=["csv"])
    if clu_file is None:
        st.info("👋 Upload a CSV file to continue.")
        st.stop()
    df_clu = pd.read_csv(clu_file)
    # Fill missing values with mean
    if df_clu.isnull().sum().sum() > 0:
        st.sidebar.warning(f"⚠️ Found {df_clu.isnull().sum().sum()} missing values. Filling with mean")
        df_clu = df_clu.fillna(df_clu.mean(numeric_only=True))
    st.sidebar.info(f"✓ Data: {df_clu.shape[0]} rows, {df_clu.shape[1]} columns")
    # Dataset Insights button
    show_insights = st.sidebar.button("📊 Dataset Insights", key="clu_insights_btn")
    st.sidebar.markdown("---")
    # NOW render insights (df_clu exists)
    if show_insights:
        st.write("## Dataset Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Shape:** {df_clu.shape[0]} rows × {df_clu.shape[1]} columns")
            st.write(f"**Missing values:** {df_clu.isnull().sum().sum()}")
            st.write("**Data types:**")
            st.write(df_clu.dtypes)
        with col2:
            st.write("**Numeric stats:**")
            st.write(df_clu.describe())
        
        st.write("### Correlation Heatmap")
        import plotly.figure_factory as ff
        corr = df_clu.corr(numeric_only=True)
        fig = ff.create_annotated_heatmap(
            z=corr.values, 
            x=list(corr.columns), 
            y=list(corr.columns), 
            colorscale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
    st.sidebar.markdown("---")
    st.sidebar.header("Framework & Settings")
    # 1) Multi-select frameworks
    frameworks = st.sidebar.multiselect(
        "Clustering frameworks",

        ["SklearnAutoCluster", "Clusteval", "OptunaAutoCluster"],
        default=["SklearnAutoCluster"],
        key="clu_frameworks"
    )
    if not frameworks:
        st.info("👋 Select at least one framework to continue.")
        st.stop()
    # 2) Per-framework algorithm checkboxes
    _defaults = {
        "SklearnAutoCluster": ["KMeans", "MiniBatchKMeans", "GaussianMixture", "Agglomerative", "Birch", "HDBSCAN"],
        "Clusteval": ["kmeans", "agglomerative", "dbscan", "hdbscan"],  # lowercase, different set
        "OptunaAutoCluster": ["best"]  # Auto-searches all algorithms
    }
    algorithms = {}
    for fw in frameworks:
        st.sidebar.markdown(f"**{fw} algorithms**")
        algorithms[fw] = {}
        choices = _defaults.get(fw, [])
        for name in choices:
            algorithms[fw][name] = st.sidebar.checkbox(
                name,
                value=(name in choices[:2]),
                key=f"clu_{fw}_{name}"
            )
    # 3) Global k range and search settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Search Parameters")
    k_min = st.sidebar.number_input(
        "k min", 
        min_value=2, 
        max_value=50, 
        value=2, 
        step=1,
        key="clu_kmin"
    )
    k_max = st.sidebar.number_input(
        "k max", 
        min_value=3, 
        max_value=100, 
        value=8, 
        step=1,
        key="clu_kmax"
    )
    # 4) Optimization metric
    metric = st.sidebar.selectbox(
        "Optimization metric",
        ["silhouette", "calinski_harabasz", "davies_bouldin"],
        index=0,
        key="clu_metric"
    )
    # 5) Time budget
    time_budget_sec = st.sidebar.selectbox(
        "Time budget (sec)",
        [15, 30, 60, 120],
        index=2,
        key="clu_time"
    )
    # 6) Optuna trials (for OptunaAutoCluster)
    n_trials = st.sidebar.number_input(
        "Optuna trials",
        min_value=5,
        max_value=200,
        value=30,
        step=5,
        key="clu_trials"
    )
    # 7) Random seed
    clu_seed = st.sidebar.number_input(
        "Random seed (clustering)",
        min_value=0,
        max_value=2_147_483_647,
        value=42,
        step=1,
        key="clu_seed"
    )
    # 8) Preset save/load
    st.sidebar.markdown("---")
    preset = {
        "frameworks": frameworks,
        "algorithms": {
            fw: {name: bool(val) for name, val in algorithms.get(fw, {}).items()}
            for fw in frameworks
        },
        "k_min": int(k_min),
        "k_max": int(k_max),
        "metric": metric,
        "time_budget_sec": int(time_budget_sec),
        "n_trials": int(n_trials),
        "seed": int(clu_seed),
    }
    # 9) Main run button
    run_btn = st.sidebar.button("▶️ Run Clustering", key="clu_run", use_container_width=True)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Tool")
    if st.sidebar.button("📊 Clustering Model Leaderboard", key="clu_leaderboard_btn"):
        st.session_state["show_clu_interpretability"] = False
        st.session_state["show_clu_leaderboard"] = True
        st.session_state["show_clu_feature_importance"] = False
        st.session_state["show_clu_profile"] = False
        st.session_state["show_clu_tradeoff"] = False
        st.session_state["show_run_success"] = False
        st.session_state["show_clu_audience_explanation"] = False
        st.session_state["show_clu_timebudget"] = False
        st.session_state["show_clu_modelcard"] = False 
        st.rerun()
    if st.sidebar.button("🎯 Feature Importance", key="clu_feature_importance_btn"):
        st.session_state["show_clu_feature_importance"] = True
        st.session_state["show_clu_interpretability"] = False
        st.session_state["show_clu_leaderboard"] = False
        st.session_state["show_clu_profile"] = False
        st.session_state["show_clu_tradeoff"] = False
        st.session_state["show_run_success"] = False
        st.session_state["show_clu_audience_explanation"] = False
        st.session_state["show_clu_timebudget"] = False
        st.session_state["show_clu_modelcard"] = False 
        st.rerun()
    if st.sidebar.button("🌱 Sustainability-Silhouette Trade-off", key="clu_tradeoff_btn"):
        st.session_state["show_clu_tradeoff"] = True
        st.session_state["show_clu_interpretability"] = False
        st.session_state["show_clu_leaderboard"] = False
        st.session_state["show_clu_feature_importance"] = False
        st.session_state["show_clu_profile"] = False
        st.session_state["show_run_success"] = False
        st.session_state["show_clu_audience_explanation"] = False
        st.session_state["show_clu_modelcard"] = False 
        st.rerun()
    if st.sidebar.button("👥 Audience Explanation Panel", key="clu_audience_btn"):
        st.session_state["show_clu_audience_explanation"] = True
        st.session_state["show_clu_interpretability"] = False
        st.session_state["show_clu_leaderboard"] = False
        st.session_state["show_clu_feature_importance"] = False
        st.session_state["show_clu_profile"] = False
        st.session_state["show_clu_tradeoff"] = False
        st.session_state["show_run_success"] = False
        st.session_state["show_clu_timebudget"] = False
        st.session_state["show_clu_modelcard"] = False 
        st.rerun()
    if st.sidebar.button("🔍 Interpretability Panel", key="clu_interpretability_btn"):
        st.session_state["show_clu_interpretability"] = True
        st.session_state["show_clu_leaderboard"] = False
        st.session_state["show_clu_feature_importance"] = False
        st.session_state["show_clu_profile"] = False
        st.session_state["show_clu_tradeoff"] = False
        st.session_state["show_run_success"] = False
        st.session_state["show_clu_audience_explanation"] = False
        st.session_state["show_clu_timebudget"] = False
        st.session_state["show_clu_modelcard"] = False 
        st.rerun()
    if st.sidebar.button("⏱️ Time Budgets Comparison", key="clu_timebudget_btn"):
        st.session_state["show_clu_timebudget"] = True
        st.session_state["show_clu_leaderboard"] = False
        st.session_state["show_clu_feature_importance"] = False
        st.session_state["show_clu_profile"] = False
        st.session_state["show_clu_tradeoff"] = False
        st.session_state["show_clu_interpretability"] = False
        st.session_state["show_clu_audience_explanation"] = False
        st.session_state["show_clu_modelcard"] = False 
        st.rerun()
    if st.sidebar.button("📋 Download Clustering Model Card (PDF)", key="clu_modelcard_btn"):
        st.session_state["show_clu_modelcard"] = True
        st.session_state["show_clu_leaderboard"] = False
        st.session_state["show_clu_feature_importance"] = False
        st.session_state["show_clu_profile"] = False
        st.session_state["show_clu_tradeoff"] = False
        st.session_state["show_clu_interpretability"] = False
        st.session_state["show_clu_audience_explanation"] = False
        st.session_state["show_clu_timebudget"] = False
        st.rerun()

    if run_btn:
        payload = {
            "data": df_clu.to_json(),
            "frameworks": frameworks,
            "algorithms": algorithms,  # {"FW": {"Algo": True/False}}
            "metric": metric,
            "k_range": [int(k_min), int(k_max)],
            "time_budget_sec": int(time_budget_sec),
            "n_trials": int(n_trials),
            "seed": int(clu_seed),   
        }
        try:
            with st.spinner("Running clustering..."):
                r = requests.post(
                    f"{BACKEND_URL}/run_clustering_compare",
                    json=payload,
                    timeout=1200
                )
            r.raise_for_status()
            # Accumulate results instead of replacing
            new_results = r.json().get("results", {})
            if "clu_results" not in st.session_state:
                st.session_state["clu_results"] = {}

            # Rename keys to include time_budget to avoid overwrites
            for key, val in new_results.items():
                time_budget = val.get("time_budget_sec", time_budget_sec)
                k_tag = f"k{int(k_min)}-{int(k_max)}"
                new_key = f"{key}_{time_budget}s_{k_tag}"
                st.session_state["clu_results"][new_key] = val
        except Exception as e:
            st.session_state["clu_results"] = {}
            st.error(f"Compare request failed: {e}")

    # Always render whatever we have in session (persists across reruns)
    clu_out = st.container()
    with clu_out:
        clu_res = st.session_state.get("clu_results", {})
        if not clu_res:
            st.info("Run clustering to see results here.")
        else:
            # Build results table with metrics + sustainability
            rows = []
            for key, v in clu_res.items():
                if "error" in v:
                    rows.append({"Run": key, "Error": v["error"]})
                else:
                    m = v.get("metrics", {})
                    rows.append({
                        "Run": key,
                        "Framework": v.get("framework"),
                        "Algo": v.get("algo"),
                        "k": v.get("best_k"),
                        "Silhouette": m.get("Silhouette"),
                        "CH": m.get("Calinski-Harabasz"),
                        "DB": m.get("Davies-Bouldin"),
                        "CO2 (µkg)": v.get("CO2 Emission"),
                        "Energy (µWh)": v.get("Energy Consumption"),
                    })
            if not st.session_state.get("show_clu_leaderboard") and not st.session_state.get("show_clu_feature_importance") and not st.session_state.get("show_clu_profile") and not st.session_state.get("show_clu_tradeoff"):
                st.session_state["show_run_success"] = True
            if st.session_state.get("show_run_success"):
                st.success("✅ Clustering run completed! Check **Analysis Tools** to view results.")
            if st.session_state.get("show_clu_leaderboard"):
                clu_res = st.session_state.get("clu_results", {})
                if clu_res:
                    st.subheader("🏆 Clustering Leaderboard")
                    
                    leaderboard_rows = []
                    for key, v in clu_res.items():
                        if "error" not in v:
                            m = v.get("metrics", {})
                            leaderboard_rows.append({
                                "Run": key,
                                "Framework": v.get("framework"),
                                "Algorithm": v.get("algo"),
                                "k": v.get("best_k"),
                                "Silhouette": round(m.get("Silhouette", 0), 3),
                                "CH Index": round(m.get("Calinski-Harabasz", 0), 1),
                                "DB Index": round(m.get("Davies-Bouldin", 0), 3),
                                "CO2 (µkg)": round(v.get("CO2 Emission", 0), 6),
                            })
                    df_leaderboard = pd.DataFrame(leaderboard_rows)
                    
                    # Top Performers Cards
                    st.write("### 🌟 Top Performers")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        best_sil = df_leaderboard.loc[df_leaderboard["Silhouette"].idxmax()]
                        st.metric("Best Silhouette", f"{best_sil['Silhouette']:.3f}", delta=f"{best_sil['Run'][:20]}")
                    with col2:
                        best_ch = df_leaderboard.loc[df_leaderboard["CH Index"].idxmax()]
                        st.metric("Best CH Index", f"{best_ch['CH Index']:.1f}", delta=f"{best_ch['Run'][:20]}")
                    with col3:
                        best_db = df_leaderboard.loc[df_leaderboard["DB Index"].idxmin()]
                        st.metric("Best DB Index", f"{best_db['DB Index']:.3f}", delta=f"{best_db['Run'][:20]}")
                    with col4:
                        best_eco = df_leaderboard.loc[df_leaderboard["CO2 (µkg)"].idxmin()]
                        st.metric("Most Sustainable", f"{best_eco['CO2 (µkg)']:.6f}", delta=f"{best_eco['Run'][:20]}")
                    
                    # Metrics Comparison Charts
                    st.write("### 📊 Metrics Comparison")
                    tab1, tab2, tab3 = st.tabs(["Clustering Quality", "Efficiency", "Data Table"])
                    
                    with tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_sil = go.Figure()
                            fig_sil.add_trace(go.Bar(x=df_leaderboard["Run"], y=df_leaderboard["Silhouette"], name="Silhouette", marker_color="lightblue"))
                            fig_sil.update_layout(title="Silhouette Score (Higher is Better)", xaxis_tickangle=-45, height=400)
                            st.plotly_chart(fig_sil, use_container_width=True)
                        with col2:
                            fig_ch = go.Figure()
                            fig_ch.add_trace(go.Bar(x=df_leaderboard["Run"], y=df_leaderboard["CH Index"], name="CH Index", marker_color="lightgreen"))
                            fig_ch.update_layout(title="Calinski-Harabasz Index (Higher is Better)", xaxis_tickangle=-45, height=400)
                            st.plotly_chart(fig_ch, use_container_width=True)
                    
                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_db = go.Figure()
                            fig_db.add_trace(go.Bar(x=df_leaderboard["Run"], y=df_leaderboard["DB Index"], name="DB Index", marker_color="lightsalmon"))
                            fig_db.update_layout(title="Davies-Bouldin Index (Lower is Better)", xaxis_tickangle=-45, height=400)
                            st.plotly_chart(fig_db, use_container_width=True)
                        with col2:
                            fig_co2 = go.Figure()
                            fig_co2.add_trace(go.Bar(x=df_leaderboard["Run"], y=df_leaderboard["CO2 (µkg)"], name="CO2", marker_color="gold"))
                            fig_co2.update_layout(title="CO2 Emissions (Lower is Better)", xaxis_tickangle=-45, height=400)
                            st.plotly_chart(fig_co2, use_container_width=True)
                    
                    with tab3:
                        metric_col = st.selectbox("Sort by:", ["Silhouette", "CH Index", "DB Index", "CO2 (µkg)"])
                        ascending = metric_col in ["DB Index", "CO2 (µkg)"]
                        df_sorted = df_leaderboard.sort_values(by=metric_col, ascending=ascending)
                        st.dataframe(df_sorted, use_container_width=True)
                    # ===== Cluster Visualizations =====
                    st.write("### 🔬 Cluster Visualizations")
                    ok_run_keys = [k for k, v in clu_res.items() if "error" not in v]
                    default_idx = ok_run_keys.index(best_sil["Run"]) if best_sil["Run"] in ok_run_keys else 0
                    viz_run = st.selectbox(
                        "Select run to visualize:",
                        ok_run_keys,
                        index=default_idx,
                        key="clu_leaderboard_viz_run"
                    )
                    viz_out = clu_res[viz_run]
                    viz_labels = np.array(viz_out.get("labels", []))

                    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                        "2D PCA Scatter",
                        "Silhouette Analysis",
                        "Elbow Curve (k selection)",
                        "Cluster Sizes"
                    ])

                    # --- 2D PCA Scatter Plot ---
                    with viz_tab1:
                        emb_data = viz_out.get("embedding_2d", [])
                        if len(emb_data) and len(viz_labels):
                            emb_df = pd.DataFrame(emb_data, columns=["pc1", "pc2"])
                            emb_df["cluster"] = viz_labels.astype(str)
                            fig_pca = px.scatter(
                                emb_df, x="pc1", y="pc2", color="cluster",
                                title=f"2D PCA Scatter — {viz_run}",
                                labels={"pc1": "PC1", "pc2": "PC2"}
                            )
                            fig_pca.update_layout(height=450)
                            st.plotly_chart(fig_pca, use_container_width=True)
                        else:
                            st.info("No 2D embedding available for this run.")

                    # --- Silhouette Analysis Plot ---
                    with viz_tab2:
                        if len(viz_labels) and len(np.unique(viz_labels)) > 1:
                            from sklearn.metrics import silhouette_samples
                            X_viz = df_clu.select_dtypes(include="number").values
                            if X_viz.shape[0] == len(viz_labels):
                                sample_sil = silhouette_samples(X_viz, viz_labels)
                                unique_c = np.unique(viz_labels)
                                per_cluster_sil = [sample_sil[viz_labels == c].mean() for c in unique_c]
                                sil_df = pd.DataFrame({
                                    "Cluster": [f"Cluster {c}" for c in unique_c],
                                    "Avg Silhouette": per_cluster_sil
                                })
                                colors = ["crimson" if s < 0 else "seagreen" for s in per_cluster_sil]
                                fig_sil_analysis = go.Figure(data=[go.Bar(
                                    x=sil_df["Cluster"], y=sil_df["Avg Silhouette"],
                                    marker_color=colors
                                )])
                                fig_sil_analysis.add_hline(y=0, line_dash="dash", line_color="gray")
                                fig_sil_analysis.update_layout(
                                    title=f"Per-Cluster Silhouette Coefficient — {viz_run}",
                                    yaxis_title="Avg Silhouette Coefficient",
                                    height=450
                                )
                                st.plotly_chart(fig_sil_analysis, use_container_width=True)
                                weak_clusters = sil_df[sil_df["Avg Silhouette"] < 0]["Cluster"].tolist()
                                if weak_clusters:
                                    st.warning(f"⚠️ Weak clusters (negative silhouette): {', '.join(weak_clusters)}")
                                else:
                                    st.success("✅ All clusters have positive average silhouette.")
                            else:
                                st.info("Feature data shape doesn't match labels for this run — cannot compute silhouette samples.")
                        else:
                            st.info("Need at least 2 clusters with labels to compute silhouette analysis.")

                    # --- Elbow Curve / Quality vs k ---
                    with viz_tab3:
                        if df_leaderboard["k"].nunique() > 1:
                            elbow_metric = st.selectbox(
                                "Metric:", ["Silhouette", "CH Index", "DB Index"], key="clu_elbow_metric"
                            )
                            elbow_df = df_leaderboard.sort_values("k")
                            fig_elbow = go.Figure()
                            fig_elbow.add_trace(go.Scatter(
                                x=elbow_df["k"], y=elbow_df[elbow_metric],
                                mode="lines+markers", marker=dict(size=10), line=dict(color="royalblue")
                            ))
                            fig_elbow.update_layout(
                                title=f"{elbow_metric} vs k",
                                xaxis_title="Number of Clusters (k)",
                                yaxis_title=elbow_metric,
                                height=450
                            )
                            st.plotly_chart(fig_elbow, use_container_width=True)
                        else:
                            st.info("Only one k value found across runs — run clustering with multiple k values to see the elbow curve.")

                    # --- Cluster Size Distribution ---
                    with viz_tab4:
                        if len(viz_labels):
                            size_series = pd.Series(viz_labels).value_counts().sort_index()
                            size_df = pd.DataFrame({
                                "Cluster": [f"Cluster {c}" for c in size_series.index],
                                "Points": size_series.values
                            })
                            fig_sizes = go.Figure(data=[go.Bar(
                                x=size_df["Cluster"], y=size_df["Points"], marker_color="mediumpurple"
                            )])
                            fig_sizes.update_layout(
                                title=f"Cluster Size Distribution — {viz_run}",
                                yaxis_title="Number of Points",
                                height=450
                            )
                            st.plotly_chart(fig_sizes, use_container_width=True)
                            if size_series.max() / size_series.min() > 5:
                                st.warning("⚠️ Clusters are highly imbalanced (largest is 5x+ the smallest).")
                        else:
                            st.info("No labels available for this run.")
                else:
                    st.warning("Run clustering first.")
            if st.session_state.get("show_clu_feature_importance", False):
                st.subheader("🎯 Feature Importance Analysis")
                clu_res = st.session_state.get("clu_results", {})
                if clu_res:
                    run_options = [k for k, v in clu_res.items() if "error" not in v]
                    if run_options:
                        selected_run = st.selectbox("Select clustering run:", run_options)
                        run_data = clu_res[selected_run]
                        
                        feature_imp = run_data.get("feature_importance", {})
                        if feature_imp:
                            df_imp = pd.DataFrame(list(feature_imp.items()), columns=["Feature", "Importance"])
                            df_imp = df_imp.sort_values("Importance", ascending=False)
                            df_imp["Importance_Percent"] = (df_imp["Importance"] * 100).round(2)
                            
                            # ──── TOP METRICS ────
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "🏆 Top Feature",
                                    df_imp.iloc[0]["Feature"],
                                    f"{df_imp.iloc[0]['Importance_Percent']:.1f}%"
                                )
                            
                            with col2:
                                st.metric(
                                    "📊 Total Features",
                                    len(df_imp),
                                    f"{df_imp['Importance'].sum():.2f}"
                                )
                            
                            with col3:
                                avg_importance = df_imp["Importance"].mean()
                                st.metric(
                                    "📈 Avg Importance",
                                    f"{avg_importance:.4f}",
                                    f"{(avg_importance*100):.2f}%"
                                )
                            
                            with col4:
                                top_3_sum = df_imp.head(3)["Importance"].sum()
                                st.metric(
                                    "🎯 Top 3 Sum",
                                    f"{top_3_sum:.2f}",
                                    f"{(top_3_sum*100):.1f}% of total"
                                )
                            
                            # ──── VISUALIZATIONS ────
                            st.markdown("---")
                            
                            viz_col1, viz_col2 = st.columns([2, 1])
                            
                            # Horizontal bar chart with color gradient
                            with viz_col1:
                                st.write("**Top 10 Features by Importance**")
                                df_top10 = df_imp.head(10)
                                
                                # Create color mapping (red to green gradient)
                                max_imp = df_top10["Importance"].max()
                                colors = []
                                for imp in df_top10["Importance"]:
                                    ratio = imp / max_imp if max_imp > 0 else 0
                                    if ratio >= 0.7:
                                        colors.append('#2ecc71')  # Green
                                    elif ratio >= 0.4:
                                        colors.append('#f39c12')  # Orange
                                    else:
                                        colors.append('#e74c3c')  # Red
                                
                                fig = px.bar(
                                    df_top10, 
                                    x="Importance", 
                                    y="Feature", 
                                    orientation="h",
                                    title="Feature Importance Ranking"
                                )
                                fig.update_traces(marker_color=colors)
                                fig.update_layout(
                                    height=400,
                                    xaxis_title="Importance Score",
                                    yaxis_title="Feature",
                                    font=dict(size=12),
                                    hovermode='closest'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Distribution pie chart (top 5 + others)
                            with viz_col2:
                                st.write("**Importance Distribution**")
                                top_5 = df_imp.head(5)
                                others_sum = df_imp.iloc[5:]["Importance"].sum()
                                
                                dist_data = list(top_5[["Feature", "Importance"]].values)
                                if others_sum > 0:
                                    dist_data.append(["Others", others_sum])
                                
                                df_pie = pd.DataFrame(dist_data, columns=["Feature", "Importance"])
                                fig_pie = px.pie(
                                    df_pie, 
                                    names="Feature", 
                                    values="Importance",
                                    hole=0.3
                                )
                                fig_pie.update_layout(
                                    height=400,
                                    font=dict(size=11),
                                    showlegend=True
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # ──── FEATURE STATS TABLE ────
                            st.markdown("---")
                            st.write("**All Features with Importance Scores**")
                            
                            # Add rank column
                            df_display = df_imp.copy()
                            df_display["Rank"] = range(1, len(df_display) + 1)
                            df_display = df_display[["Rank", "Feature", "Importance", "Importance_Percent"]]
                            df_display.columns = ["Rank", "Feature Name", "Importance Score", "Importance %"]
                            
                            # Format the dataframe for display
                            df_display["Importance Score"] = df_display["Importance Score"].apply(lambda x: f"{x:.6f}")
                            df_display["Importance %"] = df_display["Importance %"].apply(lambda x: f"{x:.2f}%")
                            
                            # Color code the rows based on importance
                            def highlight_importance(row):
                                imp_pct = float(row["Importance %"].rstrip('%'))
                                if imp_pct >= 20:
                                    return ['background-color: #d5f4e6'] * len(row)
                                elif imp_pct >= 5:
                                    return ['background-color: #fef3cd'] * len(row)
                                else:
                                    return ['background-color: #f8d7da'] * len(row)
                            
                            styled_df = df_display.style.apply(highlight_importance, axis=1)
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # ──── INSIGHTS ────
                            with st.expander("💡 Insights & Interpretation", expanded=False):
                                top_feature = df_imp.iloc[0]
                                top_3 = df_imp.head(3)
                                
                                st.write(f"""
                                **Key Findings:**
                                
                                🔝 **Top Feature:** `{top_feature['Feature']}` with importance score of **{top_feature['Importance']:.4f}** ({top_feature['Importance_Percent']:.2f}%)
                                
                                This feature has the strongest impact on cluster separation.
                                
                                📊 **Top 3 Contributors:**
                                """)
                                
                                for idx, (_, row) in enumerate(top_3.iterrows(), 1):
                                    st.write(f"{idx}. `{row['Feature']}` - {row['Importance_Percent']:.2f}%")
                                
                                st.write(f"""
                                ✨ **Interpretation:**
                                - Features with higher importance scores better distinguish between clusters
                                - The color coding shows: 🟢 High (≥20%) | 🟡 Medium (5-20%) | 🔴 Low (<5%)
                                - Consider these top features for domain analysis and business insights
                                """)
                        else:
                            st.info("No feature importance data available for this run.")
                    else:
                        st.warning("No successful clustering runs to analyze.")
                else:
                    st.warning("Run clustering first.")
            if st.session_state.get("show_clu_tradeoff"):
                clu_res = st.session_state.get("clu_results", {})
                if not clu_res:
                    st.warning("Run clustering first.")
                else:
                    st.subheader("🌱 Sustainability vs Clustering Quality Trade-off")
                    selected_metric = st.selectbox(
                        "Compare Sustainability with:",
                        ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"],
                        key="tradeoff_metric_select"
                    )
                    # Build trade-off data8
                    tradeoff_data = []
                    for key, v in clu_res.items():
                        if "error" not in v:
                            m = v.get("metrics", {})
                            tradeoff_data.append({
                                "Run": key,
                                "CO2 (µkg)": v.get("CO2 Emission", 0),
                                "Silhouette": m.get("Silhouette", 0),
                                "Calinski-Harabasz": m.get("Calinski-Harabasz", 0),
                                "Davies-Bouldin": m.get("Davies-Bouldin", 0),
                                "Framework": v.get("framework"),
                                "Algorithm": v.get("algo"),
                                "k": v.get("best_k")
                            })            
                    df_tradeoff = pd.DataFrame(tradeoff_data)
                    
                    fig = go.Figure()
                    
                    # Add scatter for each framework
                    for fw in df_tradeoff["Framework"].unique():
                        df_fw = df_tradeoff[df_tradeoff["Framework"] == fw]
                        # Invert DB so higher is always better on the chart
                        metric_values = df_fw[selected_metric].copy()
                        if selected_metric == "Davies-Bouldin":
                            metric_values = -metric_values  # Negate DB so lower DB = higher y-value

                        fig.add_trace(go.Scatter(
                            x=df_fw["CO2 (µkg)"],
                            y=metric_values,
                            mode='markers+text',
                            name=fw,
                            marker=dict(size=12, opacity=0.7),
                            text=[f"{r['Algorithm']}<br>k={r['k']}" for _, r in df_fw.iterrows()],
                            textposition="top center",
                            hovertemplate="<b>%{text}</b><br>CO2: %{x:.6f} µkg<br>" + selected_metric + ": %{y:.3f}<extra></extra>"
                        ))                    
                    fig.update_layout(
                        title=f"Sustainability vs {selected_metric} Trade-off",
                        xaxis_title="CO2 Emissions (µkg) ↓ Lower is Better",
                        yaxis_title=f"{selected_metric} {'↑ Higher is Better' if selected_metric != 'Davies-Bouldin' else '↓ Lower is Better (inverted on chart)'}",  
                        hovermode="closest",
                        height=600,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary
                    st.write("### Summary")

                    # For best quality: use idxmax for Silhouette/CH, idxmin for DB (lower is better)
                    if selected_metric == "Davies-Bouldin":
                        best_quality = df_tradeoff.loc[df_tradeoff[selected_metric].idxmin()]
                    else:
                        best_quality = df_tradeoff.loc[df_tradeoff[selected_metric].idxmax()]

                    best_eco = df_tradeoff.loc[df_tradeoff["CO2 (µkg)"].idxmin()]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Best Score:** {best_quality['Run']}\n{selected_metric}: {best_quality[selected_metric]:.3f}")
                    with col2:
                        st.info(f"**Most Sustainable:** {best_eco['Run']}\nCO2: {best_eco['CO2 (µkg)']:.6f} µkg")
                            
            if st.session_state.get("show_clu_interpretability"):
                clu_res = st.session_state.get("clu_results", {})
                if not clu_res:
                    st.warning("Run clustering first.")
                else:
                    st.subheader("🔍 Clustering Interpretability Panel")
                    # Get best run
                    ok_runs = [k for k, v in clu_res.items() if "error" not in v]
                    if not ok_runs:
                        st.warning("No successful runs.")
                    else:
                        # Let user select which run to analyze
                        selected_run = st.selectbox("Select run to analyze:", ok_runs, key="clu_interp_run")
                        labels = np.array(clu_res[selected_run]["labels"])    
                        # TAB 1: CLUSTER COMPOSITION
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "Cluster Composition",
                            "Feature Separation",
                            "Outlier Detection",
                            "Feature Distributions"
                        ])
                        with tab1:
                            st.write("### Cluster Composition")
                            unique, counts = np.unique(labels, return_counts=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                fig_pie = go.Figure(data=[go.Pie(
                                    labels=[f"Cluster {i}" for i in unique],
                                    values=counts,
                                    hole=0.3
                                )])
                                fig_pie.update_layout(title="Cluster Sizes", height=400)
                                st.plotly_chart(fig_pie, use_container_width=True) 
                            with col2:
                                comp_data = {
                                    "Cluster": [f"Cluster {i}" for i in unique],
                                    "Points": counts,
                                    "Percentage": [f"{c/len(labels)*100:.1f}%" for c in counts],
                                    "Density": [f"{c/len(labels):.3f}" for c in counts]
                                }
                                st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
                        with tab2:
                            st.write("### Feature Importance for Cluster Separation")
                            from sklearn.metrics import silhouette_samples
                            X = df_clu.values
                            feature_scores = []
                            for feat_idx in range(X.shape[1]):
                                # Between-cluster variance / Within-cluster variance
                                feat_col = X[:, feat_idx]
                                cluster_means = np.array([feat_col[labels == c].mean() for c in unique])
                                global_mean = feat_col.mean()
                                between_var = sum((cluster_means[c] - global_mean)**2 * counts[np.where(unique == c)[0][0]] for c in unique)
                                within_var = sum(np.var(feat_col[labels == c]) * counts[np.where(unique == c)[0][0]] for c in unique)
                                ratio = between_var / (within_var + 1e-10)
                                feature_scores.append(ratio)
                            feature_importance_df = pd.DataFrame({
                                "Feature": df_clu.columns,
                                "Separation Score": np.array(feature_scores) / np.max(feature_scores)
                            }).sort_values("Separation Score", ascending=False)
                            fig_feat = go.Figure(data=[go.Bar(
                                x=feature_importance_df["Separation Score"],
                                y=feature_importance_df["Feature"],
                                orientation='h',
                                marker_color="lightblue"
                            )])
                            fig_feat.update_layout(title="Features Distinguishing Clusters", xaxis_title="Importance", height=400)
                            st.plotly_chart(fig_feat, use_container_width=True)
                        with tab3:
                            st.write("### Outlier Detection per Cluster")
                            X = df_clu.values
                            outlier_threshold = st.slider("Outlier threshold (percentile)", 50, 99, 90)
                            outliers_per_cluster = {}
                            for c in unique:
                                cluster_points = X[labels == c]
                                center = cluster_points.mean(axis=0)
                                distances = np.linalg.norm(cluster_points - center, axis=1)
                                threshold = np.percentile(distances, outlier_threshold)
                                outliers_per_cluster[f"Cluster {c}"] = int(sum(distances > threshold))
                            outlier_df = pd.DataFrame({
                                "Cluster": list(outliers_per_cluster.keys()),
                                "Outliers": list(outliers_per_cluster.values())
                            })
                            fig_out = go.Figure(data=[go.Bar(
                                x=outlier_df["Cluster"],
                                y=outlier_df["Outliers"],
                                marker_color="lightsalmon"
                            )])
                            fig_out.update_layout(title=f"Outliers per Cluster (>{outlier_threshold}th percentile distance)", height=400)
                            st.plotly_chart(fig_out, use_container_width=True)
                        with tab4:
                            st.write("### Feature Distributions per Cluster")
                            feature_to_plot = st.selectbox("Select feature:", df_clu.columns, key="clu_interp_feat")
                            X = df_clu.values
                            feat_idx = list(df_clu.columns).index(feature_to_plot)
                            fig_dist = go.Figure()
                            for c in unique:
                                feat_values = X[labels == c, feat_idx]
                                fig_dist.add_trace(go.Box(
                                    y=feat_values,
                                    name=f"Cluster {c}",
                                    boxmean='sd'
                                ))
                            fig_dist.update_layout(
                                title=f"Distribution of '{feature_to_plot}' across Clusters",
                                yaxis_title=feature_to_plot,
                                height=400
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)
            if st.session_state.get("show_clu_timebudget"):
                clu_res = st.session_state.get("clu_results", {})
                if not clu_res or not any("error" not in v for v in clu_res.values()):
                    st.warning("Run clustering first.")
                else:
                    st.subheader("⏱️ Time Budgets Comparison")
                    
                    # Build comparison table
                    rows = []
                    for key, v in clu_res.items():
                        if "error" not in v:
                            m = v.get("metrics", {})
                            rows.append({
                                "Run": key,
                                "Time Budget (s)": v.get("time_budget_sec", "N/A"),
                                "Framework": v.get("framework"),
                                "Algorithm": v.get("algo"),
                                "Silhouette": round(m.get("Silhouette", 0), 3),
                                "CH Index": round(m.get("Calinski-Harabasz", 0), 1),
                                "DB Index": round(m.get("Davies-Bouldin", 0), 3),                   
                                "CO2 (µkg)": round(v.get("CO2 Emission", 0), 6),
                                "Energy (µWh)": round(v.get("Energy Consumption", 0), 6),
                            })
                    
                    df_tb = pd.DataFrame(rows)
                    
                    # 1) Metric Table
                    st.write("### Metric Comparison")
                    st.dataframe(df_tb.sort_values("Time Budget (s)"), use_container_width=True)
                    
                    # 2) Energy vs CO2 scatter
                    st.write("### Energy vs CO2 for Each Framework (log scale)")                    
                    fig = go.Figure()
                    for fw in df_tb["Framework"].unique():
                        df_fw = df_tb[df_tb["Framework"] == fw]
                        fig.add_trace(go.Scatter(
                            x=df_fw["CO2 (µkg)"],
                            y=df_fw["Energy (µWh)"],
                            mode='markers+text',
                            name=fw,
                            marker=dict(size=20, opacity=0.8),
                            text=[f"{int(t)}s" for t in df_fw["Time Budget (s)"]],
                            textposition="middle center",
                            textfont=dict(size=11, color="black", family="Arial Black")
                        ))
                    
                    co2_range = df_tb["CO2 (µkg)"].values
                    energy_range = df_tb["Energy (µWh)"].values
                    co2_padding = (co2_range.max() - co2_range.min()) * 0.2 + 2 if co2_range.max() > co2_range.min() else 2
                    energy_padding = (energy_range.max() - energy_range.min()) * 0.2 + 5 if energy_range.max() > energy_range.min() else 5
                    
                    fig.update_layout(
                        title="",
                        xaxis_title="CO2 Emission (µkg)",
                        yaxis_title="Energy Consumption (µWh)",
                        yaxis_type="log",
                        xaxis=dict(range=[max(0, co2_range.min() - co2_padding), co2_range.max() + co2_padding]),
                        height=500,
                        hovermode="closest",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 3) Quality Improvement
                    st.write("### Quality Improvement vs Time Budget")
                    quality_metric = st.selectbox(
                        "Select Quality Metric:",
                        ["Silhouette", "CH Index", "DB Index"],
                        key="timebudget_metric_select"
                    )
                    fig2 = go.Figure()
                    for fw in df_tb["Framework"].unique():
                        df_fw = df_tb[df_tb["Framework"] == fw]
                        fig2.add_trace(go.Scatter(
                            x=df_fw["Time Budget (s)"],
                            y=df_fw[quality_metric],
                            mode='markers',
                            name=fw,
                            marker=dict(size=15)
                        ))
                    
                    time_range = df_tb["Time Budget (s)"].values
                    metric_range = df_tb[quality_metric].values
                    time_padding = (time_range.max() - time_range.min()) * 0.2 + 5 if time_range.max() > time_range.min() else 5
                    metric_padding = (metric_range.max() - metric_range.min()) * 0.2 + 0.1 if metric_range.max() > metric_range.min() else 0.2
                    fig2.update_layout(
                        title=f"{quality_metric} vs Time Budget",
                        xaxis_title="Time Budget (s)",
                        yaxis_title=quality_metric, 
                        xaxis=dict(range=[time_range.min() - time_padding, time_range.max() + time_padding]),
                        yaxis=dict(range=[metric_range.min() - metric_padding, metric_range.max() + metric_padding]),
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            
            if st.session_state.get("show_clu_modelcard"):
                clu_res = st.session_state.get("clu_results", {})
                ok_runs = [k for k, v in clu_res.items() if "error" not in v]
                if ok_runs:
                    st.subheader("📋 Clustering Model Card")
                    best_run = max(ok_runs, key=lambda k: clu_res[k].get("metrics", {}).get("Silhouette", -1))
                    best_result = clu_res[best_run]
                    
                    from model_card_generator import generate_clustering_model_card_pdf
                    pdf_bytes = generate_clustering_model_card_pdf(best_result, "Clustering Analysis")
                    
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name="clustering_model_card.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✓ Model card ready to download!")
                else:
                    st.warning("Run clustering first to generate model card.")
            if st.session_state.get("show_clu_audience_explanation"):
                clu_res = st.session_state.get("clu_results", {})
                clustering_action_space = _get_clustering_action_space()

                with st.expander("🧪 Debug: Available Clustering Recommendation Controls"):
                    st.json(clustering_action_space)
                if not clu_res:
                    st.warning("Run clustering first to generate audience explanations.")
                else:
                    st.subheader("👥 Audience Explanation Panel")
                    
                    # SECTION 1: SELECT AUDIENCE
                    st.markdown("### Step # 01: Select Your Audience Type")
                    audience_type = st.selectbox(
                        "Who are you?",
                        ["Normal User", "Domain Expert", "Data Scientist"],
                        index=["Normal User", "Domain Expert", "Data Scientist"].index(
                            st.session_state.get("clu_audience_selected_type", "Normal User")
                        ),
                        key="clu_audience_type_dropdown"
                    )
                    st.session_state["clu_audience_selected_type"] = audience_type
                    
                    audience_descriptions = {
                        "Normal User": "👤 Simple explanation in everyday language about cluster patterns.",
                        "Domain Expert": "👨‍⚕️ Domain-specific insights about what clusters represent.",
                        "Data Scientist": "🔬 Technical details about algorithms, metrics, and optimization."
                    }
                    st.info(audience_descriptions[audience_type])
                    
                    # SECTION 2: EXTRACT CLUSTERING INFO
                    st.markdown("### Step # 02: Generate Explanation")
                    clustering_info = {
                        "best_algorithm": max(clu_res.items(), key=lambda x: x[1].get("metrics", {}).get("Silhouette", 0) if "error" not in x[1] else -1)[1].get("algo") if clu_res else "Unknown",
                        "best_k": max(clu_res.items(), key=lambda x: x[1].get("metrics", {}).get("Silhouette", 0) if "error" not in x[1] else -1)[1].get("best_k") if clu_res else 0,
                        "frameworks_used": list(set([v.get("framework") for v in clu_res.values() if "error" not in v])),
                        "best_silhouette": max([v.get("metrics", {}).get("Silhouette", 0) for v in clu_res.values() if "error" not in v], default=0),
                        "avg_silhouette": sum([v.get("metrics", {}).get("Silhouette", 0) for v in clu_res.values() if "error" not in v]) / len([v for v in clu_res.values() if "error" not in v]) if any("error" not in v for v in clu_res.values()) else 0,
                        "total_runs": len([v for v in clu_res.values() if "error" not in v])
                    }
                    
                    if st.button(f"🚀 Generate Explanation for {audience_type}", key="clu_generate_explanation_btn", use_container_width=True):
                        st.session_state["clu_generating_explanation"] = True
                    
                    if st.session_state.get("clu_generating_explanation", False):
                        with st.spinner(f"🔄 Generating explanation for {audience_type}..."):
                            try:
                                import os
                                use_openai = bool(os.getenv("OPENAI_API_KEY"))
                                openai_api_key = os.getenv("OPENAI_API_KEY") if use_openai else None
                                ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
                                ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
                                # ✅ CORRECT ORDER: Extract variables FIRST
                                clu_res = st.session_state.get("clu_results", {})

                                # Extract CO2/Energy values into variables
                                all_co2_values = [float(v.get("CO2 Emission")) for v in clu_res.values() 
                                                if "error" not in v and v.get("CO2 Emission") is not None]
                                all_energy_values = [float(v.get("Energy Consumption")) for v in clu_res.values() 
                                                    if "error" not in v and v.get("Energy Consumption") is not None]

                                best_co2 = max(all_co2_values) if all_co2_values else None
                                best_energy = max(all_energy_values) if all_energy_values else None
                                all_algo_metrics = [
                                    {
                                        "algorithm": key,
                                        "co2_emission": float(v.get("CO2 Emission")) if v.get("CO2 Emission") is not None else None,
                                        "energy_consumption": float(v.get("Energy Consumption")) if v.get("Energy Consumption") is not None else None,
                                        "silhouette": float(v.get("metrics", {}).get("Silhouette", 0)) if "error" not in v else None,
                                    }
                                    for key, v in clu_res.items() if "error" not in v
                                ]
                                result = generate_clustering_audience_explanation(
                                    audience_type=audience_type,
                                    domain="General",
                                    model_info=clustering_info,
                                    features_info=[{"name": algo, "importance": 0.5} for algo in clustering_info["frameworks_used"]],
                                    metrics_info={
                                        "silhouette": clustering_info["best_silhouette"],
                                        "avg_silhouette": clustering_info["avg_silhouette"],
                                        "k": clustering_info["best_k"],
                                        "ch_index": clustering_info.get("ch_index", "N/A"),
                                        "db_index": clustering_info.get("db_index", "N/A"),
                                        "total_runs": clustering_info["total_runs"],
                                        "frameworks_used": clustering_info["frameworks_used"],
                                        "co2_emission_µg": best_co2,  # ← Add unit in key name
                                        "co2_emission_description": f"Carbon emissions (CO2) in micrograms: {best_co2} µg",  # ← Add context
                                        "energy_consumption_µWh": best_energy,  # ← Add unit in key name
                                        "energy_consumption_description": f"Energy consumption in microWatt-hours: {best_energy} µWh",  # ← Add context
                                        "sustainability_metrics_explanation": "These values measure the environmental impact of running the clustering algorithm.",  # ← Add explanation
                                        "all_algorithms_metrics": all_algo_metrics
                                    },
                                    use_openai=use_openai,
                                    openai_api_key=openai_api_key,
                                    ollama_url=ollama_url,
                                    ollama_model=ollama_model
                                )
                                                                
                                if "clu_explanation_results" not in st.session_state:
                                    st.session_state["clu_explanation_results"] = {}
                                st.session_state["clu_explanation_results"][audience_type] = result
                                st.session_state["clu_generating_explanation"] = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error generating explanation: {str(e)}")
                                st.session_state["clu_generating_explanation"] = False
                    # SECTION 3: DISPLAY EXPLANATION
                    explanation_results = st.session_state.get("clu_explanation_results", {})
                    if not explanation_results or audience_type not in explanation_results:
                        st.info("👉 Click the button above to generate an explanation.")
                    else:
                        result = explanation_results[audience_type]
                        if result.get("status") == "error":
                            st.error(f"**Error:** {result.get('explanation')}")
                        else:
                            explanation_text = result.get("explanation", "No explanation generated")
                            with st.container(border=True):
                                st.markdown(explanation_text)
                    # SECTION 4: FOLLOW-UP QUESTIONS
                    st.markdown("---")
                    st.markdown("### Ask Follow-up Questions")
                    st.caption("Ask clarifying questions, or ask how to improve your clustering results")

                    if "clu_followup_history" not in st.session_state:
                        st.session_state["clu_followup_history"] = []

                    if st.session_state.get("clu_followup_history"):
                        st.markdown("#### 💬 Conversation History")
                        for qa in st.session_state["clu_followup_history"]:
                            with st.chat_message("user"):
                                st.write(qa["question"])
                            with st.chat_message("assistant"):
                                st.write(qa["answer"])

                    clu_followup = st.text_input(
                        "Ask a follow-up question (leave blank to skip)",
                        key=f"clu_audience_followup_{st.session_state.get('clu_followup_history_len', 0)}" ,
                        placeholder="e.g., 'Why is the silhouette score low?' or 'How can I improve these clusters?'"
                    )

                    if clu_followup and st.button("Get Answer", key="clu_followup_btn"):
                        # ✅ Initialize BEFORE try block to ensure it exists
                        if "clu_followup_history" not in st.session_state:
                            st.session_state["clu_followup_history"] = []

                        current_audience = st.session_state.get("clu_audience_selected_type", "Normal User")
                        current_explanation = st.session_state.get("clu_explanation_results", {}).get(current_audience, {}).get("explanation", "")

                        if not current_explanation:
                            st.warning("⚠️ Please generate an explanation first before asking follow-up questions.")
                        else:
                            with st.spinner(f"🔄 Getting answer from {current_audience} perspective..."):
                                try:
                                    clu_followup_prompt = f"""You are responding to a follow-up question based on a previous clustering explanation.

AUDIENCE: {current_audience}

CLUSTERING RESULTS SUMMARY:
- Best algorithm: {clustering_info.get('best_algorithm')}
- Best k (number of clusters): {clustering_info.get('best_k')}
- Best silhouette score: {clustering_info.get('best_silhouette')}
- Average silhouette score: {clustering_info.get('avg_silhouette')}
- Frameworks used: {clustering_info.get('frameworks_used')}
- Total runs: {clustering_info.get('total_runs')}

PREVIOUS EXPLANATION:
{current_explanation}

USER FOLLOW-UP QUESTION:
{clu_followup}

INSTRUCTION:
Answer the follow-up question in the same tone and style as the previous explanation.
Keep the answer concise (2-4 sentences max).
Stay consistent with the audience type (simple for Normal User, domain-focused for Domain Expert, technical for Data Scientist).
If the user is asking how to improve the clustering results, give concrete, actionable suggestions
(e.g., trying a different k range, a different algorithm/framework, scaling or engineering features,
increasing the time budget or number of trials, or removing outliers) grounded in the metrics above."""

                                    use_openai = bool(os.getenv("OPENAI_API_KEY"))
                                    openai_api_key = os.getenv("OPENAI_API_KEY") if use_openai else None
                                    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
                                    ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:4b")

                                    if use_openai and openai_api_key:
                                        from openai import OpenAI
                                        client = OpenAI(api_key=openai_api_key)
                                        response = client.chat.completions.create(
                                            model="gpt-4o-mini",
                                            messages=[
                                                {"role": "user", "content": clu_followup_prompt}
                                            ],
                                            temperature=0.3,
                                            max_tokens=350
                                        )
                                        clu_followup_answer = response.choices[0].message.content.strip()
                                    else:
                                        response = requests.post(
                                            f"{ollama_url}/api/generate",
                                            json={
                                                "model": ollama_model,
                                                "prompt": clu_followup_prompt,
                                                "stream": False,
                                                "options": {"temperature": 0.3, "num_predict": 350}
                                            },
                                            timeout=600
                                        )
                                        response.raise_for_status()
                                        clu_followup_answer = (response.json().get("response") or "").strip()

                                    if clu_followup_answer:
                                        st.session_state["clu_followup_history"].append({
                                            "question": clu_followup,
                                            "answer": clu_followup_answer
                                        })
                                        st.session_state["clu_followup_history_len"] = len(st.session_state["clu_followup_history"])
                                        st.success("✅ Answer generated!")
                                        st.rerun()
                                    else:
                                        st.error("No answer received from the model.")
                                except Exception as e:
                                    st.error(f"Error generating follow-up answer: {str(e)}")
else: 
    def read_any_csv(file_obj):
        """
        Auto-detect the encoding of an uploaded CSV
        (UTF-8, UTF-16, Latin-1, cp1252…) and return a pandas DataFrame.
        """
        raw = file_obj.read()                          # bytes
        enc = chardet.detect(raw)["encoding"] or "utf-8"
        file_obj.seek(0)                               # rewind for pandas
        return pd.read_csv(BytesIO(raw), encoding=enc)
    default_hyperparams = {
    "FLAML": {
        "RF": {
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
        "XGBoost": {
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 6,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
        "LightGBM": {
            "num_leaves": 31,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "min_child_samples": 20,
        },
        "Extra Trees": {
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
        "KNN": {
            "n_neighbors": 5,
            "weights": "uniform",
        },
        "Logistic Regression": {
            "C": 1.0,
        },
    },

    "H2O": {
        "GLM": {
            "alpha": 0.5,
            "lambda_": 0.1,
        },
        "GBM": {
            "ntrees": 100,
            "learn_rate": 0.05,
            "max_depth": 6,
        },
        "Deep Learning": {
            "epochs": 10,
        },
        "Distributed RF": {
            "ntrees": 200,
            "max_depth": 6,
        },
    },

    "MLJAR": {
        "Baseline": {
            "C": 1.0,
        },
        "Decision Tree": {
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
        "RF": {
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
        "XGBoost": {
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 6,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
        "Neural Network": {
            "alpha": 0.0001,
            "max_iter": 500,
        },
        "Extra Trees": {
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
        "LightGBM": {
            "num_leaves": 31,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "min_child_samples": 20,
        },
        "SVM": {
            "kernel": "rbf",
            "C": 1.0,
        },
        "KNN": {
            "n_neighbors": 5,
            "weights": "uniform",
        },
    },
}
    # Set page configuration
    st.title("SustainaML AutoML")
    # Sidebar: Dataset upload
    st.sidebar.header("Dataset Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    # Add Recommend Button and display suggestion
    if uploaded_file and st.sidebar.button("📌 Recommend Best Combo"):
        df = read_any_csv(uploaded_file)
        st.session_state["dataset_df"] = df
        payload = {"data": df.to_json()}
        with st.spinner("Running recommended AutoML with selected frameworks and metrics..."):
            resp = requests.post(f"{BACKEND_URL}/recommend", json=payload, timeout=600)
        if resp.ok:
            recs = resp.json()
            acc = recs.get("accuracy_recommendations", {})
            co2 = recs.get("co2_recommendations", {})
            # --- Accuracy block --------------------------------------------------
            with st.expander("Recommendation based on **Accuracy + Time**", expanded=True):
                for fw, d in acc.items():
                    st.markdown(f"- **{fw}** — **{d['algorithm']}** · {d['time_budget']} s")
            # --- CO₂ block -------------------------------------------------------
            with st.expander("Recommendation based on **CO₂ Emission + Time**", expanded=True):
                for fw, d in co2.items():
                    st.markdown(f"- **{fw}** — **{d['algorithm']}** · {d['time_budget']} s")
        else:
            st.error("Recommendation failed: " + resp.json().get("message", "unknown error"))


    # Add a button to show dataset insights
    if "show_dataset_insights" not in st.session_state:
        st.session_state["show_dataset_insights"] = False
    if st.sidebar.button(" Dataset Insights"):
        st.session_state["show_dataset_insights"] = not st.session_state["show_dataset_insights"]
    # Display Dataset Insights only if the button is clicked
    if st.session_state["show_dataset_insights"] and uploaded_file:
        uploaded_file.seek(0)                # rewind because it has been read once already
        df = read_any_csv(uploaded_file)
        # Show dataset preview
        st.write("### Data Preview")
        st.dataframe(df.head())
        # Show dataset summary statistics
        summary_stats = df.describe().drop('count').rename(index={
            "25%": "Q1 (25th percentile)",
            "50%": "Median (50th percentile)",
            "75%": "Q3 (75th percentile)"
        }) 
        st.write("### Summary Statistics")
        # Create a styled table for summary statistics
        styled_stats = summary_stats.style.set_table_styles([
            # Style header cells in the table head
            {'selector': 'thead th', 'props': [('color', 'black'), ('background-color', 'white')]},
            # Style the row index cells (often rendered as th in tbody)
            {'selector': 'tbody th', 'props': [('color', 'black'), ('background-color', 'white')]},
            # Style the first data cell (if needed; for non-index first column cells)
            {'selector': 'tbody td:first-child', 'props': [('color', 'black'), ('background-color', 'white')]}
        ])
        # Render the styled table as HTML
        st.markdown(styled_stats.to_html(), unsafe_allow_html=True)
        # Show missing values
        st.write("### Missing Values")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.write(missing_values[missing_values > 0])
        else:
            st.write("No missing values in the dataset.")
        # # Get numeric columns
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        # Show correlation heatmap
        if len(numeric_columns) > 1:
            st.write("### Correlation Heatmap")
            plt.figure(figsize=(6, 4))  # Make the plot smaller
            corr = df[numeric_columns].corr()
            # Use smaller annotation text and rotate x-axis labels
            sns.heatmap(
                corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
                annot_kws={"size": 8}   # reduce annotation font size
            )
            plt.xticks(rotation=45, fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            st.pyplot(plt)
        else:
            st.write("Not enough numeric features for a correlation heatmap.")
    st.sidebar.header("AutoML Settings")
    # Sidebar: Framework and algorithm selection
    with st.sidebar.expander("Select Frameworks and Algorithms"):
        frameworks = ["FLAML", "H2O", "MLJAR"]
        selected_frameworks = st.multiselect("Choose AutoML Frameworks:", frameworks)
        algorithm_selection = {}
        # Algorithm Selection UI
        for framework in selected_frameworks:
            st.subheader(f"{framework} Algorithms")
            algorithm_selection[framework] = {}

            for algo in default_hyperparams[framework].keys():
                is_recommended = (
                    "recommended_algo" in st.session_state
                    and "recommended_framework" in st.session_state
                    and st.session_state.recommended_algo == algo
                    and st.session_state.recommended_framework == framework
                )
                algorithm_selection[framework][algo] = st.checkbox(
                    algo,
                    value=is_recommended,
                    key=f"{framework}_{algo}"
                )
    # Add a button to modify hyperparameters
    if st.sidebar.button("Modify Hyperparameters"):
        if not selected_frameworks or not any(any(val for val in algo.values()) for algo in algorithm_selection.values()):
            st.sidebar.warning("Please select at least one framework and algorithm.")
        else:
            st.session_state["show_dataset_insights"] = False
            st.session_state["show_hyperparam_ui"] = True
    # Time Budget Dropdown in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏱️ Choose Time Budget")
    time_options = {
        "10 seconds": 10,
        "30 seconds": 30,
        "60 seconds": 60,
        "120 seconds": 120,
    }
    default_time = "10 seconds"
                    # Initialize only once
    if "selected_time_budget" not in st.session_state:
        if "recommended_time" in st.session_state:
            reverse_map = {v: k for k, v in time_options.items()}
            selected_label = reverse_map.get(st.session_state.recommended_time, default_time)
            st.session_state.selected_time_budget = selected_label
            st.session_state.time_budget = st.session_state.recommended_time
        else:
            st.session_state.selected_time_budget = default_time
            st.session_state.time_budget = time_options[default_time]
    # Dropdown selector
    selected_label = st.sidebar.selectbox(
        "Select a time budget for AutoML:",
        list(time_options.keys()),
        index=list(time_options.keys()).index(st.session_state.selected_time_budget)
    )
    st.session_state.selected_time_budget = selected_label
    st.session_state.time_budget = time_options[selected_label]  
        # Ensure "Dataset Insights" is hidden when modifying hyperparameters
    if st.session_state.get("show_hyperparam_ui", False):
        st.session_state["show_dataset_insights"] = False
    # Show Hyperparameter Modification UI
    if "show_hyperparam_ui" in st.session_state and st.session_state["show_hyperparam_ui"]:
        st.subheader("Modify Hyperparameters")
        modified_hyperparams = {}
        for framework in selected_frameworks:
            st.write(f"**{framework} Algorithms**")
            modified_hyperparams[framework] = {}
            for algo, selected in algorithm_selection[framework].items():
                if selected:
                    st.write(f"**{algo} Hyperparameters:**")
                    hyperparams = default_hyperparams[framework].get(algo, {})
                    modified_hyperparams[framework][algo] = {}
                    for param, default_value in hyperparams.items():
                        unique_key = f"{framework}_{algo}_{param}"  # Unique key for each UI element
                        if isinstance(default_value, (int, float)):  # Handle numeric parameters
                            modified_hyperparams[framework][algo][param] = st.number_input(
                                f"{algo} - {param}",
                                value=default_value,
                                key=unique_key,  # Assign unique key
                            )
                        elif isinstance(default_value, str):  # Handle string parameters
                            options = {"weights": ["uniform", "distance"], "kernel": ["linear", "rbf", "poly", "sigmoid"]}
                            modified_hyperparams[framework][algo][param] = st.selectbox(
                                f"{algo} - {param}", #options=["uniform", "distance"],
                                options=options.get(param, [default_value]),
                                key=unique_key,  # Assign unique key
                            )
                        else:
                            st.warning(f"Skipping unsupported parameter: {param} (type: {type(default_value)})")
        if st.button("Confirm Hyperparameters"):
            st.session_state["modified_hyperparams"] = modified_hyperparams
            st.session_state["show_hyperparam_ui"] = False
            st.success("Hyperparameters updated! You can now run AutoML.")
    # Persistent state for DataFrame and valid metrics
    if "df_metrics" not in st.session_state:
        st.session_state.df_metrics = None
    if "valid_metrics" not in st.session_state:
        st.session_state.valid_metrics = []
    if "show_feature_importance" not in st.session_state:
        st.session_state["show_feature_importance"] = False
    if "show_pipeline_analysis" not in st.session_state:
        st.session_state["show_pipeline_analysis"] = False
    if "show_interpretability_panel" not in st.session_state:
        st.session_state["show_interpretability_panel"] = False
    if "iem_cache" not in st.session_state:
        st.session_state["iem_cache"] = {}
    if "automl_results" not in st.session_state:
        st.session_state["automl_results"] = {}
        # Sidebar button to toggle feature importance
    if 'show_dataset_insights' in st.session_state:
        st.session_state['show_dataset_insights'] = False
    if 'show_hyperparam_ui' in st.session_state:
        st.session_state['show_hyperparam_ui'] = False
    # Run AutoML Button / step 3
    if st.sidebar.button("Run AutoML"):
        if uploaded_file:
            uploaded_file.seek(0)                # rewind because it has been read once already
            df = read_any_csv(uploaded_file)
            problems = _validate_dataset(df, "classification")
            if problems:
                st.warning("• " + "\n• ".join(problems))
            st.info("Running AutoML with selected frameworks and metrics...")

            # Counterfactual addition - Extract and save training data BEFORE sending to backend
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            
            # Prepare training data (same preprocessing as backend)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            # Scale numeric columns
            numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            
            # Encode categorical columns
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                X[col] = LabelEncoder().fit_transform(X[col])
            
            # Split data (same as backend)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
            )
            
            # Save to session state
            st.session_state["X_train"] = X_train
            st.session_state["y_train"] = y_train
            st.session_state["X_val"] = X_val
            st.session_state["y_val"] = y_val
            # Held-out evaluation aliases. Fairness must never fall back to X_train.
            st.session_state["X_test"] = X_val.copy()
            st.session_state["y_test"] = y_val.copy()

            # Send data to backend
            dataset_json = df.to_json()
            st.session_state["last_uploaded_dataset_json"] = dataset_json
            payload = {
                "frameworks": selected_frameworks,
                "algorithms": algorithm_selection,
                "hyperparams": st.session_state.get("modified_hyperparams", {}),
                "metric": "Accuracy",
                "data": dataset_json,
                "time_budget": st.session_state.get("time_budget", 30),
                
            }   
            try:
                with st.spinner("Running AutoML..."):
                    r = requests.post(f"{BACKEND_URL}/run_automl", json=payload, timeout=600)
                r.raise_for_status()  # raise for 4xx / 5xx
                data = r.json()       # safe to parse JSON now
            except requests.exceptions.JSONDecodeError:
                # Backend likely returned HTML (e.g., Flask 500 page); show it so we can debug
                st.error(f"Backend returned non-JSON (status {r.status_code}).\n\n{r.text[:2000]}")
                st.stop()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

            # ----- success path (same logic you had) -----
            results = data.get("results", {})
            st.session_state["automl_results"] = results  # Store results for feature importance
            st.session_state["classification_recommendation"] = None
            st.session_state["classification_recommendation_context"] = None
            st.session_state["classification_validation_result"] = None
            st.session_state["classification_validation_baseline"] = None
            st.session_state["classification_validation_error"] = None
            # Store comparison by time budget
            time_label = f"{st.session_state.selected_time_budget}"
            comparison_entry = {
                "time_budget": time_label,
                "results": results
            }
            if "time_budget_comparisons" not in st.session_state:
                st.session_state["time_budget_comparisons"] = []
            st.session_state["time_budget_comparisons"].append(comparison_entry)

            if not results:
                st.error("No results were generated. Please check the backend logs for details.")
            else:
                metrics_data = []
                for algo, metrics in results.items():
                    if "error" in metrics:
                        st.warning(f"Algorithm {algo} failed: {metrics['error']}")
                    else:
                        row = {"Algorithm": algo}
                        # Ensure CO2 emissions are rounded (keeping your precision)
                        if "CO2 Emission" in metrics:
                            metrics["CO2 Emission"] = round(metrics["CO2 Emission"], 10)
                        row.update(metrics)
                        metrics_data.append(row)
                if metrics_data:

                    st.session_state.df_metrics = (
                        pd.DataFrame(metrics_data)
                        .drop(
                            columns=["cost_micro_cents"],
                            errors="ignore"
                        )
                    )

                    st.session_state.valid_metrics = [
                        col
                        for col in st.session_state.df_metrics.columns[1:]
                        if (
                            col != "cost_micro_cents"
                            and pd.api.types.is_numeric_dtype(
                                st.session_state.df_metrics[col]
                            )
                        )
                    ]
                    st.success("AutoML run completed successfully!")
                    # NOTE: the fairness panel does NOT use this surrogate model.
                    # The backend now keeps the actual AutoML winner and fairness
                    # is calculated there on the held-out validation split.
                    try:
                        from sklearn.ensemble import RandomForestClassifier
                        best_algo = max(
                            results.keys(),
                            key=lambda x: results[x].get('Accuracy', 0)
                            if isinstance(results[x], dict) else 0
                        )
                        surrogate_model = RandomForestClassifier(random_state=42, n_estimators=100)
                        surrogate_model.fit(st.session_state["X_train"], st.session_state["y_train"])
                        st.session_state["best_model"] = surrogate_model
                        st.session_state["best_model_key"] = best_algo
                    except Exception as e:
                        print(f"Error training surrogate model for auxiliary panels: {e}")
        #  Ensure session state is initialized
        if "show_feature_importance" not in st.session_state:
            st.session_state["show_feature_importance"] = False
        if "automl_results" not in st.session_state:
            st.session_state["automl_results"] = {}
    #  Sidebar: Feature Importance Button (Always Visible)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Tool")
    # Sidebar: Show AutoML Results Button
    if st.sidebar.button("Model Leaderboard"):
        st.session_state["show_automl_results"] = True
        st.session_state["show_feature_importance"] = False
        st.session_state["show_pipeline_analysis"] = False
        st.session_state["show_comparison"] = False
        st.session_state["show_audience_mode"] = False
        st.session_state["show_model_card"] = False
        if "automl_results" in st.session_state and st.session_state["automl_results"]:
            st.session_state["show_automl_results"] = True
        else:
            st.warning("Please run AutoML first.")
            st.rerun()
    if st.sidebar.button("Feature Importance"):
        if not st.session_state.get("automl_results"):
            st.sidebar.warning("⚠️ Please run AutoML first before viewing feature importance.")
        else:
            st.session_state["show_feature_importance"] = True
            st.session_state["show_automl_results"] = False
            st.session_state["show_pipeline_analysis"] = False  # Ensure only one view is active
            st.session_state["show_model_card"] = False
            st.rerun()  #  Force UI refresh to apply changes immediately
    #  Pipeline Analysis Toggle
    if st.sidebar.button("Hyperparameter Analysis"):
        st.session_state["show_hyperimpact_analysis"] = True
        st.session_state["show_pipeline_analysis"] = False
        st.session_state["show_feature_importance"] = False
        st.session_state["show_automl_results"] = False
        st.session_state["show_model_card"] = False
        st.rerun()
    if st.sidebar.button("Time Budgets Comparison"):
        st.session_state["show_comparison"] = True
        st.session_state["show_feature_importance"] = False
        st.session_state["show_pipeline_analysis"] = False
        st.session_state["show_automl_results"] = False
        st.session_state["show_model_card"] = False
    if st.sidebar.button("Process Overview"):
        if not st.session_state.get("automl_results"):
            st.sidebar.warning("⚠️ Please run AutoML first before viewing pipeline analysis.")
        else:
            st.session_state["show_pipeline_analysis"] = True
            st.session_state["show_automl_results"] = False
            st.session_state["show_feature_importance"] = False  # Ensure only one view is active
            st.session_state["show_model_card"] = False
            st.rerun()
    if st.sidebar.button("Interpretability Panel"):
        if not st.session_state.get("automl_results"):
            st.sidebar.warning("⚠️ Please run AutoML first before viewing interpretability metrics.")
        else:
            st.session_state["show_interpretability_panel"] = True
            st.session_state["show_pipeline_analysis"] = False
            st.session_state["show_feature_importance"] = False
            st.session_state["show_automl_results"] = False
            st.session_state["show_comparison"] = False
            st.session_state["show_model_card"] = False
            st.rerun()
    if st.sidebar.button("👥 Audience Explanation Panel"):
        if not st.session_state.get("automl_results"):
            st.sidebar.warning("⚠️ Please run AutoML first before viewing audience explanations.")
        else:
            st.session_state["show_audience_mode"] = True
            st.session_state["show_interpretability_panel"] = False
            st.session_state["show_pipeline_analysis"] = False
            st.session_state["show_feature_importance"] = False
            st.session_state["show_automl_results"] = False
            st.session_state["show_comparison"] = False
            st.session_state["show_model_card"] = False
            st.rerun()
    if st.sidebar.button("♻️ Sustainability-Accuracy Trade-off"):
        if not st.session_state.get("automl_results"):
            st.sidebar.warning("⚠️ Please run AutoML first.")
        else:
            st.session_state["show_tradeoff_panel"] = True
            st.session_state["show_audience_mode"] = False
            st.session_state["show_interpretability_panel"] = False
            st.session_state["show_pipeline_analysis"] = False
            st.session_state["show_feature_importance"] = False
            st.session_state["show_automl_results"] = False
            st.session_state["show_comparison"] = False
            st.session_state["show_model_card"] = False
            st.rerun()
    if st.sidebar.button("🔮 What-If Counterfactual"):
        if not st.session_state.get("automl_results"):
            st.sidebar.warning("⚠️ Please run AutoML first.")
        else:
            st.session_state["show_counterfactual_panel"] = True
            st.session_state["show_audience_mode"] = False
            st.session_state["show_interpretability_panel"] = False
            st.session_state["show_tradeoff_panel"] = False
            st.session_state["show_pipeline_analysis"] = False
            st.session_state["show_feature_importance"] = False
            st.session_state["show_automl_results"] = False
            st.session_state["show_comparison"] = False
            st.session_state["show_model_card"] = False
            st.rerun()
    if st.sidebar.button("⚖️ Fairness & Bias Detection"):
        has_results = (
            bool(st.session_state.get("clu_results"))
            if task.startswith("Clustering")
            else bool(st.session_state.get("automl_results"))
        )
        if not has_results:
            st.sidebar.warning("⚠️ Please run AutoML first.")
        else:
            st.session_state["show_fairness_panel"] = True
            st.session_state["show_audience_mode"] = False
            st.session_state["show_interpretability_panel"] = False
            st.session_state["show_tradeoff_panel"] = False
            st.session_state["show_counterfactual_panel"] = False
            st.session_state["show_pipeline_analysis"] = False
            st.session_state["show_feature_importance"] = False
            st.session_state["show_automl_results"] = False
            st.session_state["show_comparison"] = False
            st.session_state["show_model_card"] = False
            st.rerun()
    if st.sidebar.button("📋 Download Model Card (PDF)"):
        if not st.session_state.get("automl_results"):
            st.sidebar.warning("⚠️ Please run AutoML first.")
        else:
            st.session_state["show_model_card"] = True
            st.session_state["show_fairness_panel"] = False
            st.session_state["show_audience_mode"] = False
            st.session_state["show_interpretability_panel"] = False
            st.session_state["show_tradeoff_panel"] = False
            st.session_state["show_counterfactual_panel"] = False
            st.session_state["show_pipeline_analysis"] = False
            st.session_state["show_feature_importance"] = False
            st.session_state["show_automl_results"] = False
            st.session_state["show_comparison"] = False


    color_map = {
        "FLAML_RF": "blue",
        "FLAML_XGBoost": "red",
        "FLAML_LightGBM": "green",
        "FLAML_Extra Trees": "yellow",
        "FLAML_KNN": "purple",
        "FLAML_Logistic Regression": "orange",
        # "H2O_Naive Bayes": "gray",
        "H2O_GBM": "black",
        "H2O_GLM": "pink",
        "H2O_Distributed RF": "cyan",
        "H2O_Deep Learning": "magenta",
        "MLJAR_Baseline": "lime",
        "MLJAR_Decision Tree": "olive",
        "MLJAR_RF": "teal",
        "MLJAR_XGBoost": "maroon",
        "MLJAR_Neural Network": "navy",
        "MLJAR_Extra Trees": "silver",
        "MLJAR_LightGBM": "gold",
        "MLJAR_SVM": "beige",
        "MLJAR_KNN": "brown",
    }
    # Updated color_map to include both time budget and algorithm combinations
    color_map_time_budget = {
        "10 seconds_FLAML_RF": "blue",
        "30 seconds_FLAML_RF": "blue",
        "60 seconds_FLAML_RF": "blue",
        "120 seconds_FLAML_RF": "blue",
        "10 seconds_FLAML_XGBoost": "red",
        "30 seconds_FLAML_XGBoost": "red",
        "60 seconds_FLAML_XGBoost": "red",
        "120 seconds_FLAML_XGBoost": "red",
        "10 seconds_FLAML_LightGBM": "green",
        "30 seconds_FLAML_LightGBM": "green",
        "60 seconds_FLAML_LightGBM": "green",
        "120 seconds_FLAML_LightGBM": "green",
        "10 seconds_FLAML_Extra Trees": "yellow",
        "30 seconds_FLAML_Extra Trees": "yellow",
        "60 seconds_FLAML_Extra Trees": "yellow",
        "120 seconds_FLAML_Extra Trees": "yellow",
        "10 seconds_FLAML_KNN": "purple",
        "30 seconds_FLAML_KNN": "purple",
        "60 seconds_FLAML_KNN": "purple",
        "120 seconds_FLAML_KNN": "purple",
        "10 seconds_FLAML_Logistic Regression": "orange",
        "30 seconds_FLAML_Logistic Regression": "orange",
        "60 seconds_FLAML_Logistic Regression": "orange",
        "120 seconds_FLAML_Logistic Regression": "orange",
        "10 seconds_H2O_GBM": "black",
        "30 seconds_H2O_GBM": "black",
        "60 seconds_H2O_GBM": "black",
        "120 seconds_H2O_GBM": "black",
        "10 seconds_H2O_GLM": "pink",
        "30 seconds_H2O_GLM": "pink",
        "60 seconds_H2O_GLM": "pink",
        "120 seconds_H2O_GLM": "pink",
        "10 seconds_H2O_Distributed RF": "cyan",
        "30 seconds_H2O_Distributed RF": "cyan",
        "60 seconds_H2O_Distributed RF": "cyan",
        "120 seconds_H2O_Distributed RF": "cyan",
        "10 seconds_H2O_Deep Learning": "magenta",
        "30 seconds_H2O_Deep Learning": "magenta",
        "60 seconds_H2O_Deep Learning": "magenta",
        "120 seconds_H2O_Deep Learning": "magenta",
        "10 seconds_MLJAR_Baseline": "lime",
        "10 seconds_MLJAR_Decision Tree": "olive",
        "10 seconds_MLJAR_RF": "teal",
        "10 seconds_MLJAR_XGBoost": "maroon",
        "10 seconds_MLJAR_Neural Network": "navy",
        "10 seconds_MLJAR_Extra Trees": "silver",
        "10 seconds_MLJAR_LightGBM": "gold",
        "10 seconds_MLJAR_SVM": "beige",
        "10 seconds_MLJAR_KNN": "brown",
        "30 seconds_MLJAR_Baseline": "lime",
        "30 seconds_MLJAR_Decision Tree": "olive",
        "30 seconds_MLJAR_RF": "teal",
        "30 seconds_MLJAR_XGBoost": "maroon",
        "30 seconds_MLJAR_Neural Network": "navy",
        "30 seconds_MLJAR_Extra Trees": "silver",
        "30 seconds_MLJAR_LightGBM": "gold",
        "30 seconds_MLJAR_SVM": "beige",
        "30 seconds_MLJAR_KNN": "brown",
        "60 seconds_MLJAR_Baseline": "lime",
        "60 seconds_MLJAR_Decision Tree": "olive",
        "60 seconds_MLJAR_RF": "teal",
        "60 seconds_MLJAR_XGBoost": "maroon",
        "60 seconds_MLJAR_Neural Network": "navy",
        "60 seconds_MLJAR_Extra Trees": "silver",
        "60 seconds_MLJAR_LightGBM": "gold",
        "60 seconds_MLJAR_SVM": "beige",
        "60 seconds_MLJAR_KNN": "brown",
        "120 seconds_MLJAR_Baseline": "lime",
        "120 seconds_MLJAR_Decision Tree": "olive",
        "120 seconds_MLJAR_RF": "teal",
        "120 seconds_MLJAR_XGBoost": "maroon",
        "120 seconds_MLJAR_Neural Network": "navy",
        "120 seconds_MLJAR_Extra Trees": "silver",
        "120 seconds_MLJAR_LightGBM": "gold",
        "120 seconds_MLJAR_SVM": "beige",
        "120 seconds_MLJAR_KNN": "brown",   
    }
    #  Always initialize df_metrics at the start to prevent NameError
    df_metrics = st.session_state.df_metrics if "df_metrics" in st.session_state else None
    if st.session_state.get("show_automl_results", False):
        valid_metrics = st.session_state.valid_metrics if st.session_state.valid_metrics else ["No Metrics Available"]
        st.subheader("Algorithm Metrics")
        if df_metrics is not None and not df_metrics.empty:
            available_frameworks = list(set(algo.split("_")[0] for algo in df_metrics["Algorithm"]))
            selected_frameworks = st.multiselect(
                "Select Framework(s) to Display:", available_frameworks, default=available_frameworks
            )
            max_algorithms = len(df_metrics)
            top_n = st.number_input(
                f"Select Number of Top Algorithms (Max: {max_algorithms}):",
                min_value=1, max_value=max_algorithms, value=max_algorithms, step=1
            )
            df_filtered = df_metrics[df_metrics["Algorithm"].str.startswith(tuple(selected_frameworks))]
            df_filtered = df_filtered.sort_values(by="Accuracy", ascending=False).head(top_n)
            # Combine framework and algorithm name to create a unique identifier
            df_filtered["Framework"] = df_filtered["Algorithm"].apply(lambda x: x.split("_")[0])
            # Add a new "Color" column based on the algorithm
            df_filtered["Color"] = df_filtered["Algorithm"].apply(lambda x: color_map.get(x, "white"))
            selected_metric = st.selectbox(
                "Select which single metric to display:",
                ["Accuracy", "F1 Score", "CO2 Emission"]
            )
            columns_to_keep = ["Color", "Algorithm", selected_metric]  # Move Color to the left
            df_filtered = df_filtered[columns_to_keep]
            # Function to color the 'Color' column without displaying the text
            def color_cells(val):
                color = val
                return f"background-color: {color}; color: {color}; width: 10px;"  # Ensure the text is hidden and set width
            # Apply the color to the Color column using style
            styled_df = df_filtered.style.map(color_cells, subset=["Color"])
            st.markdown("""
                <style>
                    .dataframe tbody td:nth-child(1) {
                        width: 30px !important; /* Adjust width of Color column */
                        text-align: center;
                    }
                    .dataframe thead th:nth-child(1) {
                        width: 30px !important; /* Adjust width of Color column header */
                        text-align: center;
                    }
                    .dataframe tbody td {
                        padding: 8px;about:blank#blocked
                    }
                </style>
            """, unsafe_allow_html=True)
            # Display the styled table using st.dataframe with index=False to hide the index
            st.dataframe(styled_df, use_container_width=False, hide_index=True)
        st.subheader("Algorithm Performance")
        if df_metrics is not None and not df_metrics.empty:
            available_frameworks = list(set(algo.split("_")[0] for algo in df_metrics["Algorithm"]))
            selected_frameworks = st.multiselect(
                "Select Framework(s) to Display:", available_frameworks, default=available_frameworks,
                key="framework_selection_performance"
            )
            metric_to_plot = st.selectbox("Select Metric to Visualize", valid_metrics, index=0, key="bar_metric")
            max_algorithms = len(df_metrics)
            top_n = st.number_input(
                f"Select Number of Top Algorithms (Max: {max_algorithms}):",
                min_value=1, max_value=max_algorithms, value=max_algorithms, step=1, key="top_n_performance"
            )
            df_filtered = df_metrics[df_metrics["Algorithm"].str.startswith(tuple(selected_frameworks))]
            df_filtered = df_filtered.sort_values(by=metric_to_plot, ascending=False).head(top_n)
            df_filtered["Framework"] = df_filtered["Algorithm"].apply(lambda x: x.split("_")[0])
            bar_fig = px.bar(
                df_filtered,
                x="Algorithm",
                y=metric_to_plot,
                color="Framework",
                barmode="group",
                title=f"Algorithm Comparison by {metric_to_plot}",
            )
            bar_fig.update_layout(
                        xaxis=dict(
                            title=dict(font=dict(size=14, color="black", family="Arial Black")),
                            tickfont=dict(size=15, color="black", family="Arial")
                        ),
                        yaxis=dict(
                            title=dict(font=dict(size=14, color="black", family="Arial Black")),
                            tickfont=dict(size=15, color="black", family="Times New Roman")
                        ),
                        legend=dict(
                            font=dict(
                                size=13,  # Set the font size for the legend
                                color="black"  # Set the color of the legend text to black
                            ),
                        ),
                    )
            st.plotly_chart(bar_fig)
        # Performance Trends Section (Scatter Plot)
        st.subheader("Performance Trends")
        if st.session_state.get("show_automl_results", False):
            valid_metrics = st.session_state.valid_metrics if st.session_state.valid_metrics else ["No Metrics Available"]

            if valid_metrics and valid_metrics != ["No Metrics Available"]:
                selected_frameworks_scatter = st.multiselect(
                    "Select Framework(s) for Scatter Plot:", available_frameworks, default=available_frameworks,
                    key="scatter_framework"
                )
                x_axis = st.selectbox("Select X-Axis", valid_metrics, index=0, key="scatter_x")
                y_axis = st.selectbox("Select Y-Axis", valid_metrics, index=1, key="scatter_y")

                df_scatter_filtered = df_metrics[df_metrics["Algorithm"].str.startswith(tuple(selected_frameworks_scatter))]
                if "Framework" not in df_scatter_filtered.columns:
                    df_scatter_filtered["Framework"] = df_scatter_filtered["Algorithm"].apply(lambda x: x.split("_")[0])
                if not df_scatter_filtered.empty:
                    bubble_metric = st.selectbox(
                        "Select Metric for Bubble Size:", valid_metrics, index=valid_metrics.index("CO2 Emission"),
                        key="bubble_size"
                    )
                    # Map Algorithm to color using the color_map
                    df_scatter_filtered["Color"] = df_scatter_filtered["Algorithm"].apply(lambda x: color_map.get(x, "white"))
                    # Create the scatter plot with unique color for each algorithm
                    scatter_fig = px.scatter(
                        df_scatter_filtered,
                        x=x_axis,
                        y=y_axis,
                        color="Algorithm",  # Use the Algorithm column for color
                        color_discrete_map=color_map,  # Apply custom color map
                        size=bubble_metric,
                        hover_name="Algorithm",
                        hover_data=["Accuracy", "F1 Score", "CO2 Emission"],
                        title=f"{y_axis} vs {x_axis} by Algorithm",
                        size_max=12,
                        template="plotly_white"  # Use clean background
                    )

                    # Remove color scale and ensure that the color map is used correctly
                    scatter_fig.update_layout(
                        coloraxis_showscale=False,  # Disable any automatic color scale
                        legend_title="Framework_Algorithm",
                        legend=dict(
                            itemsizing="constant",
                            font=dict(size=13, color="black"),
                            traceorder="normal"
                        ),
                    )
                    # Customizing plot layout
                    scatter_fig.update_layout(
                        xaxis=dict(
                            title=dict(font=dict(size=14, color="black", family="Arial Black")),
                            tickfont=dict(size=15, color="black", family="Arial")
                        ),
                        yaxis=dict(
                            title=dict(font=dict(size=14, color="black", family="Arial Black")),
                            tickfont=dict(size=15, color="black", family="Times New Roman")
                        ),
                    )
                    # Display the performance trend scatter plot
                    st.plotly_chart(scatter_fig)
                else:
                    st.warning("No data available for the selected frameworks.")
    # If no data is available, show a warning
    if df_metrics is None or (isinstance(df_metrics, pd.DataFrame) and df_metrics.empty):
        st.warning("No data available for visualization. Run AutoML first.")
    elif st.session_state.get("show_pipeline_analysis", False):
        st.subheader("Pipeline Analysis")
        # Define pipeline steps per framework-algorithm combo
        pipeline_steps = {
            "FLAML_Random Forest": [
                "Input", "Random Forest Initialization", "Hyperparameter Initialization", "Model Fit",
                "Random Subsampling", "Majority Voting", "Model Evaluation", "Prediction"
            ],
            "FLAML_Extra Trees": [
                "Input", "Extra Trees Initialization", "Hyperparameter Initialization", "Model Fit",
                "Random Subsampling", "Tree Construction", "Majority Voting", "Model Evaluation", "Prediction"
            ],
            "FLAML_Logistic Regression": [
                "Input", "Logistic Regression Function: Sigmoid Function", "Hyperparameter Initialization",
                "Model Fit", "Optimization and Regularization", "Prediction"
            ],
            "FLAML_XGBoost": [
                "Input", "XGBoost Initialization", "Hyperparameter Initialization", "Model Fit",
                "Optimization", "Regularization", "Model Evaluation", "Prediction"
            ],
            "FLAML_CatBoost": [
                "Input", "CatBoost Initialization", "Hyperparameter Initialization", "Model Fit",
                "Optimization", "Regularization", "Model Evaluation", "Prediction"
            ],
            "FLAML_LightGBM": [
                "Input", "LightGBM Initialization", "Hyperparameter Initialization", "Model Fit",
                "Optimization", "Regularization", "Model Evaluation", "Prediction"
            ],
            "FLAML_K-Nearest Neighbors": [
                "Input", "KNN Initialization", "Model Fit", "Distance Calculation", "Finding K Neighbors",
                "Majority Voting", "Model Evaluation", "Prediction"
            ],

            "H2O_GLM": [
                "Input", "Logistic Regression Function: Sigmoid Function", "Hyperparameter Initialization",
                "Model Fit", "Optimization and Regularization", "Prediction"
            ],
            "H2O_GBM": [
                "Input", "LightGBM Initialization", "Hyperparameter Initialization", "Model Fit",
                "Optimization", "Regularization", "Model Evaluation", "Prediction"
            ],
            "H2O_Naive Bayes": [
                "Input", "Gaussian NB Initialization", "Hyperparameter Initialization", "Model Fit",
                "Parameter Estimation", "Model Evaluation", "Prediction"
            ],
            "H2O_Distributed Random Forest": [
                "Input", "Random Forest Initialization", "Hyperparameter Initialization", "Model Fit",
                "Random Subsampling", "Majority Voting", "Model Evaluation", "Prediction"
            ],
            "H2O_XGBoost": [
                "Input", "XGBoost Initialization", "Hyperparameter Initialization", "Model Fit",
                "Optimization", "Regularization", "Model Evaluation", "Prediction"
            ],

            "MLJAR_Baseline": [
                "Input", "Logistic Regression Function: Sigmoid Function", "Hyperparameter Initialization",
                "Model Fit", "Optimization and Regularization", "Prediction"
            ],
            "MLJAR_Decision Tree": [
                "Input", "Extra Trees Initialization", "Hyperparameter Initialization", "Model Fit",
                "Random Subsampling", "Tree Construction", "Majority Voting", "Model Evaluation", "Prediction"
            ],
            "MLJAR_Random Forest": [
                "Input", "Random Forest Initialization", "Hyperparameter Initialization", "Model Fit",
                "Random Subsampling", "Majority Voting", "Model Evaluation", "Prediction"
            ],
            "MLJAR_XGBoost": [
                "Input", "XGBoost Initialization", "Hyperparameter Initialization", "Model Fit",
                "Optimization", "Regularization", "Model Evaluation", "Prediction"
            ],
            "MLJAR_Neural Network": [
                "Input", "SVC Initialization", "Hyperparameter Initialization", "Model Fit",
                "Kernel Transformation", "Optimization", "Regularization", "Model Evaluation", "Prediction"
            ],
            "MLJAR_Extra Trees": [
                "Input", "Extra Trees Initialization", "Hyperparameter Initialization", "Model Fit",
                "Random Subsampling", "Tree Construction", "Majority Voting", "Model Evaluation", "Prediction"
            ],
            "MLJAR_LightGBM": [
                "Input", "LightGBM Initialization", "Hyperparameter Initialization", "Model Fit",
                "Optimization", "Regularization", "Model Evaluation", "Prediction"
            ],
            "MLJAR_Support Vector Machines": [
                "Input", "SVC Initialization", "Hyperparameter Initialization", "Model Fit",
                "Kernel Transformation", "Optimization", "Regularization", "Model Evaluation", "Prediction"
            ],
            "MLJAR_K-Nearest Neighbors": [
                "Input", "KNN Initialization", "Model Fit", "Distance Calculation", "Finding K Neighbors",
                "Majority Voting", "Model Evaluation", "Prediction"
            ]
        }
        selected_algos = list(st.session_state["automl_results"].keys())
        for algo_key in selected_algos:
            st.markdown(f"### {algo_key}")
            steps = pipeline_steps.get(algo_key)

            if steps:
                # Inline rendering
                flow_str = ""
                for i, step in enumerate(steps):
                    if "Hyperparameter Initialization" in step:
                        # Add an expandable tooltip-style container
                        with st.expander(f"🔧 {step} (click to view hyperparameters)"):
                            hyperparams = st.session_state["automl_results"].get(algo_key, {}).get("hyperparameters", {})
                            if hyperparams:
                                st.json(hyperparams)
                            else:
                                st.write("No hyperparameters found.")
                        flow_str += f"`{step}`"
                    else:
                        flow_str += f"`{step}`"
                    if i < len(steps) - 1:
                        flow_str += " ➡️ "
                st.markdown(flow_str)
            else:
                st.warning(f"No pipeline steps defined for {algo_key}")
    elif st.session_state.get("show_interpretability_panel", False):
        def _bucket_score(x: float) -> str:
            # Simple, human-friendly buckets for [0,1] dimension scores
            if x >= 0.85:
                return "very strong"
            if x >= 0.70:
                return "strong"
            if x >= 0.55:
                return "moderate"
            if x >= 0.40:
                return "weak"
            return "very weak"
        def _interpret_iem_results(df_iem: pd.DataFrame, weights_now: dict, meta: dict, selected_pipeline: str) -> dict:
            """
            Turn numeric interpretability results into plain-language insights.
            Returns a dict with:
            - headline (str)
            - ranking_notes (list[str])
            - technique_cards (list[dict])  each with {title, bullets}
            """
            if df_iem is None or df_iem.empty:
                return {
                    "headline": "No interpretability results available yet.",
                    "ranking_notes": [],
                    "technique_cards": [],
                }
            dims = ["Fidelity", "Stability", "Compactness", "Correctness", "Simplicity", "Completeness"]
            # ---- Headline: top technique + why (top 2 dimensions) ----
            top = df_iem.iloc[0]
            top_dims_sorted = sorted([(d, float(top[d])) for d in dims], key=lambda t: t[1], reverse=True)
            top_dim_1, top_dim_2 = top_dims_sorted[0], top_dims_sorted[1]

            headline = (
                f"For **{selected_pipeline}**, the top-ranked explanation technique is **{top['Technique']}** "
                f"(IEM={float(top['IEM']):.3f}). "
                f"It is strongest on **{top_dim_1[0]}** ({top_dim_1[1]:.2f}) and **{top_dim_2[0]}** ({top_dim_2[1]:.2f})."
            )
            # ---- Notes about weights (what they do and do not do) ----
            wsum = sum(float(v) for v in weights_now.values()) if weights_now else 1.0
            weights_norm = {k: (float(v) / wsum if wsum > 0 else 0.0) for k, v in (weights_now or {}).items()}
            # Identify “most emphasized” dimensions
            weight_pairs = []
            for d in dims:
                key = d.lower()
                if key in weights_norm:
                    weight_pairs.append((d, weights_norm[key]))
            weight_pairs.sort(key=lambda t: t[1], reverse=True)
            emphasized = [f"{d} ({w:.0%})" for d, w in weight_pairs[:2]] if weight_pairs else []
            ranking_notes = []
            if emphasized:
                ranking_notes.append(
                    f"Your current IEM ranking is influenced most by: **{emphasized[0]}**"
                    + (f" and **{emphasized[1]}**." if len(emphasized) > 1 else ".")
                )
            ranking_notes.append(
                "The **dimension scores** (Fidelity, Stability, etc.) come from evaluation computations. "
                "Changing weights does **not** change those scores. It only changes how they are combined into IEM."
            )
            # ---- Per-technique cards ----
            technique_cards = []
            for _, row in df_iem.iterrows():
                tname = str(row["Technique"])
                iem = float(row["IEM"])
                dim_vals = {d: float(row[d]) for d in dims}

                # strengths/weaknesses
                strongest = sorted(dim_vals.items(), key=lambda t: t[1], reverse=True)[:2]
                weakest = sorted(dim_vals.items(), key=lambda t: t[1])[:2]

                bullets = []
                bullets.append(f"**IEM:** {iem:.3f} (higher means better under your chosen weights).")
                bullets.append(
                    f"**Strongest:** {strongest[0][0]} ({strongest[0][1]:.2f}, {_bucket_score(strongest[0][1])}), "
                    f"{strongest[1][0]} ({strongest[1][1]:.2f}, {_bucket_score(strongest[1][1])})."
                )
                bullets.append(
                    f"**Weakest:** {weakest[0][0]} ({weakest[0][1]:.2f}, {_bucket_score(weakest[0][1])}), "
                    f"{weakest[1][0]} ({weakest[1][1]:.2f}, {_bucket_score(weakest[1][1])})."
                )
                # Practical implication (simple)
                if dim_vals["Fidelity"] >= 0.70 and dim_vals["Correctness"] >= 0.70:
                    bullets.append("**Practical read:** This is relatively trustworthy for reflecting what the model actually uses.")
                elif dim_vals["Simplicity"] >= 0.70 or dim_vals["Compactness"] >= 0.70:
                    bullets.append("**Practical read:** This is easy to communicate. But it may not fully match the model’s true behavior.")
                elif dim_vals["Stability"] < 0.55:
                    bullets.append("**Practical read:** Explanations may change noticeably with small input changes. Be cautious in decision-making.")
                else:
                    bullets.append("**Practical read:** Mixed profile. Use together with other techniques or focus on your priority dimensions.")

                technique_cards.append({"title": f"{tname}", "bullets": bullets})
            return {"headline": headline, "ranking_notes": ranking_notes, "technique_cards": technique_cards}
        st.subheader("Interpretability Panel")
        # Initialize cached variable
        cached = False  # or set based on conditions
        if cached:
            st.markdown("### Interpretation of Results")

            # Call the interpretation helper function
            interpretation = _interpret_iem_results(df_iem, weights_now, meta, selected_pipeline)

            # Show headline of the best technique and its top dimensions
            st.markdown(f"**Top-ranked explanation technique: ** {interpretation['headline']}")

            # Show ranking notes (i.e., why we see the current ranking)
            for note in interpretation['ranking_notes']:
                st.write(note)

            # Show detailed per-technique cards
            for card in interpretation['technique_cards']:
                with st.expander(f"**{card['title']}** - Detailed explanation", expanded=True):
                    for bullet in card['bullets']:
                        st.write(f"- {bullet}")
        # Safety checks
        if "last_uploaded_dataset_json" not in st.session_state:
            st.warning("No dataset found. Please upload a CSV and run AutoML first.")
            st.stop()

        if not st.session_state.get("automl_results"):
            st.warning("No AutoML results found. Please run AutoML first.")
            st.stop()
        # Build pipeline options from existing results
        # Expected keys look like: FLAML_RF, H2O_GBM, MLJAR_XGBoost, etc.
        pipeline_keys = sorted(list(st.session_state["automl_results"].keys()))
        selected_pipeline = st.selectbox("Select a pipeline to evaluate", pipeline_keys, index=0)
        # Parse framework and algorithm for backend
        if "_" not in selected_pipeline:
            st.error(f"Unexpected pipeline key format: {selected_pipeline}")
            st.stop()
        framework = selected_pipeline.split("_", 1)[0].strip()
        algorithm = selected_pipeline.split("_", 1)[1].strip()
        st.caption(
            f"Evaluating interpretability for framework: {framework} . algorithm: {algorithm}"
        )
        st.markdown("### Interpretability dimensions and weights")
        # Weight presets
        preset_map = {
            "Balanced": {
                "Fidelity": 1.0, "Stability": 1.0, "Compactness": 1.0,
                "Correctness": 1.0, "Simplicity": 1.0, "Completeness": 1.0
            },
            "Faithfulness first": {
                "Fidelity": 2.5, "Stability": 1.0, "Compactness": 0.8,
                "Correctness": 2.0, "Simplicity": 0.8, "Completeness": 1.0
            },
            "Robustness first": {
                "Fidelity": 1.0, "Stability": 2.5, "Compactness": 1.0,
                "Correctness": 1.0, "Simplicity": 0.8, "Completeness": 1.0
            },
            "Simple explanations": {
                "Fidelity": 1.0, "Stability": 1.0, "Compactness": 2.2,
                "Correctness": 1.0, "Simplicity": 2.2, "Completeness": 0.8
            },
            "Complete explanations": {
                "Fidelity": 1.0, "Stability": 1.0, "Compactness": 0.8,
                "Correctness": 1.0, "Simplicity": 0.8, "Completeness": 2.5
            },
        }
        if "iem_preset" not in st.session_state:
            st.session_state["iem_preset"] = "Balanced"

        chosen_preset = st.selectbox(
            "Weight preset",
            list(preset_map.keys()),
            index=list(preset_map.keys()).index(st.session_state["iem_preset"]),
        )
        st.session_state["iem_preset"] = chosen_preset
        preset = preset_map[chosen_preset]

        c1, c2, c3 = st.columns(3)
        with c1:
            w_fidelity = st.slider("Fidelity", 0.0, 3.0, float(preset["Fidelity"]), 0.1)
            w_stability = st.slider("Stability", 0.0, 3.0, float(preset["Stability"]), 0.1)
        with c2:
            w_compactness = st.slider("Compactness", 0.0, 3.0, float(preset["Compactness"]), 0.1)
            w_correctness = st.slider("Correctness", 0.0, 3.0, float(preset["Correctness"]), 0.1)
        with c3:
            w_simplicity = st.slider("Simplicity", 0.0, 3.0, float(preset["Simplicity"]), 0.1)
            w_completeness = st.slider("Completeness", 0.0, 3.0, float(preset["Completeness"]), 0.1)
        st.markdown("### Evaluation settings")
        c4, c5 = st.columns(2)
        with c4:
            n_eval = st.number_input("Number of test instances to evaluate", min_value=10, max_value=500, value=50, step=10)
        with c5:
            n_perturb = st.number_input("Surrogate perturbations per instance", min_value=50, max_value=2000, value=200, step=50)
        cache_key = f"{selected_pipeline}__n_eval={int(n_eval)}__n_perturb={int(n_perturb)}"
        force_recompute = st.checkbox("Force recompute explanations (ignore cache)", value=False)
        compute_btn = st.button("Compute Interpretability Metrics")
        # Ensure cache exists
        if "iem_cache" not in st.session_state:
            st.session_state["iem_cache"] = {}
        # Decide whether we need to call backend
        should_call_backend = force_recompute or (cache_key not in st.session_state["iem_cache"])
        # Only call backend when user clicks compute AND we need new data
        if compute_btn and should_call_backend:
            payload = {
                "data": st.session_state["last_uploaded_dataset_json"],
                "framework": framework,
                "algorithm": algorithm,
                "n_eval": int(n_eval),
                "n_perturb": int(n_perturb),
                # weights are not needed for caching dimensions, but backend accepts them.
                "weights": {
                    "fidelity": float(w_fidelity),
                    "stability": float(w_stability),
                    "compactness": float(w_compactness),
                    "correctness": float(w_correctness),
                    "simplicity": float(w_simplicity),
                    "completeness": float(w_completeness),   }, }
            with st.spinner("Computing explanation techniques and dimension scores..."):
                try:
                    r = requests.post(f"{BACKEND_URL}/interpretability_eval", json=payload, timeout=300)
                    try:
                        out = r.json()
                    except Exception:
                        st.error(f"Backend returned non-JSON (status {r.status_code}).\n\n{r.text[:2000]}")
                        st.stop()

                    if r.status_code != 200:
                        st.error(out.get("error", f"Request failed with status {r.status_code}"))
                        st.stop()

                    st.session_state["latest_interpretability_result"] = out

                except Exception as e:
                    st.error(f"Request failed: {e}")
                    st.stop()

            techniques_raw = out.get("techniques", [])
            meta = out.get("meta", {})
            if not techniques_raw:
                st.warning("No techniques returned from backend.")
            else:
                # Cache only what we need for fast recomputation of IEM
                cached_techniques = []
                for t in techniques_raw:
                    cached_techniques.append({
                        "name": t.get("name", ""),
                        "dimensions": t.get("dimensions", {}) or {},
                        "artifacts": t.get("artifacts", {}) or {}
                    })
                st.session_state["iem_cache"][cache_key] = {
                    "techniques": cached_techniques,
                    "meta": meta, }
        # Fetch cached results (if any)
        cached = st.session_state["iem_cache"].get(cache_key)
        # If not computed yet, show message but DO NOT stop the app
        if cached is None:
            st.info("Click 'Compute Interpretability Metrics' to generate interpretability results for this pipeline.")
        else:
            # Use cached dimension scores, recompute IEM locally using current weights
            techniques = cached["techniques"]
            meta = cached.get("meta", {})

            weights_now = {
                "fidelity": float(w_fidelity),
                "stability": float(w_stability),
                "compactness": float(w_compactness),
                "correctness": float(w_correctness),
                "simplicity": float(w_simplicity),
                "completeness": float(w_completeness),
            }
            w_sum = sum(weights_now.values()) if sum(weights_now.values()) > 0 else 1.0
            rows = []
            for t in techniques:
                dims = t.get("dimensions", {}) or {}
                f = float(dims.get("fidelity", 0.0))
                s = float(dims.get("stability", 0.0))
                c = float(dims.get("compactness", 0.0))
                cor = float(dims.get("correctness", 0.0))
                simp = float(dims.get("simplicity", 0.0))
                comp = float(dims.get("completeness", 0.0))

                iem = (
                    weights_now["fidelity"] * f
                    + weights_now["stability"] * s
                    + weights_now["compactness"] * c
                    + weights_now["correctness"] * cor
                    + weights_now["simplicity"] * simp
                    + weights_now["completeness"] * comp
                ) / w_sum

                rows.append({
                    "Technique": t.get("name", ""),
                    "IEM": float(iem),
                    "Fidelity": f,
                    "Stability": s,
                    "Compactness": c,
                    "Correctness": cor,
                    "Simplicity": simp,
                    "Completeness": comp,
                })
            df_iem = pd.DataFrame(rows).sort_values("IEM", ascending=False).reset_index(drop=True)
            st.markdown("### Interpretability Evaluation Metric (IEM) results")
            st.caption(f"n_features: {meta.get('n_features')} . n_eval: {meta.get('n_eval')}")
            st.dataframe(df_iem, hide_index=True)
            # Download results as CSV (includes settings in filename)
            csv_bytes = df_iem.to_csv(index=False).encode("utf-8")
            download_name = f"iem_results__{selected_pipeline}__n_eval={int(meta.get('n_eval', 0))}__n_perturb={int(n_perturb)}.csv"
            st.download_button(
                label="Download IEM results (CSV)",
                data=csv_bytes,
                file_name=download_name,
                mime="text/csv",)
            fig = px.bar(df_iem, x="Technique", y="IEM", title="IEM by explanation technique")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### Dimension profile by technique")
            dims_order = ["Fidelity", "Stability", "Compactness", "Correctness", "Simplicity", "Completeness"]
            radar_fig = go.Figure()
            for _, row in df_iem.iterrows():
                values = [float(row[d]) for d in dims_order]
                values_closed = values + [values[0]]
                dims_closed = dims_order + [dims_order[0]]
                radar_fig.add_trace(
                    go.Scatterpolar(
                        r=values_closed,
                        theta=dims_closed,
                        fill="toself",
                        name=str(row["Technique"]),
                    )
                )
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Interpretability dimensions (0 to 1)",
            )
            st.plotly_chart(radar_fig, use_container_width=True)
            st.markdown("### Explanation visuals (per technique)")
            tech_by_name = {t.get("name", ""): (t.get("artifacts", {}) or {}) for t in techniques}
            # Show in the same order as the IEM ranking table
            for _, rrow in df_iem.iterrows():
                tname = str(rrow["Technique"])
                artifacts = tech_by_name.get(tname, {}) or {}
                top_feats = artifacts.get("top_features", []) or []
                label = {
                    "shap": "SHAP global feature impact",
                    "permutation": "Permutation importance",
                    "local_surrogate": "Local surrogate coefficients"
                }.get(tname, tname)
                with st.expander(f"{label} . Top features", expanded=(tname == str(df_iem.iloc[0]['Technique']))):
                    if not top_feats:
                        st.info("No visualization data returned for this technique.")
                    else:
                        df_top = pd.DataFrame(top_feats)
                        df_top = df_top.sort_values("importance", ascending=True)

                        fig_top = px.bar(
                            df_top,
                            x="importance",
                            y="feature",
                            orientation="h",
                            title=f"Top features by {tname}"
                        )
                        st.plotly_chart(fig_top, use_container_width=True)

                        if tname == "local_surrogate":
                            sr2 = artifacts.get("surrogate_r2", None)
                            npb = artifacts.get("n_perturb", None)
                            if sr2 is not None:
                                st.caption(f"Surrogate fit proxy (R²): {float(sr2):.4f} . Perturbations per instance: {npb}")
            st.markdown("### Local explanation for one instance")
            df_all = pd.read_json(StringIO(st.session_state["last_uploaded_dataset_json"]))
            max_row = max(0, len(df_all) - 1)
            c_loc1, c_loc2, c_loc3 = st.columns([2, 2, 2])
            with c_loc1:
                row_index = st.number_input("Row index (from uploaded dataset)", min_value=0, max_value=int(max_row), value=0, step=1)
            with c_loc2:
                top_k = st.number_input("Top features to show", min_value=5, max_value=30, value=10, step=1)
            with c_loc3:
                n_perturb_local = st.number_input("Surrogate perturbations (local)", min_value=50, max_value=2000, value=int(n_perturb), step=50)
            explain_btn = st.button("Explain this instance")
            if explain_btn:
                payload_local = {
                    "data": st.session_state["last_uploaded_dataset_json"],
                    "framework": framework,
                    "algorithm": algorithm,
                    "row_index": int(row_index),
                    "n_perturb": int(n_perturb_local),
                    "top_k": int(top_k)
                }
                with st.spinner("Computing local explanations..."):
                    try:
                        r = requests.post(f"{BACKEND_URL}/explain_instance", json=payload_local, timeout=300)
                        out_local = r.json()
                        if r.status_code != 200:
                            st.error(out_local.get("error", f"Request failed with status {r.status_code}"))
                            st.stop()
                    except Exception as e:
                        st.error(f"Request failed: {e}")
                        st.stop()
                pred = out_local.get("prediction", {}) or {}
                meta_l = out_local.get("meta", {}) or {}
                loc = out_local.get("local", {}) or {}
                st.success(
                    f"Prediction computed for row {meta_l.get('row_index')} . "
                    f"pred_class_index: {pred.get('pred_class_index')} . "
                    f"pred_class: {pred.get('pred_class')}"
                )
                proba = pred.get("proba", [])
                if proba:
                    st.caption("Predicted probabilities (order follows model classes): " + ", ".join([f"{p:.4f}" for p in proba]))
                # Feature values table
                top_vals = loc.get("top_feature_values", []) or []
                if top_vals:
                    st.markdown("#### Feature values (for top explained features)")
                    st.dataframe(pd.DataFrame(top_vals), hide_index=True)
                cA, cB, cC = st.columns(3)
                # SHAP local contributions
                with cA:
                    st.markdown("#### SHAP local contributions")
                    shap_items = loc.get("shap", []) or []
                    if not shap_items:
                        st.info("SHAP local values not available for this model or environment.")
                    else:
                        df_sh = pd.DataFrame(shap_items)
                        df_sh = df_sh.sort_values("contribution", ascending=True)
                        fig_sh = px.bar(df_sh, x="contribution", y="feature", orientation="h", title="SHAP (signed contributions)")
                        st.plotly_chart(fig_sh, use_container_width=True)
                # Occlusion sensitivity
                with cB:
                    st.markdown("#### Occlusion sensitivity")
                    occ_items = loc.get("occlusion", []) or []
                    if not occ_items:
                        st.info("Occlusion sensitivity not available (needs SHAP top features).")
                    else:
                        df_occ = pd.DataFrame(occ_items)
                        df_occ = df_occ.sort_values("drop", ascending=True)
                        fig_occ = px.bar(df_occ, x="drop", y="feature", orientation="h", title="Drop in confidence when masked")
                        st.plotly_chart(fig_occ, use_container_width=True)
                # Local surrogate coefficients
                with cC:
                    st.markdown("#### Local surrogate coefficients")
                    sur_items = loc.get("local_surrogate", []) or []
                    if not sur_items:
                        st.info("Local surrogate not available.")
                    else:
                        df_su = pd.DataFrame(sur_items)
                        df_su = df_su.sort_values("weight", ascending=True)
                        fig_su = px.bar(df_su, x="weight", y="feature", orientation="h", title="Surrogate (signed weights)")
                        st.plotly_chart(fig_su, use_container_width=True)
                # ---- Add the Interpretation Section Here ----
                st.markdown("### Interpretation of Results")
                # Call the interpretation helper function
                interpretation = _interpret_iem_results(df_iem, weights_now, meta, selected_pipeline)
                # Show headline of the best technique and why
                st.markdown(f"**Top-ranked explanation technique:** {interpretation['headline']}")
                # Show ranking notes (i.e., why we see the current ranking)
                for note in interpretation['ranking_notes']:
                    st.write(note)
                # Show detailed per-technique cards
                for card in interpretation['technique_cards']:
                    with st.expander(f"**{card['title']}** - Detailed explanation", expanded=True):
                        for bullet in card['bullets']:
                            st.write(f"- {bullet}")
            with st.expander("Interpretability dimensions, how we compute them, and scientific grounding"):
                st.markdown(
                    """
            **Fidelity (faithfulness)**
            - **What it captures:** Does the explanation highlight features that truly drive the model’s prediction.
            - **How we compute it here:** AOPC-style masking. We replace the top-k features (by explanation importance) with their training mean and measure the relative drop in the model’s confidence for the originally predicted class.
            - **Why this is grounded:** Perturbation-based faithfulness metrics are standard for evaluating explanation quality.

            **Stability (robustness)**
            - **What it captures:** Similar inputs should yield similar explanations.
            - **How we compute it here:** We add small Gaussian noise to the evaluation inputs and recompute an explanation vector. Then we measure cosine similarity between the original and noisy explanation vectors.
            - **Why this is grounded:** Robustness and stability are widely recognized desiderata for reliable explanations.

            **Correctness (infidelity-style)**
            - **What it captures:** How well the explanation function aligns with the model’s behavior under input changes.
            - **How we compute it here:** We use a simplified infidelity objective. We sample random perturbations and compare the true change in model output to the change predicted by the explanation vector. We convert mean squared error into a bounded score: `1 / (1 + infidelity)`.
            - **Why this is grounded:** Infidelity and sensitivity were proposed as objective measures of explanation quality.

            **Compactness**
            - **What it captures:** Whether the explanation focuses on a small set of features.
            - **How we compute it here:** Minimum number of top features needed to cover 90% of total importance mass, normalized so fewer features means higher compactness.

            **Simplicity**
            - **What it captures:** Whether the explanation distribution is concentrated or diffuse.
            - **How we compute it here:** Inverted normalized entropy of the importance distribution. Lower entropy means higher simplicity.

            **Completeness**
            - **What it captures:** Whether explanations “account for” the model output.
            - **How we compute it here:**
            - **SHAP:** completeness proxy based on SHAP’s additive local accuracy idea. We check how well `base_value + sum(SHAP)` reconstructs the model score. Smaller reconstruction error gives higher completeness.
            - **Local surrogate:** completeness proxy uses surrogate fit quality (R²). Better surrogate fit gives higher completeness.
            - **Permutation:** treated as neutral for completeness in the current version.

            **IEM (Interpretability Evaluation Metric)**
            - A weighted composite score across the above dimensions. Each dimension is normalized to [0, 1]. You can tune weights to reflect your priorities.
                    """
                )
                st.markdown("### Key references (you can cite these in the paper)")
                st.markdown(
                    """
            - **SHAP:** Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*.
            - **LIME / local surrogate concept:** Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *“Why Should I Trust You?”: Explaining the Predictions of Any Classifier*.
            - **Infidelity / sensitivity correctness metrics:** Yeh, C. K., Hsieh, C. Y., Suggala, A. S., Inouye, D. I., & Ravikumar, P. (2019). *On the (In)fidelity and Sensitivity of Explanations*.
            - **Faithfulness via perturbation:** AOPC-style perturbation faithfulness metrics are commonly used in explanation evaluation literature.
                    """
                )
        with st.expander("💬 Ask about the interpretability results", expanded=True):
            q_interp = st.text_input(
                "Ask a question about the model’s explanation:",
                "",
                key="interp_llm_q"
            )
            verbose_interp = st.checkbox(
                "Verbose explanation",
                value=False,
                key="interp_llm_verbose"
            )
            ask_interp = st.button("Ask", key="interp_llm_btn")

            if ask_interp and q_interp.strip():
                with st.spinner("Thinking..."):
                    answer = grounded_llm_answer_interp(
                        q_interp.strip(),
                        verbose=verbose_interp
                    )
                st.write(answer)
    elif st.session_state.get("show_audience_mode", False):

        st.title("👥 Audience-Specific Explanation Panel")
        st.caption(
            "Generate tailored explanations for different audiences. "
            "Select your audience type and domain to receive explanations in their language."
        )
        audience_ctx = _build_audience_context()
        classification_action_space = _get_classification_action_space()
        if not audience_ctx.get("best_model"):
            st.warning("No AutoML results found. Please run AutoML first.")
        else:
            # ==========================================================
            # VISUAL MODEL OVERVIEW
            # ==========================================================

            st.markdown("### 📊 Model Overview")

            best_metrics = (
                audience_ctx.get(
                    "best_model_metrics",
                    {}
                )
                or {}
            )

            best_model = audience_ctx.get(
                "best_model",
                "N/A"
            )

            accuracy = (
                f"{float(best_metrics['accuracy']):.3f}"
                if best_metrics.get(
                    "accuracy"
                ) is not None
                else "N/A"
            )

            f1 = (
                f"{float(best_metrics['f1']):.3f}"
                if best_metrics.get(
                    "f1"
                ) is not None
                else "N/A"
            )

            co2 = (
                f"{float(best_metrics['co2']):.4f}"
                if best_metrics.get(
                    "co2"
                ) is not None
                else "N/A"
            )

            energy = (
                f"{float(best_metrics['energy']):.4f}"
                if best_metrics.get(
                    "energy"
                ) is not None
                else "N/A"
            )

            overview_cols = st.columns(
                [1.45, 1, 1, 1, 1],
                gap="small"
            )

            cards = [
                (
                    "sml-blue",
                    "🏆 Current Winner",
                    best_model,
                    "Best-performing AutoML result"
                ),
                (
                    "sml-blue",
                    "🎯 Accuracy",
                    accuracy,
                    "Predictive performance"
                ),
                (
                    "sml-blue",
                    "📊 F1 Score",
                    f1,
                    "Class-balanced performance"
                ),
                (
                    "sml-green",
                    "🌱 CO₂",
                    co2,
                    "Measured during training"
                ),
                (
                    "sml-green",
                    "⚡ Energy",
                    energy,
                    "Measured during training"
                ),
            ]

            for column, card in zip(
                overview_cols,
                cards
            ):
                css_class, title, value, subtitle = card

                with column:
                    st.markdown(
                        f"""
                        <div class="sml-card {css_class}">
                            <div class="sml-card-title">
                                {title}
                            </div>
                            <div class="sml-card-value">
                                {value}
                            </div>
                            <div class="sml-card-subtitle">
                                {subtitle}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            top_features = (
                audience_ctx.get(
                    "top_features",
                    []
                )
                or []
            )

            if top_features:

                feature_rows = []

                for feature in top_features[:5]:

                    try:
                        importance = abs(
                            float(
                                feature.get(
                                    "importance",
                                    0
                                )
                            )
                        )
                    except (
                        TypeError,
                        ValueError
                    ):
                        importance = 0.0

                    feature_rows.append({
                        "Feature": str(
                            feature.get(
                                "feature",
                                "Unknown"
                            )
                        ),
                        "Influence": importance,
                    })

                feature_df = pd.DataFrame(
                    feature_rows
                )

                feature_df = feature_df.sort_values(
                    "Influence",
                    ascending=True
                )

                st.markdown(
                    "### 🔎 Most Influential Features"
                )

                feature_fig = px.bar(
                    feature_df,
                    x="Influence",
                    y="Feature",
                    orientation="h",
                    text="Influence",
                )

                feature_fig.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside",
                    cliponaxis=False,
                )

                feature_fig.update_layout(
                    height=280,
                    showlegend=False,
                    xaxis_title=(
                        "Relative influence magnitude"
                    ),
                    yaxis_title=None,
                    margin=dict(
                        l=10,
                        r=50,
                        t=10,
                        b=25
                    ),
                )

                st.plotly_chart(
                    feature_fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    }
                )

                st.caption(
                    "Influence magnitude shows which features had "
                    "the strongest effect on the model. It does not "
                    "imply that a feature causes the outcome."
                )
            # SECTION 1: AUDIENCE & DOMAIN SELECTION
            st.markdown("### 1 · Choose the Audience")

            audience_type = st.selectbox(
                "Who will receive this explanation?",
                options=[
                    "Normal User",
                    "Domain Expert",
                    "Data Scientist"
                ],
                index=["Normal User", "Domain Expert", "Data Scientist"].index(
                    st.session_state.get("audience_selected_type", "Normal User")
                ),
                key="audience_mode_type_dropdown",
                help="Select the audience type"
            )
            st.session_state["audience_selected_type"] = audience_type
            audience_descriptions = {
                "Normal User": "👤 Someone with no data science background. Explanation will be simple and use everyday language.",
                "Domain Expert": "👨‍⚕️ Professional in the relevant field (doctor, banker, engineer). Explanation will use domain-specific terminology.",
                "Data Scientist": "🔬 ML professional. Explanation will be technical with focus on algorithms and metrics."
            }
            st.info(audience_descriptions[audience_type])
            # Auto-detect domain
            if "dataset_df" in st.session_state:
                domain = auto_detect_domain(st.session_state["dataset_df"])
            else:
                domain = "General"
            st.session_state["selected_domain"] = domain
            # SECTION 3: GENERATE EXPLANATIONS
            st.markdown(  "### 2 · Generate the Explanation")
            col_gen, col_gen_space = st.columns([3, 1])
            with col_gen:
                if st.button(
                    f"🚀 Generate Explanation for {audience_type}",
                    key="generate_explanation_btn",
                    use_container_width=True
                ):
                    st.session_state["generating_explanation"] = True
            if st.session_state.get("generating_explanation", False):
                with st.spinner(f"🔄 Generating explanation for {audience_type} in {domain} domain..."):
                    try:
                        # Extract model, features, and metrics info
                        model_info = _extract_model_info_for_explanation()
                        features_info = _extract_features_for_explanation(top_k=5)
                        metrics_info = _extract_metrics_for_explanation()
                        xai_signals = _extract_xai_signals_for_explanation()  # ADD THIS
                        # ----------------------------------------------------
                        # Build the constrained recommendation context.
                        # The recommendation is generated once and reused
                        # across all audience types.
                        # ----------------------------------------------------
                        classification_action_space = _get_classification_action_space()

                        recommendation_context = json.dumps(
                            classification_action_space,
                            sort_keys=True,
                            default=str
                        )

                        # If the model/configuration changed, discard the
                        # previous recommendation and create a fresh one.
                        if (
                            st.session_state.get(
                                "classification_recommendation_context"
                            )
                            != recommendation_context
                        ):
                            st.session_state["classification_recommendation"] = None
                            st.session_state[
                                "classification_recommendation_context"
                            ] = recommendation_context
                        # Load baseline  # ADD THIS
                        try:
                            baseline = load_baseline("./baseline_medical.json")
                        except:
                            baseline = None
                        
                        st.session_state["current_features_for_answer"] = features_info
                        
                        use_openai = bool(os.getenv("OPENAI_API_KEY"))
                        openai_api_key = os.getenv("OPENAI_API_KEY") if use_openai else None
                        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
                        ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
                        # Generate exactly one recommendation only when
                        # one has not already been generated for this
                        # model/configuration.
                        if st.session_state.get(
                            "classification_recommendation"
                        ) is None:

                            recommendation = generate_dashboard_recommendation(
                                action_space=classification_action_space,
                                model_info=model_info,
                                metrics_info=metrics_info,
                                use_openai=use_openai,
                                openai_api_key=openai_api_key,
                                ollama_url=ollama_url,
                                ollama_model=ollama_model
                            )

                            st.session_state[
                                "classification_recommendation"
                            ] = recommendation
                        result = generate_audience_explanation(
                            audience_type=audience_type,
                            domain=domain,
                            model_info=model_info,
                            features_info=features_info,
                            metrics_info=metrics_info,
                            task="classification",
                            use_openai=use_openai,
                            openai_api_key=openai_api_key,
                            ollama_url=ollama_url,
                            ollama_model=ollama_model,
                            xai_signals=xai_signals,  # ADD THIS
                            baseline=baseline, # ADD THIS
                            dashboard_recommendation=(
                                st.session_state.get(
                                    "classification_recommendation"
                                )
                                if (
                                    st.session_state.get(
                                        "classification_recommendation",
                                        {}
                                    )
                                    or {}
                                ).get("status") == "success"
                                else None
                            )
                                                    )
                        # Store result in session state
                        if "explanation_results" not in st.session_state:
                            st.session_state["explanation_results"] = {}
                        st.session_state["explanation_results"][audience_type] = result
                        st.session_state["generating_explanation"] = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating explanation: {str(e)}")
                        st.session_state["generating_explanation"] = False
            # SECTION 4: DISPLAY EXPLANATIONS
            explanation_results = st.session_state.get("explanation_results", {})
            if not explanation_results or audience_type not in explanation_results:
                st.info("👉 Click the button above to generate an explanation.")
            else:
                result = explanation_results[audience_type]
                if result.get("status") == "error":
                    st.error(f"**Error:** {result.get('explanation')}")
                else:
                    explanation_text = result.get("explanation", "No explanation generated")
                    
                    # Clean asterisks and format as HTML
                    import re
                    # Convert * **text:** to bold headings
                    formatted = re.sub(r'\* \*\*(.+?):\*\*', r'<strong style="color: #1f77b4;">\1:</strong>', explanation_text)
                    # Convert remaining ** to nothing (already handled above)
                    formatted = formatted.replace('**', '')
                    # Convert newlines to breaks
                    formatted = formatted.replace('\n', '<br>')
                    
                    st.markdown(
                        f"""
                        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; line-height: 1.8;">
                        {formatted}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            # ==============================================================
            # STRUCTURED DASHBOARD RECOMMENDATION
            # ==============================================================
            repeated_validation = st.session_state.get(
                "classification_repeated_validation"
            )

            if repeated_validation:
                st.markdown(
                    "### ✅ What Happened After Validation"
                )

                validated_explanation = (
                    _format_repeated_validation_for_audience(
                        repeated_bundle=repeated_validation,
                        audience_type=audience_type
                    )
                )

                st.info(
                    validated_explanation
                )
            recommendation = st.session_state.get(
                "classification_recommendation"
            )

            if recommendation:
                st.markdown("---")
                if repeated_validation:
                    st.markdown(
                        "### 🧪 Original Experiment Hypothesis"
                    )
                else:
                    st.markdown(
                        "### 🧪 Suggested Dashboard Experiment"
                    )

                if recommendation.get("status") == "success":

                    rec_col1, rec_col2 = st.columns(2)

                    with rec_col1:
                        st.markdown("#### Current")
                        st.code(
                            f"{recommendation.get('control')} = "
                            f"{recommendation.get('current_value')}"
                        )

                    with rec_col2:
                        st.markdown("#### Suggested")
                        st.code(
                            f"{recommendation.get('control')} = "
                            f"{recommendation.get('suggested_value')}"
                        )

                    st.markdown("#### Hypothesis")
                    st.info(
                        recommendation.get(
                            "hypothesis",
                            "No hypothesis provided."
                        )
                    )

                    expected_effect = recommendation.get(
                        "expected_effect",
                        {}
                    )

                    if expected_effect:
                        with st.expander("Expected effect — not yet validated"):
                            st.json(expected_effect)

                    st.warning(
                        "This recommendation is an experimental hypothesis. "
                        "It has NOT been validated yet. A rerun is required "
                        "before determining whether the change improves or "
                        "worsens the result."
                    )
                    st.markdown("#### Validate This Recommendation")

                    st.caption(
                        "This reruns only the current winning model with "
                        "exactly the one proposed change. Your original "
                        "AutoML results remain unchanged."
                    )

                    if st.button(
                        "▶ Apply & Validate Recommendation",
                        key="validate_classification_recommendation_btn",
                        use_container_width=True
                    ):
                        try:
                            # ------------------------------------------
                            # 1. Preserve original baseline result
                            # ------------------------------------------

                            current_results = (
                                st.session_state.get(
                                    "automl_results",
                                    {}
                                )
                                or {}
                            )

                            best_model_key, best_metrics = (
                                _best_model_from_results(
                                    current_results
                                )
                            )

                            if (
                                not best_model_key
                                or not best_metrics
                            ):
                                st.error(
                                    "Could not identify the baseline "
                                    "winning model."
                                )
                                st.stop()

                            framework, algorithm = (
                                _split_model_key(
                                    best_model_key
                                )
                            )

                            baseline_snapshot = {
                                "model": best_model_key,
                                "framework": framework,
                                "algorithm": algorithm,
                                "Accuracy": best_metrics.get(
                                    "Accuracy"
                                ),
                                "F1 Score": best_metrics.get(
                                    "F1 Score"
                                ),
                                "CO2 Emission": best_metrics.get(
                                    "CO2 Emission"
                                ),
                                "Energy Consumption": best_metrics.get(
                                    "Energy Consumption"
                                ),
                              
                            }

                            # ------------------------------------------
                            # 2. Preserve any user-edited hyperparams
                            # ------------------------------------------

                            # Use only hyperparameters that are actually
                            # exposed by the dashboard/recommendation space.
                            # Do NOT send the estimator's full get_params()
                            # output back into the model.
                            action_space_for_validation = (
                                _get_classification_action_space()
                            )

                            baseline_params = (
                                action_space_for_validation
                                .get(
                                    "current_values",
                                    {}
                                )
                                .copy()
                            )

                            validation_params = (
                                baseline_params.copy()
                            )

                            validation_time_budget = int(
                                st.session_state.get(
                                    "time_budget",
                                    30
                                )
                            )

                            # ------------------------------------------
                            # 3. Apply EXACTLY ONE recommendation
                            # ------------------------------------------

                            control_type = (
                                recommendation.get(
                                    "control_type"
                                )
                            )

                            control = recommendation.get(
                                "control"
                            )

                            suggested_value = (
                                recommendation.get(
                                    "suggested_value"
                                )
                            )

                            if control_type == "hyperparameter":
                                validation_params[
                                    control
                                ] = suggested_value

                            elif control_type == "time_budget":
                                validation_time_budget = int(
                                    suggested_value
                                )

                            else:
                                st.error(
                                    "Unsupported classification "
                                    "validation control."
                                )
                                st.stop()

                            dataset_json = (
                                st.session_state.get(
                                    "last_uploaded_dataset_json"
                                )
                            )

                            if not dataset_json:
                                st.error(
                                    "Original dataset is not available "
                                    "in session state. Please rerun "
                                    "AutoML once."
                                )
                                st.stop()

                            # ------------------------------------------
                            # 4. Dedicated validation request
                            # ------------------------------------------

                            validation_seed = 42

                            with st.spinner(
                                "Running matched baseline and recommended "
                                "configurations..."
                            ):

                                baseline_result = (
                                    _run_classification_validation_request(
                                        dataset_json=dataset_json,
                                        framework=framework,
                                        algorithm=algorithm,
                                        hyperparams=baseline_params,
                                        time_budget=int(
                                            st.session_state.get(
                                                "time_budget",
                                                30
                                            )
                                        ),
                                        seed=validation_seed
                                    )
                                )

                                validation_result = (
                                    _run_classification_validation_request(
                                        dataset_json=dataset_json,
                                        framework=framework,
                                        algorithm=algorithm,
                                        hyperparams=validation_params,
                                        time_budget=validation_time_budget,
                                        seed=validation_seed
                                    )
                                )

                
                            comparison = (
                                _compare_classification_validation(
                                    baseline_result,
                                    validation_result
                                )
                            )

                            st.session_state[
                                "classification_validation_baseline"
                            ] = baseline_result

                            st.session_state[
                                "classification_validation_result"
                            ] = {
                                "rerun": validation_result,
                                "comparison": comparison,
                                "recommendation": (
                                    recommendation.copy()
                                ),
                            }

                            st.session_state[
                                "classification_validation_error"
                            ] = None

                            st.rerun()

                        except Exception as e:
                            st.session_state[
                                "classification_validation_error"
                            ] = str(e)

                            st.error(
                                f"Recommendation validation failed: {e}"
                            )
                else:
                    st.error(
                        "The generated recommendation was rejected by the "
                        "dashboard validator."
                    )

                    with st.expander(
                        "🔍 Recommendation validation details"
                    ):
                        st.json(recommendation)
            validation_bundle = st.session_state.get(
                "classification_validation_result"
            )

            validation_error = st.session_state.get(
                "classification_validation_error"
            )

            if validation_error:
                st.error(
                    f"Latest validation error: {validation_error}"
                )

            if validation_bundle:
                st.markdown("---")
                st.markdown(
                     "### 📊 Preliminary Single-Run Validation"
                )

                st.caption(
                    "This is a preliminary single-run comparison. "
    "Use the repeated validation below for the main empirical conclusion."
                )

                baseline = st.session_state.get(
                    "classification_validation_baseline",
                    {}
                )

                rerun = validation_bundle.get(
                    "rerun",
                    {}
                )

                comparison = validation_bundle.get(
                    "comparison",
                    {}
                )

                metric_rows = []

                display_metrics = [
                    (
                        "Accuracy",
                        "Accuracy"
                    ),
                    (
                        "F1 Score",
                        "F1 Score"
                    ),
                    (
                        "ROC-AUC",
                        "ROC-AUC"
                    ),
                    (
                        "Log Loss",
                        "Log Loss"
                    ),

                    (
                        "CO₂ Emission",
                        "CO2 Emission"
                    ),
                    (
                        "Energy Consumption",
                        "Energy Consumption"
                    ),
                ]

                comparison_metrics = (
                    comparison.get(
                        "metrics",
                        {}
                    )
                )

                for label, key in display_metrics:
                    metric_info = (
                        comparison_metrics.get(
                            key,
                            {}
                        )
                    )

                    metric_rows.append({
                        "Metric": label,
                        "Before": metric_info.get(
                            "baseline"
                        ),
                        "After": metric_info.get(
                            "validation"
                        ),
                        "Δ": metric_info.get(
                            "delta"
                        ),
                        "% Change": metric_info.get(
                            "percent_change"
                        ),
                    })

                validation_df = pd.DataFrame(
                    metric_rows
                )

                with st.expander(
                    "View preliminary single-run metrics",
                    expanded=False
                ):
                    st.dataframe(
                        validation_df,
                        hide_index=True,
                        use_container_width=True
                    )
                outcome = comparison.get(
                    "outcome",
                    "no_clear_change"
                )

                outcome_messages = {
                    "quality_improved": (
                        "✅ In this single rerun, predictive quality "
                        "improved without a >5% measured sustainability "
                        "penalty."
                    ),
                    "sustainability_improved": (
                        "🌱 In this single rerun, measured sustainability "
                        "improved by more than 5% without a meaningful "
                        "predictive-quality decrease."
                    ),
                    "tradeoff": (
                        "⚖️ This rerun produced a trade-off: one objective "
                        "improved while another worsened."
                    ),
                    "worse": (
                        "❌ In this single rerun, predictive quality "
                        "decreased without a compensating measured "
                        "sustainability improvement."
                    ),
                    "no_clear_change": (
                        "➖ This rerun did not show a clear change beyond "
                        "the current comparison tolerances."
                    ),
                }

                st.info(
                    outcome_messages.get(
                        outcome,
                        outcome_messages[
                            "no_clear_change"
                        ]
                    )
                )

                rec = validation_bundle.get(
                    "recommendation",
                    {}
                )
                st.markdown(
                    "#### Experiment actually tested"
                )

                st.code(
                    f"{rec.get('control')}: "
                    f"{rec.get('current_value')} → "
                    f"{rec.get('suggested_value')}"
                )

                with st.expander(
                    "🔍 Validation details"
                ):
                    st.json(validation_bundle)

            st.markdown("---")
            st.markdown(
                "### 🔁 Repeated Recommendation Validation"
            )

            st.caption(
                "For stronger evidence, rerun BOTH the current configuration "
                "and the recommended configuration using the same set of "
                "experimental seeds. Each seed forms a paired comparison."
            )

            repeat_count = st.selectbox(
                "Number of paired repetitions",
                options=[3, 5, 10],
                index=1,
                key="classification_validation_repeat_count",
                help=(
                    "Five paired repetitions is a reasonable development "
                    "default. More repetitions take longer but provide a "
                    "better estimate of variability."
                )
            )

            if st.button(
                "🔁 Run Repeated Validation",
                key="run_repeated_classification_validation",
                use_container_width=True
            ):

                try:
                    recommendation = st.session_state.get(
                        "classification_recommendation"
                    )

                    if (
                        not recommendation
                        or recommendation.get("status") != "success"
                    ):
                        st.error(
                            "No valid recommendation is available."
                        )
                        st.stop()

                    current_results = (
                        st.session_state.get(
                            "automl_results",
                            {}
                        )
                        or {}
                    )

                    best_model_key, best_metrics = (
                        _best_model_from_results(
                            current_results
                        )
                    )

                    if (
                        not best_model_key
                        or not best_metrics
                    ):
                        st.error(
                            "Could not identify the current winning model."
                        )
                        st.stop()

                    framework, algorithm = (
                        _split_model_key(
                            best_model_key
                        )
                    )

                    dataset_json = (
                        st.session_state.get(
                            "last_uploaded_dataset_json"
                        )
                    )

                    if not dataset_json:
                        st.error(
                            "Dataset is not available in session state. "
                            "Please rerun AutoML."
                        )
                        st.stop()

                    # --------------------------------------------------
                    # Use ACTUAL parameters reported by backend.
                    # This is much safer than assuming frontend defaults.
                    # --------------------------------------------------

                    action_space_for_validation = (
                        _get_classification_action_space()
                    )

                    baseline_params = (
                        action_space_for_validation
                        .get(
                            "current_values",
                            {}
                        )
                        .copy()
                    )

                    recommended_params = (
                        baseline_params.copy()
                    )

                    baseline_time_budget = int(
                        st.session_state.get(
                            "time_budget",
                            30
                        )
                    )

                    recommended_time_budget = (
                        baseline_time_budget
                    )

                    control_type = (
                        recommendation.get(
                            "control_type"
                        )
                    )

                    control = recommendation.get(
                        "control"
                    )

                    suggested_value = (
                        recommendation.get(
                            "suggested_value"
                        )
                    )

                    if control_type == "hyperparameter":
                        recommended_params[
                            control
                        ] = suggested_value

                    elif control_type == "time_budget":
                        recommended_time_budget = int(
                            suggested_value
                        )

                    else:
                        st.error(
                            "Unsupported classification recommendation."
                        )
                        st.stop()

                    # Fixed deterministic seed pool.
                    seed_pool = [
                        42,
                        123,
                        202,
                        314,
                        777,
                        1001,
                        2027,
                        4099,
                        8081,
                        16001,
                    ]

                    seeds = seed_pool[
                        :int(repeat_count)
                    ]

                    baseline_runs = []
                    recommended_runs = []

                    progress = st.progress(0)

                    status_text = st.empty()

                    total_jobs = (
                        len(seeds) * 2
                    )

                    completed_jobs = 0

                    for repeat_index, seed_value in enumerate(
                        seeds,
                        start=1
                    ):

                        baseline_first = (
                            repeat_index % 2 == 1
                        )

                        if baseline_first:
                            run_order = [
                                "baseline",
                                "recommended"
                            ]
                        else:
                            run_order = [
                                "recommended",
                                "baseline"
                            ]

                        pair_results = {}

                        for run_type in run_order:

                            status_text.write(
                                f"Pair {repeat_index}/{len(seeds)}: "
                                f"{run_type}, seed={seed_value}"
                            )

                            if run_type == "baseline":
                                params = baseline_params
                                budget = baseline_time_budget
                            else:
                                params = recommended_params
                                budget = recommended_time_budget

                            run_result = (
                                _run_classification_validation_request(
                                    dataset_json=dataset_json,
                                    framework=framework,
                                    algorithm=algorithm,
                                    hyperparams=params,
                                    time_budget=budget,
                                    seed=seed_value
                                )
                            )

                            run_result[
                                "repeat_index"
                            ] = repeat_index

                            run_result[
                                "execution_order"
                            ] = (
                                1
                                if run_type == run_order[0]
                                else 2
                            )

                            pair_results[
                                run_type
                            ] = run_result

                            completed_jobs += 1

                            progress.progress(
                                completed_jobs / total_jobs
                            )

                        baseline_runs.append(
                            pair_results["baseline"]
                        )

                        recommended_runs.append(
                            pair_results["recommended"]
                        )

                    status_text.empty()
                    progress.empty()

                    repeated_summary = (
                        _summarize_repeated_classification_validation(
                            baseline_runs=baseline_runs,
                            recommended_runs=recommended_runs
                        )
                    )
                    repeated_interpretation = (
                        _interpret_repeated_classification_validation(
                            repeated_summary
                        )
                    )

                    repeated_summary[
                        "empirical_interpretation"
                    ] = repeated_interpretation

                    repeated_summary[
                        "recommendation"
                    ] = recommendation.copy()

                    repeated_summary[
                        "seeds"
                    ] = seeds

                    st.session_state[
                        "classification_repeated_validation"
                    ] = repeated_summary

                    st.session_state[
                        "classification_repeated_validation_error"
                    ] = None

                    st.rerun()

                except Exception as e:

                    st.session_state[
                        "classification_repeated_validation_error"
                    ] = str(e)

                    st.error(
                        f"Repeated validation failed: {e}"
                    )
            repeated_error = st.session_state.get(
                "classification_repeated_validation_error"
            )

            if repeated_error:
                st.error(
                    f"Latest repeated-validation error: "
                    f"{repeated_error}"
                )

            repeated_bundle = st.session_state.get(
                "classification_repeated_validation"
            )

            if repeated_bundle:

                st.markdown(
                    "#### Repeated Validation Summary"
                )

                st.caption(
                    f"Paired runs: "
                    f"{repeated_bundle.get('n_repeats', 0)}"
                )

                summary = repeated_bundle.get(
                    "summary",
                    {}
                )

                summary_rows = []

                metric_labels = {
                    "Accuracy": "Accuracy",
                    "F1 Score": "F1 Score",
                    "CO2 Emission": "CO₂ Emission",
                    "Energy Consumption": "Energy Consumption",
                    "ROC-AUC": "ROC-AUC",
                    "Log Loss": "Log Loss",
                }

                for metric_key, label in metric_labels.items():

                    metric_summary = (
                        summary.get(
                            metric_key,
                            {}
                        )
                    )

                    baseline_mean = (
                        metric_summary.get(
                            "baseline_mean"
                        )
                    )

                    baseline_std = (
                        metric_summary.get(
                            "baseline_std"
                        )
                    )

                    recommended_mean = (
                        metric_summary.get(
                            "recommended_mean"
                        )
                    )

                    recommended_std = (
                        metric_summary.get(
                            "recommended_std"
                        )
                    )

                    mean_delta = (
                        metric_summary.get(
                            "mean_delta"
                        )
                    )

                    delta_std = (
                        metric_summary.get(
                            "delta_std"
                        )
                    )

                    summary_rows.append({
                        "Metric": label,

                        "Baseline Mean": (
                            baseline_mean
                        ),

                        "Baseline SD": (
                            baseline_std
                        ),

                        "Baseline Median": (
                            metric_summary.get(
                                "baseline_median"
                            )
                        ),

                        "Baseline IQR": (
                            metric_summary.get(
                                "baseline_iqr"
                            )
                        ),

                        "Recommended Mean": (
                            recommended_mean
                        ),

                        "Recommended SD": (
                            recommended_std
                        ),

                        "Recommended Median": (
                            metric_summary.get(
                                "recommended_median"
                            )
                        ),

                        "Recommended IQR": (
                            metric_summary.get(
                                "recommended_iqr"
                            )
                        ),

                        "Mean Δ": (
                            mean_delta
                        ),

                        "Median Δ": (
                            metric_summary.get(
                                "median_delta"
                            )
                        ),

                        "Δ SD": (
                            delta_std
                        ),
                        "Outliers (Baseline)": (
                            metric_summary.get(
                                "baseline_outlier_count",
                                0
                            )
                        ),
                        "Outliers (Recommended)": (
                            metric_summary.get(
                                "recommended_outlier_count",
                                0
                            )
                        ),
                        "Pair Consistency": (
                            f"{metric_summary.get('improved_count', 0)}/"
                            f"{metric_summary.get('total_pairs', 0)} improved"
                        ),                        
                    })
                repeated_df = pd.DataFrame(
                    summary_rows
                )
                empirical = (
                    repeated_bundle.get(
                        "empirical_interpretation",
                        {}
                    )
                )

                predictive_status = (
                    empirical.get(
                        "predictive_status",
                        "unknown"
                    )
                )

                sustainability_status = (
                    empirical.get(
                        "sustainability_status",
                        "unknown"
                    )
                )

                overall_status = (
                    empirical.get(
                        "overall_status",
                        "inconclusive"
                    )
                )

                st.markdown(
                    "#### Empirical Validation Outcome"
                )
                status_styles = {
                    "improved": (
                        "sml-status-good",
                        "↑ Improved"
                    ),
                    "worsened": (
                        "sml-status-bad",
                        "↓ Worsened"
                    ),
                    "mixed": (
                        "sml-status-warning",
                        "↕ Mixed"
                    ),
                    "no_clear_change": (
                        "sml-status-neutral",
                        "— No Clear Change"
                    ),
                    "uncertain_due_to_variability": (
                        "sml-status-warning",
                        "⚠ Uncertain"
                    ),
                    "unknown": (
                        "sml-status-neutral",
                        "Unknown"
                    ),
                }

                pred_class, pred_text = (
                    status_styles.get(
                        predictive_status,
                        status_styles["unknown"]
                    )
                )

                sust_class, sust_text = (
                    status_styles.get(
                        sustainability_status,
                        status_styles["unknown"]
                    )
                )

                col_pred, col_sust = st.columns(
                    2,
                    gap="medium"
                )

                with col_pred:
                    st.markdown(
                        f"""
                        <div class="
                            sml-status-card
                            {pred_class}
                        ">
                            <div class="
                                sml-status-label
                            ">
                                Predictive Effect
                            </div>

                            <div class="
                                sml-status-value
                            ">
                                {pred_text}
                            </div>

                            <div class="
                                sml-card-subtitle
                            ">
                                Based on repeated paired runs
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col_sust:
                    st.markdown(
                        f"""
                        <div class="
                            sml-status-card
                            {sust_class}
                        ">
                            <div class="
                                sml-status-label
                            ">
                                Sustainability Effect
                            </div>

                            <div class="
                                sml-status-value
                            ">
                                {sust_text}
                            </div>

                            <div class="
                                sml-card-subtitle
                            ">
                                CO₂ and energy measurements
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                overall_messages = {
                    "supported_on_both_objectives": (
                        "✅ Across these paired runs, the recommended "
                        "configuration showed practical improvement in "
                        "both predictive and sustainability measures."
                    ),

                    "predictive_improvement_no_clear_sustainability_change": (
                        "📈 Across these paired runs, predictive quality "
                        "improved while sustainability showed no clear "
                        "practical change."
                    ),

                    "predictive_improvement_sustainability_uncertain": (
                        "⚠️ Predictive quality improved across the paired "
                        "runs, but sustainability measurements were too "
                        "variable for a stable conclusion."
                    ),

                    "tradeoff": (
                        "⚖️ The repeated experiment produced a trade-off: "
                        "predictive quality improved while sustainability "
                        "worsened."
                    ),

                    "not_supported": (
                        "❌ The repeated experiment did not support the "
                        "proposed change under the current practical "
                        "effect criteria."
                    ),

                    "inconclusive": (
                        "➖ The repeated experiment did not produce a "
                        "clear overall empirical outcome."
                    ),
                }

                st.info(
                    overall_messages.get(
                        overall_status,
                        overall_messages[
                            "inconclusive"
                        ]
                    )
                )
                st.markdown(
                    "#### 📈 Improvement Consistency"
                )

                st.caption(
                    "Percentage of matched runs in which the "
                    "recommended configuration improved each metric."
                )

                consistency_labels = {
                    "Accuracy": "Accuracy",
                    "F1 Score": "F1",
                    "ROC-AUC": "ROC-AUC",
                    "Log Loss": "Log Loss",
                    "CO2 Emission": "CO₂",
                    "Energy Consumption": "Energy",
                }

                consistency_rows = []

                for (
                    metric_key,
                    display_label
                ) in consistency_labels.items():

                    metric_info = (
                        summary.get(
                            metric_key,
                            {}
                        )
                    )

                    improved = int(
                        metric_info.get(
                            "improved_count",
                            0
                        )
                    )

                    total = int(
                        metric_info.get(
                            "total_pairs",
                            0
                        )
                    )

                    if total <= 0:
                        continue

                    improvement_rate = (
                        improved
                        / total
                        * 100.0
                    )

                    metric_group = (
                        "Predictive"
                        if metric_key in [
                            "Accuracy",
                            "F1 Score",
                            "ROC-AUC",
                            "Log Loss",
                        ]
                        else "Sustainability"
                    )

                    consistency_rows.append({
                        "Metric": display_label,
                        "Improvement Rate": (
                            improvement_rate
                        ),
                        "Runs": (
                            f"{improved}/{total}"
                        ),
                        "Group": metric_group,
                    })

                if consistency_rows:

                    consistency_df = (
                        pd.DataFrame(
                            consistency_rows
                        )
                    )

                    consistency_df = (
                        consistency_df
                        .sort_values(
                            "Improvement Rate",
                            ascending=True
                        )
                    )

                    consistency_fig = px.bar(
                        consistency_df,
                        x="Improvement Rate",
                        y="Metric",
                        orientation="h",
                        color="Group",
                        text="Runs",
                        color_discrete_map={
                            "Predictive": "#3b82f6",
                            "Sustainability": "#10b981",
                        },
                    )

                    consistency_fig.add_vline(
                        x=50,
                        line_dash="dash",
                        line_color="#94a3b8",
                        annotation_text="50%"
                    )

                    consistency_fig.update_traces(
                        textposition="outside"
                    )

                    consistency_fig.update_layout(
                        height=360,
                        xaxis=dict(
                            title=(
                                "Paired runs improved"
                            ),
                            range=[0, 105],
                            ticksuffix="%"
                        ),
                        yaxis_title=None,
                        legend_title=None,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        margin=dict(
                            l=15,
                            r=55,
                            t=45,
                            b=25
                        ),
                    )

                    st.plotly_chart(
                        consistency_fig,
                        use_container_width=True,
                        config={
                            "displayModeBar": False
                        }
                    )
                with st.expander(
                    "📋 Detailed repeated-validation statistics",
                    expanded=False
                ):

                    st.caption(
                        "Research inspection view: mean, standard "
                        "deviation, median, IQR, paired deltas, "
                        "outliers and pair consistency."
                    )

                    st.dataframe(
                        repeated_df,
                        hide_index=True,
                        use_container_width=True
                    )
                variability_info = (
                    empirical.get(
                        "sustainability_variability",
                        {}
                    )
                )

                if variability_info.get(
                    "unstable",
                    False
                ):
                    reasons = (
                        variability_info.get(
                            "reasons",
                            []
                        )
                    )

                    st.warning(
                        "Sustainability measurements are unstable across "
                        "the repeated runs. Mean values may therefore be "
                        "misleading."
                    )

                    if reasons:
                        with st.expander(
                            "Why sustainability is marked uncertain"
                        ):
                            for reason in reasons:
                                st.write(
                                    f"• {reason}"
                                )
                with st.expander(
                    "🔍 Sustainability outlier diagnostics"
                ):
                    for metric_key in [
                        "CO2 Emission",
                        "Energy Consumption",
                    ]:

                        metric_info = (
                            summary.get(
                                metric_key,
                                {}
                            )
                        )

                        outlier_count = (
                            metric_info.get(
                                "recommended_outlier_count",
                                0
                            )
                        )

                        if outlier_count > 0:
                            st.write(
                                f"**{metric_key}**"
                            )

                            st.write(
                                "Recommended-run outlier values:",
                                metric_info.get(
                                    "recommended_outlier_values",
                                    []
                                )
                            )

                            st.write(
                                "Run indices:",
                                [
                                    index + 1
                                    for index in metric_info.get(
                                        "recommended_outlier_indices",
                                        []
                                    )
                                ]
                            )

                rec = repeated_bundle.get(
                    "recommendation",
                    {}
                )

                st.markdown(
                    "#### Repeated experiment"
                )

                st.code(
                    f"{rec.get('control')}: "
                    f"{rec.get('current_value')} → "
                    f"{rec.get('suggested_value')}"
                )

                with st.expander(
                    "ℹ️ How repeated validation works"
                ):
                    st.write(
                        "Baseline and recommended configurations "
                        "are evaluated using the same experimental "
                        "seeds. Mean and standard deviation summarize "
                        "average behavior, while median and IQR provide "
                        "a more robust view when unusual runtime or "
                        "sustainability measurements occur."
                    )

                    st.caption(
                        "The empirical outcome is calculated "
                        "deterministically in Python and is not "
                        "generated by the LLM."
                    )
                with st.expander(
                    "🔍 Repeated validation details"
                ):
                    st.json(
                        repeated_bundle
                    )
            # SECTION 6: FOLLOW-UP QUESTIONS (OPTIONAL FUTURE ENHANCEMENT)
            st.markdown("---")
            st.markdown("### Ask Follow-up Questions")
            st.caption("(Optional: Ask clarifying questions about the explanation)")
            # Initialize follow-up Q&A history if not exists
            if "followup_history" not in st.session_state:
                st.session_state["followup_history"] = []
            
            # Display previous follow-up Q&A
            if st.session_state.get("followup_history"):
                st.markdown("#### 💬 Conversation History")
                for qa in st.session_state["followup_history"]:
                    with st.chat_message("user"):
                        st.write(qa["question"])
                    with st.chat_message("assistant"):
                        st.write(qa["answer"])
            # New follow-up question input
            followup = st.text_input(
                "Ask a follow-up question (leave blank to skip)",
                key=f"audience_followup_{st.session_state.get('followup_history_len', 0)}",
                placeholder="e.g., 'Why is chest pain the most important feature?'"
            )
            
            if followup and st.button("Get Answer", key="followup_btn"):
                audience_type = st.session_state.get("audience_selected_type", "Normal User")
                domain = st.session_state.get("selected_domain", "Medical")
                current_explanation = st.session_state.get("explanation_results", {}).get(audience_type, {}).get("explanation", "")
                
                if not current_explanation:
                    st.warning("⚠️ Please generate an explanation first before asking follow-up questions.")
                else:
                    with st.spinner(f"🔄 Getting answer from {audience_type} perspective..."):
                        try:
                            # Get features for context
                            features_context = st.session_state.get("current_features_for_answer", [])
                            features_str = "\n".join(
                                [f"- {f['name']}: {f['importance']:.3f}" 
                                for f in features_context]
                            ) if features_context else "No features available"
                            # Prepare context for follow-up
                            followup_prompt = f"""You are responding to a follow-up question based on a previous explanation.

    AUDIENCE: {audience_type}
    DOMAIN: {domain}

    PREVIOUS EXPLANATION:
    {current_explanation}

    USER FOLLOW-UP QUESTION:
    {followup}

    INSTRUCTION:
    Answer the follow-up question in the same tone and style as the previous explanation.
    Keep the answer concise (2-3 sentences max).
    Stay consistent with the audience type (simple for Normal User, technical for Data Scientist, professional for Domain Expert)."""
                            
                            # Call LLM for follow-up answer
                            use_openai = bool(os.getenv("OPENAI_API_KEY"))
                            openai_api_key = os.getenv("OPENAI_API_KEY") if use_openai else None
                            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
                            ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:4b")
                            
                            if use_openai and openai_api_key:
                                # Use OpenAI
                                from openai import OpenAI
                                client = OpenAI(api_key=openai_api_key)
                                response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "user", "content": followup_prompt}
                                    ],
                                    temperature=0.3,
                                    max_tokens=300
                                )
                                followup_answer = response.choices[0].message.content.strip()
                            else:
                                # Use Ollama
                                response = requests.post(
                                    f"{ollama_url}/api/generate",
                                    json={
                                        "model": ollama_model,
                                        "prompt": followup_prompt,
                                        "stream": False,
                                        "options": {"temperature": 0.3, "num_predict": 300}
                                    },
                                    timeout=600
                                )
                                response.raise_for_status()
                                followup_answer = (response.json().get("response") or "").strip()
                            if followup_answer:
                                # Store in history
                                st.session_state["followup_history"].append({
                                    "question": followup,
                                    "answer": followup_answer
                                })
                                # Clear input and refresh
                                st.session_state["followup_history_len"] = len(st.session_state["followup_history"])
                                st.success("✅ Answer generated!")
                                st.rerun()
                            else:
                                st.error("No answer received from the model.")
                        except Exception as e:
                            st.error(f"Error generating follow-up answer: {str(e)}")       
    elif st.session_state.get("show_tradeoff_panel", False):
        st.title("♻️ Sustainability-Accuracy Trade-off Explorer")
        st.caption("Find the best model for your needs: maximize accuracy, sustainability, or balance both")
        results = st.session_state.get("automl_results", {}) or {}
        if not results:
            st.warning("No AutoML results found. Please run AutoML first.")
        else:
            # SECTION 1: INTERACTIVE PARETO FRONTIER
            st.markdown("### 📊 Pareto Frontier")
            st.caption("Stars indicate optimal models (not dominated by others)")
            
            fig = create_tradeoff_chart(results)
            st.plotly_chart(fig, use_container_width=True)
            # SECTION 2: PREFERENCE SLIDER & RECOMMENDATION
            st.markdown("### 🎯 Find Your Best Model")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                sustainability_weight = st.slider(
                    "What's your priority?",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get("sustainability_weight", 0.5),
                    step=0.1,
                    format="%.1f",
                    help="0.0 = Accuracy First | 0.5 = Balanced | 1.0 = Sustainability First"
                )
                st.session_state["sustainability_weight"] = sustainability_weight
            with col2:
                if sustainability_weight < 0.3:
                    st.metric("Priority", "Accuracy 🎯")
                elif sustainability_weight > 0.7:
                    st.metric("Priority", "Green ♻️")
                else:
                    st.metric("Priority", "Balanced ⚖️")
            # Get recommendation
            recommendation = recommend_model(results, sustainability_weight)
            if recommendation:
                col1, col2, col3 = st.columns(3)
                col1.metric("Recommended Model", recommendation['Model'])
                col2.metric("Accuracy", f"{recommendation['Accuracy']:.4f}")
                col3.metric("CO2 (µg)", f"{recommendation['CO2']:.6f}")
                
                st.info(f"✅ {recommendation['Reason']}")
            # SECTION 3: PARETO MODELS COMPARISON TABLE
            st.markdown("### 📋 Pareto Optimal Models")
            pareto_df = calculate_pareto_frontier(results)
            
            if not pareto_df.empty:
                display_df = pd.DataFrame({
                    'Model': pareto_df['Model'],
                    'Accuracy': pareto_df['Accuracy'].apply(lambda x: f"{x:.4f}"),
                    'CO2 (µg)': pareto_df['CO2'].apply(lambda x: f"{x:.6f}"),
                    'Energy (µWh)': pareto_df['Energy'].apply(lambda x: f"{x:.6f}"),
                    'Time (s)': pareto_df['Time'].apply(lambda x: f"{x:.2f}")
                })
                st.dataframe(display_df, use_container_width=True)
            # SECTION 4: ALL MODELS COMPARISON
            st.markdown("### 📊 All Models Comparison")
            all_models_df = create_comparison_table(results)
            st.dataframe(all_models_df, use_container_width=True)
            # SECTION 5: CO2 SAVINGS CALCULATOR
            st.markdown("### 💰 CO2 Savings Calculator")
            st.caption("Compare two models to see sustainability benefits")
            model_names = list(results.keys())
            col1, col2 = st.columns(2)
            with col1:
                baseline = st.selectbox(
                    "Current Model",
                    options=model_names,
                    key="baseline_model"
                )
            with col2:
                comparison = st.selectbox(
                    "Alternative Model",
                    options=model_names,
                    key="comparison_model"
                )
            if baseline != comparison and baseline and comparison:
                savings = calculate_co2_savings(results, baseline, comparison) 
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "CO2 Saved (µg)",
                    f"{savings['co2_saved']:.6f}",
                    delta=f"{savings['savings_percentage']:.2f}%"
                )
                col2.metric(
                    "Accuracy Change",
                    f"{savings['accuracy_change']:+.2f}%"
                )
                if savings['accuracy_change'] >= 0:
                    col3.success("✅ Win-Win: Better sustainability & accuracy!")
                else:
                    acc_loss = abs(savings['accuracy_change'])
                    col3.info(f"⚖️ Trade-off: {acc_loss:.2f}% less accuracy")
    elif st.session_state.get("show_counterfactual_panel", False):
        st.title("🔮 What-If Counterfactual Explanations")
        st.caption("Discover what changes would flip the model's prediction")
        results = st.session_state.get("automl_results", {}) or {}
        X_train = st.session_state.get("X_train", None)
        y_train = st.session_state.get("y_train", None)
        if not results or X_train is None:
            st.warning("⚠️ Please run AutoML first with training data available.")
        else:
            # Get best model
            best_model_name = max(results.keys(), key=lambda x: results[x].get('Accuracy', 0))
            best_model = st.session_state.get("best_model")
            if best_model is None:
                st.error("Model object not found in session state.")
            else:
                # Get feature ranges
                feature_ranges = get_feature_ranges(X_train)
                # Select a sample to explain
                st.markdown("### 📊 Step 1: Select Sample to Explain")
                sample_idx = st.slider(
                    "Choose a sample:",
                    min_value=0,
                    max_value=len(X_train) - 1,
                    value=0,
                    help="Select a training sample to generate counterfactual for"
                )
                sample = X_train.iloc[sample_idx]
                actual_label = y_train.iloc[sample_idx] if y_train is not None else None
                # Show original prediction
                st.markdown("#### Original Sample")
                col1, col2 = st.columns(2)
                try:
                    original_pred = best_model.predict([sample.values])[0]
                    col1.metric("Model Prediction", "Disease ✓" if original_pred == 1 else "No Disease ✗")
                    if actual_label is not None:
                        col2.metric("Actual Label", "Disease ✓" if actual_label == 1 else "No Disease ✗")
                except:
                    st.error("Error making prediction")
                # Show sample features
                st.dataframe(pd.DataFrame({'Feature': sample.index, 'Value': sample.values}), hide_index=True)
                # Find minimal changes
                st.markdown("### 🎯 Step 2: Minimal Changes to Flip Prediction")
                changes = find_minimal_changes(sample, best_model, feature_ranges)
                if changes:
                    explanation = generate_counterfactual_explanation(
                        changes,
                        sample.index.tolist(),
                        audience=st.session_state.get("audience_selected_type", "Normal User")
                    )
                    st.info(explanation)
                    # Show changes table
                    st.markdown("#### Changes Required")
                    changes_df = pd.DataFrame([
                        {
                            'Feature': feature,
                            'Original': f"{change['original']:.4f}" if change['type'] == 'numeric' else change['original'],
                            'Counterfactual': f"{change['counterfactual']:.4f}" if change['type'] == 'numeric' else change['counterfactual']
                        }
                        for feature, change in changes.items()
                    ])
                    st.dataframe(changes_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No simple changes found to flip prediction.")
                # Interactive mode
                st.markdown("### 🎚️ Step 3: Interactive Exploration")
                st.caption("Adjust features and see how prediction changes")
                with st.form("counterfactual_form"):
                    modified_sample, new_pred = create_interactive_counterfactual(
                        sample, best_model, feature_ranges, X_train
                    )
                    submit = st.form_submit_button("Update Prediction")
                if submit and new_pred is not None:
                    st.markdown("#### Prediction After Changes")
                    col1, col2 = st.columns(2)
                    col1.metric("New Prediction", "Disease ✓" if new_pred == 1 else "No Disease ✗")
                    col2.metric("Changed", "Yes ✓" if new_pred != original_pred else "No")
                    # Show comparison
                    st.markdown("#### Feature Changes")
                    comparison_df = create_counterfactual_comparison_table(sample, modified_sample)
                    if not comparison_df.empty:
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No features changed")
                    st.markdown("### 🎚️ Step 4: Top Features Impact Analysis")
                    st.caption("Adjust the most impactful features and see prediction changes in real-time")
                    modified_sample, metrics = interactive_top_features_panel(
                        sample, best_model, feature_ranges, n_features=4
                    )
    elif st.session_state.get("show_fairness_panel", False):

        st.title("⚖️ Fairness & Bias Detection")

        # =====================================================================
        # CLASSIFICATION FAIRNESS
        # =====================================================================
        if not task.startswith("Clustering"):
            st.caption("Fairness is evaluated on the held-out validation set using the actual AutoML winning model.")

            raw_json = st.session_state.get("last_uploaded_dataset_json")
            if not raw_json:
                st.warning("⚠️ Please upload a dataset and run classification AutoML first.")
            else:
                try:
                    raw_cls = pd.read_json(StringIO(raw_json))
                    target_col = raw_cls.columns[-1]
                    raw_features = raw_cls.drop(columns=[target_col])

                    # IMPORTANT: demographic groups come from the ORIGINAL data,
                    # not standardized/label-encoded model features.
                    demographic_cols = detect_demographic_columns(raw_features)
                    if not demographic_cols:
                        demographic_cols = list(raw_features.columns)
                        st.info("No demographic column was automatically detected. Please select it manually.")

                    demographic_col = st.selectbox(
                        "Select sensitive / demographic attribute:",
                        demographic_cols,
                        key="fairness_cls_demographic_col"
                    )

                    target_values = list(pd.unique(raw_cls[target_col].dropna()))
                    positive_label = None
                    if len(target_values) == 2:
                        positive_label = st.selectbox(
                            "Positive outcome (used for selection rate, TPR/FPR/FNR):",
                            target_values,
                            key="fairness_positive_label"
                        )
                    elif len(target_values) > 2:
                        st.info("Multiclass target detected. Selection-rate/TPR/FPR metrics require a single positive class, so only group performance gaps will be used unless you choose a binary target.")

                    if st.button("🔎 Run Classification Fairness Analysis", key="run_cls_fairness"):

                        # Convert NumPy scalar values (e.g. np.int64) to native Python values
                        # so that requests/json can serialize them.
                        if isinstance(positive_label, np.generic):
                            positive_label = positive_label.item()

                        # Make sure the column name is a normal Python string
                        demographic_col = str(demographic_col)

                        payload = {
                            "demographic_col": demographic_col,
                            "positive_label": positive_label,
                            "alpha": 0.05,
                        }

                        with st.spinner(
                            "Evaluating fairness on held-out data using the actual AutoML winner..."
                        ):
                            response = requests.post(
                                f"{BACKEND_URL}/fairness_analysis",
                                json=payload,
                                timeout=120
                            )

                        response.raise_for_status()
                        fairness_result = response.json()

                        if fairness_result.get("status") != "success":
                            st.error(
                                fairness_result.get(
                                    "message",
                                    "Fairness analysis failed."
                                )
                            )
                        else:
                            st.session_state["classification_fairness_result"] = fairness_result

                    result = st.session_state.get("classification_fairness_result")
                    if result:
                        bias = result["bias"]
                        metrics = result["metrics"]
                        model_info = result["model"]

                        st.success(
                            f"Evaluated **{model_info['framework']} / {model_info['algorithm']}** "
                            f"(AutoML accuracy: {model_info['accuracy']:.2%}) on **{result['sample_count']} held-out samples**."
                        )
                        st.markdown("### 📊 Fairness Metrics Heatmap")
                        
                        # Get metrics
                        fairness_metrics_summary = result.get('fairness_metrics_summary', {})
                        
                        # Build heatmap data
                        metrics_list = ['Demographic Parity', 'Equalized Odds', 'Predictive Parity', 
                                       'Calibration', 'Accuracy Parity', 'FNR/FPR Parity']
                        values = [fairness_metrics_summary.get(m, 0) for m in metrics_list]
                        
                        # Color coding: 0.8-1.0 = Green, 0.6-0.8 = Yellow, <0.6 = Red
                        def get_status(val):
                            if val >= 0.80:
                                return "🟢 PASS"
                            elif val >= 0.60:
                                return "🟡 WARNING"
                            else:
                                return "🔴 FAIL"
                        
                        # Display as table
                        heatmap_data = {
                            'Metric': metrics_list,
                            'Score': [f"{v:.2f}" for v in values],
                            'Status': [get_status(v) for v in values]
                        }                        
                        heatmap_df = pd.DataFrame(heatmap_data)
                        st.dataframe(heatmap_df, use_container_width=True, hide_index=True)
                        # Interactive visualization
                        fig = go.Figure(data=go.Bar(
                            x=metrics_list,
                            y=values,
                            marker=dict(
                                color=values,
                                colorscale='RdYlGn',
                                cmin=0,
                                cmax=1,
                                colorbar=dict(title="Score")
                            ),
                            text=[f"{v:.2f}" for v in values],
                            textposition='auto',
                        ))
                        
                        fig.update_layout(
                            title="Fairness Metrics Performance",
                            xaxis_title="Fairness Metric",
                            yaxis_title="Score (0-1)",
                            height=400,
                            showlegend=False,
                            yaxis=dict(range=[0, 1])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Metric explanations (expandable)
                        st.markdown("### 📖 Metric Explanations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            with st.expander("🟢 Demographic Parity", expanded=False):
                                st.write("**Definition:** P(ŷ=1|A) = P(ŷ=1|B) across groups")
                                st.write("**Why it matters:** Same % selected for all groups (no discrimination in selection rate)")
                                st.write(f"**Your score:** {values[0]:.2f} - {get_status(values[0])}")
                            
                            with st.expander("🟡 Equalized Odds", expanded=False):
                                st.write("**Definition:** TPR & FPR equal across groups")
                                st.write("**Why it matters (Medical):** Model has same error rates for all patient groups")
                                st.write(f"**Your score:** {values[1]:.2f} - {get_status(values[1])}")
                            
                            with st.expander("🟢 Predictive Parity", expanded=False):
                                st.write("**Definition:** P(y=1|ŷ=1) equal across groups (precision parity)")
                                st.write("**Why it matters:** Among positive predictions, % correct is same for all groups")
                                st.write(f"**Your score:** {values[2]:.2f} - {get_status(values[2])}")
                        
                        with col2:
                            with st.expander("🟠 Calibration", expanded=False):
                                st.write("**Definition:** Model confidence matches reality for all groups")
                                st.write("**Why it matters (Medical):** When model says 70% confident, it's actually ~70% correct for all groups")
                                st.write(f"**Your score:** {values[3]:.2f} - {get_status(values[3])}")
                            
                            with st.expander("🟢 Accuracy Parity", expanded=False):
                                st.write("**Definition:** Accuracy equal across groups")
                                st.write("**Why it matters:** Model works equally well for all patient demographics")
                                st.write(f"**Your score:** {values[4]:.2f} - {get_status(values[4])}")
                            
                            with st.expander("🔴 FNR/FPR Parity", expanded=False):
                                st.write("**Definition:** False negative & false positive rates equal across groups")
                                st.write("**Why it matters (Medical - CRITICAL):** Don't miss diagnoses more for any patient group")
                                st.write(f"**Your score:** {values[5]:.2f} - {get_status(values[5])}")
                        
                        # Root cause analysis
                        st.markdown("### 🔍 Root Cause Analysis")
                        worst_metric_idx = values.index(min(values))
                        worst_metric = metrics_list[worst_metric_idx]
                        worst_score = values[worst_metric_idx]
                        
                        if worst_score < 0.60:
                            st.error(f"🔴 **{worst_metric} is CRITICAL** ({worst_score:.2f})")
                            st.info("**Action required before deployment:**\n• Review group representation in training data\n• Check for proxy features\n• Consider fairness-aware learning algorithms")
                        elif worst_score < 0.80:
                            st.warning(f"🟡 **{worst_metric} needs improvement** ({worst_score:.2f})")
                            st.info("**Recommended actions:**\n• Oversample underrepresented groups\n• Add fairness regularization\n• Use stratified cross-validation")
                        
                        # Metrics by group table
                        st.markdown("### 📈 Metrics by Group (Detailed)")
                        metrics_df = create_fairness_comparison_df(metrics)
                        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                        st.markdown("### 📏 Disparity Summary")
                        disparity_df = pd.DataFrame([
                            {"Metric": "Accuracy gap", "Disparity": bias.get("accuracy_disparity", 0.0), "Severity": get_severity_color(bias.get("accuracy_disparity", 0.0))},
                            {"Metric": "Selection-rate gap", "Disparity": bias.get("selection_disparity", 0.0), "Severity": get_severity_color(bias.get("selection_disparity", 0.0))},
                            {"Metric": "TPR gap", "Disparity": bias.get("tpr_disparity", 0.0), "Severity": get_severity_color(bias.get("tpr_disparity", 0.0))},
                            {"Metric": "FPR gap", "Disparity": bias.get("fpr_disparity", 0.0), "Severity": get_severity_color(bias.get("fpr_disparity", 0.0))},
                            {"Metric": "FNR gap", "Disparity": bias.get("fnr_disparity", 0.0), "Severity": get_severity_color(bias.get("fnr_disparity", 0.0))},
                        ])
                        disparity_df["Disparity"] = disparity_df["Disparity"].map(lambda x: f"{x:.2%}")
                        st.dataframe(disparity_df, use_container_width=True, hide_index=True)

                        significance = result.get("significance", {})
                        st.markdown("### 🧪 Statistical Check")
                        if significance.get("p_value") is not None:
                            if significance.get("is_significant"):
                                st.warning(significance.get("interpretation", "Statistically significant group difference detected."))
                            else:
                                st.success(significance.get("interpretation", "No statistically significant difference detected."))
                        else:
                            st.info(significance.get("interpretation", "Statistical test unavailable."))

                        st.markdown("### 💡 Recommendations")
                        if bias["has_bias"]:
                            st.error(
                                "**Fairness concerns detected.** Review group representation, possible proxy features, "
                                "thresholds, and error rates before deployment. Do not conclude that the model is fair "
                                "from accuracy alone."
                            )
                        else:
                            st.success(
                                "No configured fairness threshold was exceeded on this held-out evaluation. "
                                "Continue monitoring fairness after deployment and across new data distributions."
                            )

                        with st.expander("How is fairness calculated?", expanded=False):
                            st.write(
                                "The dashboard checks 7 key fairness metrics:\n\n"
                                "1. **Demographic Parity** - Selection rates equal?\n"
                                "2. **Equalized Odds** - TPR & FPR equal? (same error rates)\n"
                                "3. **Predictive Parity** - Precision equal? (among positive predictions)\n"
                                "4. **Calibration** - Confidence honest for all groups?\n"
                                "5. **Accuracy Parity** - Overall accuracy equal?\n"
                                "6. **FNR/FPR Parity** - False error rates equal? (CRITICAL for medical)\n"
                                "7. **Disparate Impact** - 80% rule met?\n\n"
                                "A held-out validation set is used for unbiased evaluation."
                            )
                            
                        # disparity_df["Disparity"] = disparity_df["Disparity"].map(lambda x: f"{x:.2%}")
                        st.dataframe(disparity_df, use_container_width=True, hide_index=True)

                        st.markdown("### 🔍 Root Cause Analysis")
                        sample_counts = {group: len(metrics[group].get('samples', [])) for group in metrics if isinstance(metrics[group], dict)}
                        smallest_group = min(sample_counts.values()) if sample_counts else 0
                        
                        if smallest_group < 10:
                            st.warning(f"⚠️ **Underrepresented groups detected** (smallest n={smallest_group})")
                            st.info("**Recommended fixes:**\n• Oversample minority groups in training\n• Use stratified cross-validation\n• Add demographic balancing constraints\n• **Expected fairness improvement: +10-15%**")
                        
                        if bias.get("tpr_disparity", 0) > 0.2:
                            st.warning("⚠️ **TPR Gap is high** - Model misses diagnoses more for some groups")
                            st.info("**Action:** Adjust decision threshold per group or use fairness-aware learning")
                        
                        if bias.get("fnr_disparity", 0) > 0.2:
                            st.error("🔴 **FNR Gap is critical** - False negatives are much higher for some groups")
                            st.info("**Action (Priority):** This is dangerous in medical domain. Retrain with FNR penalty or rebalance data")

                        significance = result.get("significance", {})
                        st.markdown("### 🧪 Statistical Check")
                        if significance.get("p_value") is not None:
                            if significance.get("is_significant"):
                                st.warning(significance.get("interpretation", "Statistically significant group difference detected."))
                            else:
                                st.success(significance.get("interpretation", "No statistically significant difference detected."))
                        else:
                            st.info(significance.get("interpretation", "Statistical test unavailable."))

                        st.markdown("### 💡 Recommendations")
                        if bias["has_bias"]:
                            st.error(
                                "**Fairness concerns detected.** Review group representation, possible proxy features, "
                                "thresholds, and error rates before deployment. Do not conclude that the model is fair "
                                "from accuracy alone."
                            )
                        else:
                            st.success(
                                "No configured fairness threshold was exceeded on this held-out evaluation. "
                                "Continue monitoring fairness after deployment and across new data distributions."
                            )

                        with st.expander("How is fairness calculated?", expanded=False):
                            st.write(
                                "The dashboard checks: **Accuracy gap, Selection-rate gap, TPR gap, FPR gap, FNR gap**, and "
                                "**Disparate Impact** (for binary classification). Medical domain prioritizes **Equalized Odds** "
                                "(same TPR/FPR) and **FNR Parity** (don't miss diagnoses more for any group). "
                                "A held-out validation set is used instead of training data for unbiased evaluation."
                            )
                except Exception as e:
                    st.error(f"Error analyzing classification fairness: {e}")
                    import traceback
                    with st.expander("Debug Info"):
                        st.code(traceback.format_exc())

        # =====================================================================
        # CLUSTERING FAIRNESS
        # =====================================================================
        else:
            st.caption("For clustering, fairness is measured from cluster membership distributions across groups, not accuracy/F1.")
            clu_res = st.session_state.get("clu_results", {})
            if not clu_res:
                st.warning("⚠️ Run clustering AutoML first.")
            else:
                valid_runs = [k for k, v in clu_res.items() if "error" not in v and v.get("labels") is not None]
                if not valid_runs:
                    st.warning("No clustering run contains usable cluster labels.")
                else:
                    selected_run = st.selectbox("Select clustering run:", valid_runs, key="fairness_clu_run")
                    run_data = clu_res[selected_run]
                    labels = np.asarray(run_data["labels"])

                    demographic_cols = detect_demographic_columns(df_clu)
                    if not demographic_cols:
                        demographic_cols = list(df_clu.columns)
                        st.info("No demographic column was automatically detected. Please select it manually.")
                    demographic_col = st.selectbox(
                        "Select sensitive / demographic attribute:",
                        demographic_cols,
                        key="fairness_clu_demographic_col"
                    )

                    if len(labels) != len(df_clu):
                        st.error("The stored cluster labels do not match the uploaded dataset length. Re-run clustering on the current dataset.")
                    elif st.button("🔎 Run Clustering Fairness Analysis", key="run_clu_fairness"):
                        try:
                            clustering_metrics = calculate_clustering_fairness(
                                labels, df_clu[demographic_col]
                            )
                            clustering_bias = detect_clustering_bias(clustering_metrics)
                            st.session_state["clustering_fairness_result"] = {
                                "metrics": clustering_metrics,
                                "bias": clustering_bias,
                                "run": selected_run,
                                "demographic_col": demographic_col,
                            }
                        except Exception as e:
                            st.error(f"Error analyzing clustering fairness: {e}")

                    result = st.session_state.get("clustering_fairness_result")
                    if result and result.get("run") == selected_run and result.get("demographic_col") == demographic_col:
                        bias = result["bias"]
                        metrics = result["metrics"]

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Largest Cluster Gap", f"{bias['max_disparity']:.2%}")
                        c2.metric("Status", bias["recommendation"])
                        ratio = bias.get("min_representation_ratio")
                        c3.metric("Min Representation Ratio", f"{ratio:.3f}" if ratio is not None else "N/A")

                        st.caption(
                            f"Run: **{selected_run}** · Sensitive attribute: **{demographic_col}** · "
                            f"Cluster assignments: **{len(labels)} samples**"
                        )

                        st.markdown("### 📈 Cluster Membership by Group")
                        cluster_df = create_clustering_fairness_df(metrics)
                        rate_cols = [c for c in cluster_df.columns if c.endswith(" Rate")]
                        for col in rate_cols:
                            cluster_df[col] = cluster_df[col].map(lambda x: f"{x:.2%}")
                        st.dataframe(cluster_df, use_container_width=True, hide_index=True)

                        st.markdown("### 💡 Recommendations")
                        if bias["has_bias"]:
                            st.error(
                                "**Unequal cluster representation detected.** Examine whether the clustering "
                                "features or preprocessing create systematically different cluster membership "
                                "across demographic groups."
                            )
                        else:
                            st.success(
                                "No configured cluster-representation threshold was exceeded. "
                                "This does not prove the clustering is universally fair."
                            )

                        with st.expander("How clustering fairness differs from classification", expanded=False):
                            st.write(
                                "Clustering has no ground-truth target in the usual unsupervised setting. "
                                "Therefore accuracy, precision, recall and F1 are not used. Instead, the dashboard "
                                "compares the probability of each demographic group being assigned to each cluster."
                            )

    elif st.session_state.get("show_feature_importance", False):
        #Ensure Main UI is hidden and Feature Importance is fully displayed
        st.subheader(" Feature Importance Analysis")

        results = st.session_state.automl_results
        feature_data = []
        # Extract feature importance data from results
        for algo, metrics in results.items():
            if "feature_importance" in metrics and metrics["feature_importance"]:
                for feature, importance in metrics["feature_importance"].items():
                    feature_data.append({
                        "Algorithm": algo,
                        "Feature": feature,
                        "Importance": float(importance)
                    })
        if feature_data:
            df_feature_importance = pd.DataFrame(feature_data).sort_values(by="Importance", ascending=False)

            # ✅ Add Framework Selection
            available_frameworks = list(set(algo.split("_")[0] for algo in df_feature_importance["Algorithm"]))
            selected_frameworks = st.multiselect(
                "Select Framework(s) to Display:", available_frameworks, default=available_frameworks,
                key="feature_framework"
            )
            # ✅ Add Top N Selection
            max_features = len(df_feature_importance)
            top_n = st.number_input(
                f"Select Number of Top Features (Max: {max_features}):",
                min_value=1, max_value=max_features, value=max_features, step=1, key="top_n_features"
            )
            # ✅ Filter Data by Selected Frameworks
            df_filtered = df_feature_importance[
                df_feature_importance["Algorithm"].str.startswith(tuple(selected_frameworks))]
            # ✅ Show only Top N most important features
            df_filtered = df_filtered.sort_values(by="Importance", ascending=False).head(top_n)
            # ✅ Remove Index Column
            df_filtered = df_filtered.reset_index(drop=True)
            # ✅ Display the Filtered Table
            st.dataframe(df_filtered, hide_index=True)
            # ✅ Feature Importance Bar Chart
            fig = px.bar(
                df_filtered, x="Feature", y="Importance", color="Algorithm",
                barmode="group", title="Feature Importance Across Models"
            )
            fig.update_layout(
                        xaxis=dict(
                            title=dict(font=dict(size=14, color="black", family="Arial Black")),
                            tickfont=dict(size=14, color="black", family="Arial")
                        ),
                        yaxis=dict(
                            title=dict(font=dict(size=14, color="black", family="Arial Black")),
                            tickfont=dict(size=14, color="black", family="Times New Roman")
                        )
                    )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No feature importance data found for the selected models.")
    # Comparison View: Show metrics over time budget history
    elif st.session_state.get("show_comparison", False):
        st.subheader("Metric comparison")

        comparisons = st.session_state.get("time_budget_comparisons", [])
        if not comparisons:
            st.warning("No time budget comparisons available yet.")
        else:
            comparison_records = []
            for entry in comparisons:
                tb = entry["time_budget"]
                for algo, metrics in entry["results"].items():
                    if "error" not in metrics:
                        time_budget_key = f"{tb}_{algo}"
                        color_map_time_budget = {
        "10 seconds_FLAML_RF": "blue",
        "30 seconds_FLAML_RF": "blue",
        "60 seconds_FLAML_RF": "blue",
        "120 seconds_FLAML_RF": "blue",
        "10 seconds_FLAML_XGBoost": "red",
        "30 seconds_FLAML_XGBoost": "red",
        "60 seconds_FLAML_XGBoost": "red",
        "120 seconds_FLAML_XGBoost": "red",
        "10 seconds_FLAML_LightGBM": "green",
        "30 seconds_FLAML_LightGBM": "green",
        "60 seconds_FLAML_LightGBM": "green",
        "120 seconds_FLAML_LightGBM": "green",
        "10 seconds_FLAML_Extra Trees": "yellow",
        "30 seconds_FLAML_Extra Trees": "yellow",
        "60 seconds_FLAML_Extra Trees": "yellow",
        "120 seconds_FLAML_Extra Trees": "yellow",
        "10 seconds_FLAML_KNN": "purple",
        "30 seconds_FLAML_KNN": "purple",
        "60 seconds_FLAML_KNN": "purple",
        "120 seconds_FLAML_KNN": "purple",
        "10 seconds_FLAML_Logistic Regression": "orange",
        "30 seconds_FLAML_Logistic Regression": "orange",
        "60 seconds_FLAML_Logistic Regression": "orange",
        "120 seconds_FLAML_Logistic Regression": "orange",
        "10 seconds_H2O_Naive Bayes": "gray",
        "30 seconds_H2O_Naive Bayes": "gray",
        "60 seconds_H2O_Naive Bayes": "gray",
        "120 seconds_H2O_Naive Bayes": "gray",
        "10 seconds_H2O_GBM": "black",
        "30 seconds_H2O_GBM": "black",
        "60 seconds_H2O_GBM": "black",
        "120 seconds_H2O_GBM": "black",
        "10 seconds_H2O_GLM": "pink",
        "30 seconds_H2O_GLM": "pink",
        "60 seconds_H2O_GLM": "pink",
        "120 seconds_H2O_GLM": "pink",
        "10 seconds_H2O_Distributed RF": "cyan",
        "30 seconds_H2O_Distributed RF": "cyan",
        "60 seconds_H2O_Distributed RF": "cyan",
        "120 seconds_H2O_Distributed RF": "cyan",
        "10 seconds_H2O_XGBoost": "magenta",
        "30 seconds_H2O_XGBoost": "magenta",
        "60 seconds_H2O_XGBoost": "magenta",
        "120 seconds_H2O_XGBoost": "magenta",
        "10 seconds_MLJAR_Baseline": "lime",
        "10 seconds_MLJAR_Decision Tree": "olive",
        "10 seconds_MLJAR_RF": "teal",
        "10 seconds_MLJAR_XGBoost": "maroon",
        "10 seconds_MLJAR_Neural Network": "navy",
        "10 seconds_MLJAR_Extra Trees": "silver",
        "10 seconds_MLJAR_LightGBM": "gold",
        "10 seconds_MLJAR_SVM": "beige",
        "10 seconds_MLJAR_KNN": "brown",
        "30 seconds_MLJAR_Baseline": "lime",
        "30 seconds_MLJAR_Decision Tree": "olive",
        "30 seconds_MLJAR_RF": "teal",
        "30 seconds_MLJAR_XGBoost": "maroon",
        "30 seconds_MLJAR_Neural Network": "navy",
        "30 seconds_MLJAR_Extra Trees": "silver",
        "30 seconds_MLJAR_LightGBM": "gold",
        "30 seconds_MLJAR_SVM": "beige",
        "30 seconds_MLJAR_KNN": "brown",
        "60 seconds_MLJAR_Baseline": "lime",
        "60 seconds_MLJAR_Decision Tree": "olive",
        "60 seconds_MLJAR_RF": "teal",
        "60 seconds_MLJAR_XGBoost": "maroon",
        "60 seconds_MLJAR_Neural Network": "navy",
        "60 seconds_MLJAR_Extra Trees": "silver",
        "60 seconds_MLJAR_LightGBM": "gold",
        "60 seconds_MLJAR_SVM": "beige",
        "60 seconds_MLJAR_KNN": "brown",
        "120 seconds_MLJAR_Baseline": "lime",
        "120 seconds_MLJAR_Decision Tree": "olive",
        "120 seconds_MLJAR_RF": "teal",
        "120 seconds_MLJAR_XGBoost": "maroon",
        "120 seconds_MLJAR_Neural Network": "navy",
        "120 seconds_MLJAR_Extra Trees": "silver",
        "120 seconds_MLJAR_LightGBM": "gold",
        "120 seconds_MLJAR_SVM": "beige",
        "120 seconds_MLJAR_KNN": "brown",

        
    }
                        # Get the color based on the time budget and algorithm combination
                        unique_color = color_map_time_budget.get(time_budget_key, "gray")  # Default color is gray


                        comparison_records.append({
                            "Time Budget": tb,
                            "Algorithm": algo,
                            "Color": unique_color,  # Add the color to the record
                            "CO2 Emission": round(metrics.get("CO2 Emission", 0), 4),
                            "Energy Consumption": round(metrics.get("Energy Consumption", 0), 6),
                        })
            if comparison_records:
                df_comparison = pd.DataFrame(comparison_records)
                st.dataframe(df_comparison)
                st.subheader("Evaluation by time")
                if not df_comparison.empty:
                    # Extract framework from algorithm name (e.g., FLAML from FLAML_XGBoost)
                    df_comparison["Framework"] = df_comparison["Algorithm"].apply(lambda x: x.split("_")[0])
                    df_comparison["Time Label"] = df_comparison["Time Budget"].apply(lambda x: x.split()[0] + "s")
                    df_comparison["Time Budget Value"] = df_comparison["Time Budget"].str.extract(r'(\d+)').astype(int)
                    df_comparison = df_comparison.sort_values(by=["Energy Consumption", "CO2 Emission"])
                    df_comparison["Time_Algo"] = df_comparison["Time Budget"] + "_" + df_comparison["Algorithm"]

                    fig_line = px.line(
                        df_comparison,
                        x="CO2 Emission",
                        y="Energy Consumption",
                        color="Time_Algo",
                        text="Time Label",
                        hover_name="Algorithm",
                        hover_data={
                            "Algorithm": True,
                            "Time Label": True,
                            "CO2 Emission": True,
                            "Energy Consumption": True,
                            "Framework": False   },
                        markers=True,  # Ensures points show up with lines
                        line_group="Time_Algo",
                        log_y=True,
                        title="Energy vs CO2 for Each Framework (log scale)",
                        labels={"Energy Consumption": "Energy Consumption"},
                        color_discrete_map=color_map_time_budget  # 🔥 Correct color mapping
                        
                        )
                    #fig_line.update_traces(mode="lines+markers+text", textposition="top center")
                    fig_line.update_traces(
                    mode="lines+markers+text",
                    textposition="middle center",
                    textfont=dict(
                        size=11,
                        family="Arial Black",
                        color="black"  ),
                    marker=dict(
                        size=22,
                        color="white",  # White inner circle
                        line=dict(width=2),
                    #  line=dict(width=2, color="black"    #df_comparison["Color"]  # Outline color mapped from color_map
    )      )
                    for trace in fig_line.data:
                        trace.marker.line.color = trace.line.color  #  Match border color to line color
                        trace.marker.line.width = 2  #  Make sure border is visible

                    fig_line.update_layout(
                    yaxis=dict(
                        type="log",
                        tickvals=[0.1, 0.3, 1, 3, 10],
                        showgrid=False,        #  remove grey horizontal grid lines
                        zeroline=False,        #  remove baseline if any
                        showline=True,         #  show Y-axis line
                        linecolor='black',     # axis color
                        ticks="outside",
                        tickfont=dict(size=12) ),
                    xaxis=dict(
                        showgrid=False,        #  remove grey vertical grid lines
                        zeroline=False,
                        showline=True,         # show X-axis line
                        linecolor='black',
                        ticks="outside",
                        tickfont=dict(size=12) ),
                    plot_bgcolor="white",      # Set background to white
                )
                    # Update x axis and y axis font properties:
                    fig_line.update_layout(
                        xaxis=dict(
                            title="CO2 Emission (µg)",
                            title_font=dict(size=16, color='black', family='Arial', weight='bold'),  # Bold and black title
                            tickfont=dict(size=16, color='black', weight='bold')  # Bold and black ticks
                        ),
                        yaxis=dict(
                            title="Energy Consumption (µWh)",
                            title_font=dict(size=16, color='black', family='Arial', weight='bold'),  # Bold and black title
                            tickfont=dict(size=16, color='black', weight='bold')  # Bold and black ticks
                        ),                            )   
                    st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.warning("No comparison data available to plot.")
            # Update in the Evaluation by Framework section
            if st.session_state.get("show_comparison", False):
                st.subheader("Evaluation by Framework")
                # Create a dictionary to store best performing algorithms for each framework and time budget
                best_algorithms_per_time_budget = {}
                # Loop through the time budget comparisons to find the best performing algorithm for each framework and time budget
                for entry in st.session_state.get("time_budget_comparisons", []):
                    time_budget = entry["time_budget"]
                    results = entry["results"]
                    # Loop through the results and find the best performing algorithm for each framework and time budget
                    for algo, metrics in results.items():
                        framework = algo.split("_")[0]
                        co2_emission = metrics.get("CO2 Emission", float('inf'))
                        energy_consumption = metrics.get("Energy Consumption", float('inf'))
                        # If the framework and time budget do not exist in the dictionary, initialize them
                        if framework not in best_algorithms_per_time_budget:
                            best_algorithms_per_time_budget[framework] = {}
                        # If the time budget does not exist for this framework, initialize it
                        if time_budget not in best_algorithms_per_time_budget[framework]:
                            best_algorithms_per_time_budget[framework][time_budget] = {"algorithm": algo, 
                                                                                    "CO2 Emission": co2_emission, 
                                                                                    "Energy Consumption": energy_consumption}
                        # Check if the current algorithm has a better performance for the given time budget
                        current_best = best_algorithms_per_time_budget[framework][time_budget]
                        if co2_emission < current_best["CO2 Emission"] and energy_consumption < current_best["Energy Consumption"]:
                            best_algorithms_per_time_budget[framework][time_budget] = {
                                "algorithm": algo,
                                "CO2 Emission": co2_emission,
                                "Energy Consumption": energy_consumption
                            }
                # Prepare the final list of best algorithms to plot
                best_algo_data = []
                for framework, time_budgets in best_algorithms_per_time_budget.items():
                    for time_budget, data in time_budgets.items():
                        best_algo_data.append({
                            "Framework": framework,
                            "Algorithm": data["algorithm"],
                            "CO2 Emission": data["CO2 Emission"],
                            "Energy Consumption": data["Energy Consumption"],
                            "Time Budget": time_budget
                        })
                # Create a DataFrame with the best performing algorithms per framework and time budget
                df_best_algos = pd.DataFrame(best_algo_data)
                # Update the text column to display only "10s"
                df_best_algos['Time Budget'] = df_best_algos['Time Budget'].apply(lambda x: str(x)[:2] + "s")  # Convert time to "10s"
                # Plot the best performing algorithms for each time budget
                if not df_best_algos.empty:
                    fig = px.scatter(
                        df_best_algos,
                        x="CO2 Emission",
                        y="Energy Consumption",
                        color="Framework",
                        hover_name="Algorithm",
                        title="Best Performing Algorithms by Framework",
                        text="Time Budget",  # Show time budget as "10s"
                        color_discrete_map=color_map_time_budget  # Ensures color map is consistent
                    )

                    # Update the layout to format the time budget text
                    fig.update_traces(
                        texttemplate="%{text}",  # Show the text (time budget) only
                        textposition="middle center",  # Center the text inside the circles
                        textfont=dict(
                            family="Arial", 
                            size=14, 
                            color="black",  # Make the text color black
                            weight="bold"  # Make the text bold
                        ),
                        marker=dict(
                sizemode='diameter',  # Keep the circle size fixed
                size=30  # Fixed circle size (adjust as needed)
            )  )
                    # Customize axis titles and font sizes (make axis text and numbers black and bold)
                    fig.update_layout(
                        xaxis=dict(
                            title="CO2 Emission (µg)",
                            title_font=dict(size=16, color='black', family='Arial', weight='bold'),  # Bold and black title
                            tickfont=dict(size=16, color='black', weight='bold')  # Bold and black ticks
                        ),
                        yaxis=dict(
                            title="Energy Consumption (µWh)",
                            title_font=dict(size=16, color='black', family='Arial', weight='bold'),  # Bold and black title
                            tickfont=dict(size=16, color='black', weight='bold')  # Bold and black ticks
                              ),
                        title=dict(font=dict(size=18)),)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No best algorithms to display. Please ensure results are available for the selected frameworks and time budgets.") 
    elif st.session_state.get("show_hyperimpact_analysis", False):
        st.subheader("Hyperparameter Impact Analysis")
        # Ask the user to select a performance metric for analysis.
        selected_metric = st.selectbox(
            "Select performance metric for analysis:",
            options=["Accuracy", "F1 Score", "CO2 Emission"] )
        # Let the user choose which algorithm's hyperparameters to display.
        available_algorithms = list(st.session_state.get("automl_results", {}).keys())
        if not available_algorithms:
            st.warning("No AutoML results available. Run AutoML first.")
        else:
            selected_algo = st.selectbox(
                "Select an algorithm for hyperparameter impact analysis:",
                options=available_algorithms
            )
            # Retrieve the hyperparameters and the selected metric value for the chosen algorithm.
            algo_result = st.session_state["automl_results"].get(selected_algo, {})
            hyperparams = algo_result.get("hyperparameters", {})
            metric_value = algo_result.get(selected_metric, None)
            if not hyperparams or metric_value is None:
                st.warning("Hyperparameter data or the selected performance metric is missing for the chosen algorithm.")
            else:
                # Filter to keep only numeric hyperparameters.
                numeric_hyperparams = {k: v for k, v in hyperparams.items() if isinstance(v, (int, float))}
                
                if not numeric_hyperparams:
                    st.warning("No numeric hyperparameters found for impact analysis.")
                else:
                    # Normalize the hyperparameter values by dividing by the maximum value.
                    max_val = max(numeric_hyperparams.values())
                    impact_data = []
                    for param, value in numeric_hyperparams.items():
                        normalized_value = value / max_val if max_val != 0 else 0
                        # Compute an "impact" score by scaling the normalized value with the performance metric.
                        impact_score = normalized_value * metric_value
                        impact_data.append({
                            "Hyperparameter": param,
                            "Value": value,
                            "Impact": impact_score })
                    df_impact = pd.DataFrame(impact_data)
                    # Create a grouped bar chart using Plotly Express.
                    fig = px.bar(
                        df_impact,
                        x="Hyperparameter",
                        y="Impact",
                        #text="Value",
                        hover_data=["Value"],
                        title=f"Impact of Hyperparameters on {selected_metric.capitalize()} ({selected_algo})" )
                    # Update x axis and y axis font properties:
                    fig.update_layout(
                        xaxis=dict(
                            title=dict(font=dict(size=14, color="black", family="Arial black")),
                            tickfont=dict(size=16, color="black", family="Arial")),
                        yaxis=dict(
                            title=dict(font=dict(size=14, color="black", family="Arial black")),
                            tickfont=dict(size=16, color="black", family="Times New Roman"), 
                            title_text=f"Impact ({selected_metric})"
                        )    )
                    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.get("show_model_card", False):
        st.title("📋 Model Card Generator")
        st.caption("Generate professional documentation for your model")
        results = st.session_state.get("automl_results", {}) or {}
        if not results:
            st.warning("No AutoML results found. Please run AutoML first.")
        else:
            # Select model and audience
            model_names = list(results.keys())
            selected_model = st.selectbox(
                "Select Model:",
                options=model_names,
                key="model_card_model"
            )
            if selected_model and results[selected_model]:
                metrics = results[selected_model]
                # Prepare model info
                model_info = {
                    'name': selected_model,
                    'framework': selected_model.split('_')[0],
                    'algorithm': selected_model.split('_')[1] if '_' in selected_model else 'Unknown'
                }
                # Prepare metrics
                model_metrics = {
                    'accuracy': float(metrics.get('Accuracy', 0)),
                    'f1_score': float(metrics.get('F1 Score', 0)),
                    'precision': float(metrics.get('Precision', 0)) if metrics.get('Precision') else 0,
                    'recall': float(metrics.get('Recall', 0)) if metrics.get('Recall') else 0,
                    'co2_emission': float(metrics.get('CO2 Emission', 0)),
                    'energy_consumption': float(metrics.get('Energy Consumption', 0)),
                }
                # Prepare features
                features_info = {
                    'top_features': list(metrics.get('feature_importance', {}).keys())[:5] if metrics.get('feature_importance') else [],
                    'total_features': len(metrics.get('feature_importance', {})) if metrics.get('feature_importance') else 0
                }
                # Preview section
                st.markdown("### 📄 Card Preview")
                preview_cols = st.columns([1, 1, 1, 1])
                preview_cols[0].metric("Model", model_info['name'])
                preview_cols[1].metric("Accuracy", f"{model_metrics['accuracy']:.4f}")
                preview_cols[2].metric("CO2", f"{model_metrics['co2_emission']:.6f} µg")
                # preview_cols[3].metric("Audience", audience)                                
                # Select card sections
                st.markdown("### 📋 Select Card Sections")
                cols = st.columns(3)
                with cols[0]:
                    include_overview = st.checkbox("📊 Model Overview", value=True)
                with cols[1]:
                    include_metrics = st.checkbox("📈 Performance Metrics", value=True)
                with cols[2]:
                    include_sustainability = st.checkbox("🌱 Sustainability Metrics", value=True)
                cols2 = st.columns(3)
                with cols2[0]:
                    include_features = st.checkbox("⭐ Top Features", value=True)
                with cols2[1]:
                    include_limitations = st.checkbox("⚠️ Limitations", value=True)
                with cols2[2]:
                    include_recommendations = st.checkbox("💡 Recommendations", value=True)
                #  Generate PDF Button
                st.markdown("### 📥 Download Model Card")
                
                st.info("Click below to generate and download the model card PDF")
                
                if st.button("📥 Download PDF", use_container_width=True):
                        try:
                            # Generate PDF
                            pdf_bytes = generate_model_card_pdf(
                                 model_info=model_info, 
                                metrics=model_metrics,
                                features=features_info,
                                audience="Normal User",
                                all_results=results,
                                selected_algos=selected_algos,
                                include_overview=include_overview,
                                include_metrics=include_metrics,
                                include_sustainability=include_sustainability,
                                include_features=include_features,
                                include_limitations=include_limitations,
                                include_recommendations=include_recommendations,
                                
                            )
                            
                            # Download button
                            st.download_button(
                                label="💾 Save PDF",
                                data=pdf_bytes,
                                file_name=f"model_card_{selected_model}.pdf",
                                mime="application/pdf"
                            )
                            st.success("✅ PDF generated successfully!")
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                # Card Info
                st.markdown("### ℹ️ What's in the Card")
                card_items = []
                if include_overview:
                    card_items.append("✓ **Model Overview** - Name, framework, algorithm, creation date")
                if include_metrics:
                    card_items.append("✓ **Performance Metrics** - Accuracy, F1, Precision, Recall")
                if include_sustainability:
                    card_items.append("✓ **Sustainability Metrics** - CO2 and Energy")
                if include_features:
                    card_items.append("✓ **Top Features** - Most important features")
                if include_limitations:
                    card_items.append("✓ **Limitations** - When model may fail")
                if include_recommendations:
                    card_items.append("✓ **Recommendations** - How to use safely")
                st.markdown("\n".join(card_items))