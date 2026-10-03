from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from flask import Flask, request, jsonify
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from codecarbon import OfflineEmissionsTracker
import pandas as pd
import numpy as np
from metaknowledge_collector import (
    compute_stability, compute_fairness, compute_edge_case_performance, compute_calibration, load_baseline
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier
import time
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from sklearn.tree import DecisionTreeClassifier
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import joblib
import warnings
from io import StringIO
from pymfe.mfe import MFE
from tabpfn import TabPFNClassifier
# from algorithm import framework_algorithms   # same file we used for benchmarking
from pathlib import Path
from sklearn.base import clone         
from fairness_detection import (
    calculate_fairness_metrics,
    detect_bias,
    test_fairness_significance,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
import optuna
import random

# Load baseline for medical domain
try:
    BASELINE = load_baseline("./baseline_medical.json")
except:
    BASELINE = None
# --- Clusteval ---
from clusteval import clusteval as CE
try:
    import hdbscan as _hdbscan
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False

# ──────────────────────────────────────────────────────────────────────────
# 2)  LOAD  SCALER  &  TABPFN  MODELS  **ONCE**
#     (place right after the imports)
# ──────────────────────────────────────────────────────────────────────────
# ─── one-time model loading block (leave exactly like this) ─────────────
MODELS_DIR = Path(__file__).parent / "models"

SCALER = joblib.load(MODELS_DIR / "scaler.pkl")

META_MODELS = {
    fw: {
        "algo": joblib.load(MODELS_DIR / f"meta_tabpfn_{fw.lower()}_algo.pkl"),
        "time": joblib.load(MODELS_DIR / f"meta_tabpfn_{fw.lower()}_time.pkl"),
    }
    for fw in ["FLAML", "H2O", "MLJAR"]
}

 #––– H2O setup (only once) –––
app = Flask(__name__)

# Runtime state for the most recent classification AutoML run.
# Fairness is evaluated later from this held-out set using the actual winning model.
CLASSIFICATION_RUNTIME = {}
try: 
 h2o.init(max_mem_size="2G", nthreads=1)
 
except Exception as e:
    print(f"Failed to initialize H2O cluster: {e}")
def calculate_feature_importance(model, X, y, framework=None):
    """Calculate feature importance for a given model.
      Special handling for H2O framework."""
    if framework == "H2O":
        try:
            fi = model.varimp(use_pandas=True)
            return dict(zip(fi['variable'], fi['relative_importance']))
        except Exception as e:
            print(f"H2O feature importance error: {e}")
            return {}

    try:
        if hasattr(model, "feature_importances_"):  # Tree-based models
            return dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, "coef_") and model.coef_.ndim == 1:  # Linear models
            return dict(zip(X.columns, model.coef_))
        else:  # Use permutation importance
            try:
                result = permutation_importance(model, X, y, scoring="accuracy", n_repeats=5, random_state=42)
                return dict(zip(X.columns, result.importances_mean))
            except Exception:
                return {}  # Return empty dictionary if permutation importance fails
    except Exception as e:
        print(f"Feature importance calculation failed: {e}")
        return {}  # Return empty dictionary if feature importance can't be calculated
def calculate_clustering_feature_importance(X, labels, feature_names=None):
    """
    Calculate feature importance for clustering based on cluster separation.
    Higher score = feature strongly separates clusters.
    """
    if X.shape[1] == 0:
        return {}
    
    try:
        feature_importances = {}
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            return {}
        
        # Within-cluster variance (lower is better for separation)
        within_var = np.zeros(X.shape[1])
        for label in unique_labels:
            cluster_mask = labels == label
            if np.sum(cluster_mask) > 1:
                within_var += np.var(X[cluster_mask], axis=0)
        within_var /= len(unique_labels)
        
        # Between-cluster variance (higher is better for separation)
        cluster_means = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
        between_var = np.var(cluster_means, axis=0)
        
        # Separation score: between / (within + epsilon)
        separation_score = between_var / (within_var + 1e-6)
        
        # Normalize to 0-1 range
        if separation_score.max() > 0:
            normalized = separation_score / separation_score.max()
        else:
            normalized = np.zeros_like(separation_score)
        
        # Create dictionary
        # Create dictionary with real feature names
        if feature_names is not None:
            for i, name in enumerate(feature_names):
                feature_importances[name] = float(normalized[i])
        elif hasattr(X, 'columns'):  # pandas DataFrame
            for i, col in enumerate(X.columns):
                feature_importances[col] = float(normalized[i])
        else:  # numpy array
            for i in range(X.shape[1]):
                feature_importances[f"Feature_{i}"] = float(normalized[i])
        
        return feature_importances
    
    except Exception as e:
        print(f"Clustering feature importance error: {e}")
        return {}

def build_cluster_profile(X, labels, feature_names=None):
    """Build cluster profile (mean values per cluster)."""
    import pandas as pd
    import numpy as np
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['cluster'] = labels
    
    profile = df.groupby('cluster').mean()
    return profile.to_dict()
# --- LLM run-log for grounded explanations ---
RUN_LOG = {
    "classification": [],
    "clustering": [],
    "interpretability": []
}

def _log_run(task, framework, payload, result):
    """Keep a rolling buffer of latest runs the LLM can cite."""
    RUN_LOG[task].append({
        "ts": time.time(),
        "framework": framework,
        "payload": payload,   # what was requested (metric, k_range, time budget…)
        "result": result      # what you returned to the frontend
    })
    if len(RUN_LOG[task]) > 50:
        RUN_LOG[task] = RUN_LOG[task][-50:]

def _json_safe(obj):
    """Recursively convert objects into strict-JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None

    if isinstance(obj, np.bool_):
        return bool(obj)

    # Handle plain Python floats like nan / inf
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None

    # Handle pandas missing values safely
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj

# Full algorithm mapping
framework_algorithms = {
    "FLAML": {
        "RF": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=2,
            min_samples_leaf=1,
            warm_start=True,
            random_state=42
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=2,
            min_samples_leaf=1,
            warm_start=True,
            random_state=42
        ),

        "Logistic Regression": LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            learning_rate=0.1,
            n_estimators=100,
            max_depth=6,
            subsample=1.0,
            colsample_bytree=1.0,
            eval_metric="mlogloss",
            random_state=42
        ),

        "LightGBM": LGBMClassifier(
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=100,
            min_child_samples=20,
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform"
        ),
    },

    "H2O": {
        "GLM": H2OGeneralizedLinearEstimator(
            family="binomial",
            alpha=0.5,
            lambda_=0.1,
            seed=42
        ),

        "Distributed RF": H2ORandomForestEstimator(
            ntrees=200,
            max_depth=6,
            seed=42
        ),

        "GBM": H2OGradientBoostingEstimator(
            ntrees=100,
            learn_rate=0.05,
            max_depth=6,
            seed=42
        ),

        "Deep Learning": H2ODeepLearningEstimator(
            epochs=10,
            hidden=[50, 50],
            activation="Rectifier",
            seed=42
        ),
    },

    "MLJAR": {
        "Baseline": LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ),

        "RF": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=2,
            min_samples_leaf=1,
            warm_start=True,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            learning_rate=0.1,
            n_estimators=100,
            max_depth=6,
            subsample=1.0,
            colsample_bytree=1.0,
            eval_metric="mlogloss",
            random_state=42
        ),

        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            alpha=0.0001,
            max_iter=500,
            random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=2,
            min_samples_leaf=1,
            warm_start=True,
            random_state=42
        ),

        "LightGBM": LGBMClassifier(
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=100,
            min_child_samples=20,
            random_state=42
        ),

        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform"
        ),
    },
}

@app.route('/run_automl', methods=['POST'])
def run_automl():
    try:
        # Parse incoming request
        data = request.json
        dataset_json = data.get("data")
        selected_frameworks = data.get("frameworks", [])
        selected_algorithms = data.get("algorithms", {})
        custom_hyperparams = data.get("hyperparams", {})  
        time_budget = data.get("time_budget", 60)
        # Log the received input for debugging
        print("Received frameworks:", selected_frameworks)
        print("Received algorithms:", selected_algorithms)
        print("Custom hyperparameters:", custom_hyperparams)

        # Convert JSON dataset to DataFrame
        from io import StringIO
        raw_df = pd.read_json(StringIO(dataset_json))
        response_col = raw_df.columns[-1]
        raw_features = raw_df.drop(columns=[response_col]).copy()
        y_series     = raw_df[response_col]
        X_df         = raw_features.copy()
        # Separate numerical and categorical columns
        numeric_cols     = X_df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = X_df.select_dtypes(include=['object']).columns
        # Handle missing values (numerical and categorical columns)
        imputer = SimpleImputer(strategy='mean')
        X_df[numeric_cols] = imputer.fit_transform(X_df[numeric_cols])
        scaler = StandardScaler()
        X_df[numeric_cols] = scaler.fit_transform(X_df[numeric_cols])

        # 3️⃣  categorical imputation + encoding (if any)
        if len(categorical_cols) > 0:
            cat_imp = SimpleImputer(strategy="most_frequent")
            X_df[categorical_cols] = cat_imp.fit_transform(X_df[categorical_cols])
            for col in categorical_cols:
                X_df[col] = LabelEncoder().fit_transform(X_df[col])

        # 4️⃣  rebuild dataframe with label untouched
        df = pd.concat([X_df, y_series], axis=1).drop_duplicates()

        print("Dataset shape after preprocessing:", df.shape)  # Log dataset shape for debugging

        # Keep ONE held-out split for every framework, including H2O.
        # This is the data used for both model selection and fairness evaluation.
        X = df.iloc[:, :-1].copy()
        y = df.iloc[:, -1].copy()
        n_classes = y.nunique()

        results = {}
        from sklearn.model_selection import train_test_split
        class_counts = y.value_counts()
        strat_param = y if class_counts.min() >= 2 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=strat_param, random_state=42)

        # Align the original demographic columns to exactly the same held-out rows.
        raw_features_aligned = raw_features.loc[df.index].copy()
        fairness_raw_eval = raw_features_aligned.loc[X_val.index].copy()

        # Build H2O frames from the SAME train/validation split.
        train_h2o_df = pd.concat([X_train, y_train], axis=1)
        val_h2o_df = pd.concat([X_val, y_val], axis=1)
        train_hf = h2o.H2OFrame(train_h2o_df)
        val_hf = h2o.H2OFrame(val_h2o_df)
        train_hf[response_col] = train_hf[response_col].asfactor()
        val_hf[response_col] = val_hf[response_col].asfactor()

        global CLASSIFICATION_RUNTIME
        CLASSIFICATION_RUNTIME = {
            "available": False,
            "best_key": None,
            "best_accuracy": None,
            "framework": None,
            "algorithm": None,
            "model_type": None,
            "model": None,
            "X_eval": X_val.copy(),
            "y_eval": y_val.copy(),
            "raw_eval": fairness_raw_eval.copy(),
            "positive_labels": [str(v) for v in pd.unique(y)],
        }
        # Process each selected framework and algorithm
        for framework in selected_frameworks:
            framework_algos = framework_algorithms.get(framework, {})
            algorithms = selected_algorithms.get(framework, {})
            for algo_name, is_selected in algorithms.items():

                if not is_selected or algo_name not in framework_algos:
                    continue
                if framework == "H2O" and algo_name == "GLM" and n_classes > 2:
                    print("Skipping H2O-GLM → needs a binary target")
                    continue
                # --- always start from a *fresh* estimator object ---
                template = framework_algos[algo_name]
                if framework == "H2O":
                    model = template.__class__(**template.params)      # new H2O model
                else:
                    model = clone(template)
                if framework == "H2O":
                    default_params = {k: v['actual'] for k, v in model.params.items()}
                else:
                    default_params = model.get_params()
                # Apply custom hyperparameters if provided
                user_params = custom_hyperparams.get(framework, {}).get(algo_name, {})
                final_params = {**default_params, **user_params}  # Merge default and custom params
                # Log the final parameters before applying
                print(f"Final parameters for {framework} - {algo_name}: {final_params}")
                # Update the existing model parameters
                try:
                    # Validate custom hyperparameters
                    if framework == "H2O":
                        valid_params = model.params.keys()          # H2O path
                    else:
                        valid_params = model.get_params().keys()    # scikit-learn path
                    filtered_params = {k: v for k, v in final_params.items() if k in valid_params}
                    print(f"Filtered parameters for {framework} - {algo_name}: {filtered_params}")
                    # Apply only valid parameters
                    if framework != "H2O" and filtered_params:
                         model.set_params(**filtered_params)
                    # # Update the existing model parameters
                    # model.set_params(**final_params)
                    print(f"Applied parameters to {framework} - {algo_name}: {model.get_params()}")
                except Exception as e:
                    print(f"Error applying parameters to {framework} - {algo_name}: {e}")
                    continue
                # Initialize CodeCarbon tracker
                tracker=OfflineEmissionsTracker(country_iso_code="EST",log_level="critical",allow_multiple_runs=True)
                tracker.start()
                try:
                    tracker.start_task()
                    xai_signals = {
                        "interpretability": None,
                        "stability": None,
                        "fairness": None,
                        "calibration": None,
                    }
                    if framework == "H2O":
                        features = [c for c in df.columns if c != response_col]
                        model.train(x=features, y=response_col, training_frame=train_hf)
                        pred_df = model.predict(val_hf).as_data_frame()
                        y_pred = pred_df["predict"].values
                        y_true = val_hf[response_col].as_data_frame().values.ravel()
                        accuracy = accuracy_score(y_true, y_pred)
                        f1 = f1_score(y_true, y_pred, average="weighted")
                        feature_importance = calculate_feature_importance(model, X, y, framework)
                        fitted_runtime_model = model
                        runtime_model_type = "H2O"
                    else:
                        fitted = fit_with_budget(model, X_train, y_train, time_budget)
                        y_pred = fitted.predict(X_val)
                        try:
                            y_pred_proba = fitted.predict_proba(X_val)
                        except:
                            y_pred_proba = np.column_stack([1-y_pred, y_pred])
                        
                        accuracy = accuracy_score(y_val, y_pred)
                        f1       = f1_score(y_val, y_pred, average="weighted")
                        
                        # Compute XAI signals
                        xai_signals.update({
                            "stability": compute_stability(
                                X_val,
                                y_pred_proba
                            ),
                            "fairness": compute_fairness(
                                X_val,
                                y_val,
                                y_pred
                            ),
                            "calibration": compute_calibration(
                                y_val,
                                y_pred_proba[:, 1]
                            ),
                        })
                        feature_importance = calculate_feature_importance(model, X, y, framework)
                        fitted_runtime_model = fitted
                        runtime_model_type = "sklearn"

                    # Keep the ACTUAL winning model, not a newly-created Random Forest.
                    current_best = CLASSIFICATION_RUNTIME.get("best_accuracy")
                    if current_best is None or float(accuracy) > float(current_best):
                        CLASSIFICATION_RUNTIME.update({
                            "available": True,
                            "best_key": f"{framework}_{algo_name}",
                            "best_accuracy": float(accuracy),
                            "framework": framework,
                            "algorithm": algo_name,
                            "model_type": runtime_model_type,
                            "model": fitted_runtime_model,
                            "X_eval": X_val.copy(),
                            "y_eval": y_val.copy(),
                            "raw_eval": fairness_raw_eval.copy(),
                            "positive_labels": [v for v in pd.unique(y)],
                            "xai_signals": xai_signals, 
                            "baseline": BASELINE  
                        })

                    # Stop tracker and get CO2 emissions
                    emissions = tracker.stop_task()

                    # Convert emissions from kg to micro-units for visualization consistency
                    co2_emissions_micro = emissions.emissions * 1_000_000
                    energy_micro_wh = emissions.energy_consumed * 1_000_000
                    results[f"{framework}_{algo_name}"] = _json_safe({
                            "Accuracy": float(accuracy) if accuracy is not None else None,
                            "F1 Score": float(f1) if f1 is not None else None,
                            "CO2 Emission": float(co2_emissions_micro) if co2_emissions_micro is not None else None,
                            "Energy Consumption": float(energy_micro_wh) if energy_micro_wh is not None else None,
                            "feature_importance": (
                                {str(k): float(v) if v is not None and np.isfinite(float(v)) else None
                                for k, v in feature_importance.items()}
                                if isinstance(feature_importance, dict) else None
                            ),
                            "xai_signals": xai_signals, 
                            "hyperparameters": _json_safe(filtered_params)
                        })
                except Exception as e:
                    # Log errors for debugging
                    print(f"Error with {framework}_{algo_name}: {e}")
                    results[f"{framework}_{algo_name}"] = {"error": str(e)}
                finally:
                    try:
                        tracker.stop()
                    except:
                        pass
                   
        # Check if results are empty
        if not results:
            print("No results generated. Check input data or algorithm configurations.")
            return jsonify({"status": "error", "message": "No results generated. Check input data or algorithm configurations."})
         # --- Log a compact summary for the copilot (classification) ---
        try:
            valid_keys = [k for k, v in results.items() if isinstance(v, dict) and "Accuracy" in v]
            if valid_keys:
                best_key = max(valid_keys, key=lambda k: results[k]["Accuracy"])
                best = results[best_key]
                fw, algo = best_key.split("_", 1) if "_" in best_key else ("", best_key)

                summary = {
                    "framework": fw,
                    "algo": algo,
                    "metrics": {
                        "accuracy": float(best.get("Accuracy", 0)),
                        "f1": float(best.get("F1 Score", 0)),
                        "co2": float(best.get("CO2 Emission", 0)) if best.get("CO2 Emission") is not None else None,
                        "energy": float(best.get("Energy Consumption", 0)) if best.get("Energy Consumption") is not None else None,
                                 }
                }
                payload_for_log = {
                    "frameworks": selected_frameworks,
                    "time_budget": time_budget,
                }
                _log_run("classification", fw or "multi", payload_for_log, summary)
        except Exception as _e:
            print("classification log skipped:", _e)

        print("Results generated:", results)
        response_payload = _json_safe({
            "status": "success",
            "results": results
        })
        return jsonify(response_payload)

    except Exception as e:
        print("Error in run_automl:", str(e))
        return jsonify({"status": "error", "message": str(e)})
    
@app.route('/validate_classification_recommendation', methods=['POST'])
def validate_classification_recommendation():
    """
    Rerun exactly ONE classification model/configuration for recommendation
    validation.

    IMPORTANT:
    - Does NOT update CLASSIFICATION_RUNTIME.
    - Does NOT replace the main AutoML results.
    - Uses the same preprocessing and held-out split logic as /run_automl.
    - Intended only for baseline-vs-recommendation comparison.
    """
    tracker = None

    try:
        data = request.json or {}

        dataset_json = data.get("data")
        framework = data.get("framework")
        algorithm = data.get("algorithm")
        custom_hyperparams = data.get("hyperparams", {}) or {}
        time_budget = int(data.get("time_budget", 60))
        seed = int(data.get("seed", 42))
        if not dataset_json:
            return jsonify({
                "status": "error",
                "message": "Missing dataset."
            }), 400

        if framework not in framework_algorithms:
            return jsonify({
                "status": "error",
                "message": f"Unknown framework: {framework}"
            }), 400

        framework_algos = framework_algorithms.get(framework, {})

        if algorithm not in framework_algos:
            return jsonify({
                "status": "error",
                "message": (
                    f"Unknown algorithm '{algorithm}' "
                    f"for framework '{framework}'."
                )
            }), 400

        # ==============================================================
        # 1. LOAD DATA
        # ==============================================================

        raw_df = pd.read_json(StringIO(dataset_json))

        if raw_df.shape[1] < 2:
            return jsonify({
                "status": "error",
                "message": "Dataset requires at least one feature and one target."
            }), 400

        response_col = raw_df.columns[-1]

        raw_features = raw_df.drop(columns=[response_col]).copy()
        y_series = raw_df[response_col].copy()
        X_df = raw_features.copy()

        # ==============================================================
        # 2. SAME PREPROCESSING AS /run_automl
        # ==============================================================

        numeric_cols = X_df.select_dtypes(
            include=['float64', 'int64']
        ).columns

        categorical_cols = X_df.select_dtypes(
            include=['object']
        ).columns

        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='mean')
            X_df[numeric_cols] = imputer.fit_transform(
                X_df[numeric_cols]
            )

            scaler = StandardScaler()
            X_df[numeric_cols] = scaler.fit_transform(
                X_df[numeric_cols]
            )

        if len(categorical_cols) > 0:
            cat_imp = SimpleImputer(strategy="most_frequent")

            X_df[categorical_cols] = cat_imp.fit_transform(
                X_df[categorical_cols]
            )

            for col in categorical_cols:
                X_df[col] = LabelEncoder().fit_transform(
                    X_df[col]
                )

        df = pd.concat(
            [X_df, y_series],
            axis=1
        ).drop_duplicates()

        X = df.iloc[:, :-1].copy()
        y = df.iloc[:, -1].copy()

        n_classes = y.nunique()

        # ==============================================================
        # 3. EXACT SAME HELD-OUT SPLIT POLICY
        # ==============================================================

        class_counts = y.value_counts()

        strat_param = (
            y
            if len(class_counts) > 0 and class_counts.min() >= 2
            else None
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=strat_param,
            random_state=seed
        )

        # ==============================================================
        # 4. CREATE FRESH MODEL
        # ==============================================================

        template = framework_algos[algorithm]

        if framework == "H2O":

            if algorithm == "GLM" and n_classes > 2:
                return jsonify({
                    "status": "error",
                    "message": "H2O GLM currently requires a binary target."
                }), 400

            # Preserve meaningful parameters already configured on the
            # template, then override ONLY the requested dashboard values.
            base_params = {}

            try:
                for key, value_info in template.params.items():
                    actual = value_info.get("actual")

                    if actual is not None:
                        base_params[key] = actual
            except Exception:
                base_params = {}

            valid_param_names = set(template.params.keys())

            requested_params = {
                key: value
                for key, value in custom_hyperparams.items()
                if (
                    key in valid_param_names
                    and value is not None
                )
            }

            base_params.update(requested_params)
            if "seed" in valid_param_names:
                base_params["seed"] = seed
            try:
                model = template.__class__(**base_params)
            except Exception:
                # Conservative fallback: instantiate with requested values only.
                model = template.__class__(**requested_params)

        else:
            model = clone(template)

            valid_param_names = set(
                model.get_params().keys()
            )

            requested_params = {
                key: value
                for key, value in custom_hyperparams.items()
                if (
                    key in valid_param_names
                    and value is not None
                )
            }

            if requested_params:
                model.set_params(**requested_params)

            # Use the same seed for stochastic estimators where supported.
            # Deterministic estimators such as KNN simply do not expose
            # random_state and are left unchanged.
            if "random_state" in valid_param_names:
                model.set_params(
                    random_state=seed
                )
                requested_params[
                    "random_state"
                ] = seed
        # ==============================================================
        # EFFECTIVE CONFIGURATION ACTUALLY SENT TO THE ESTIMATOR
        # ==============================================================

        if framework == "H2O":
            effective_params = {}

            try:
                for key, value_info in model.params.items():
                    effective_params[key] = (
                        value_info.get("actual")
                    )
            except Exception:
                effective_params = {}

        else:
            effective_params = model.get_params(
                deep=False
            )
        # ==============================================================
        # 5. ENERGY / CO2 TRACKING
        # ==============================================================

        tracker = OfflineEmissionsTracker(
            country_iso_code="EST",
            log_level="critical",
            allow_multiple_runs=True
        )

        tracker.start()
        tracker.start_task()

        # ==============================================================
        # 6. TRAIN + EVALUATE
        # ==============================================================
        if framework == "H2O":

            train_h2o_df = pd.concat(
                [X_train, y_train],
                axis=1
            )

            val_h2o_df = pd.concat(
                [X_val, y_val],
                axis=1
            )

            train_hf = h2o.H2OFrame(train_h2o_df)
            val_hf = h2o.H2OFrame(val_h2o_df)

            train_hf[response_col] = (
                train_hf[response_col].asfactor()
            )

            val_hf[response_col] = (
                val_hf[response_col].asfactor()
            )

            features = [
                c for c in df.columns
                if c != response_col
            ]

            model.train(
                x=features,
                y=response_col,
                training_frame=train_hf
            )

            pred_df = model.predict(
                val_hf
            ).as_data_frame()

            y_pred = pred_df["predict"].values
            # Probability predictions for ROC-AUC / Log Loss
            y_pred_proba = None

            try:
                probability_cols = [
                    col
                    for col in pred_df.columns
                    if str(col).startswith("p")
                ]

                if probability_cols:
                    y_pred_proba = (
                        pred_df[probability_cols]
                        .to_numpy()
                    )
            except Exception:
                y_pred_proba = None

            y_true = (
                val_hf[response_col]
                .as_data_frame()
                .values
                .ravel()
            )

        else:

            model.fit(
                X_train,
                y_train
            )

            fitted = model
            y_pred = fitted.predict(X_val)
            y_true = y_val
            try:
                y_pred_proba = fitted.predict_proba(X_val)
            except Exception:
                y_pred_proba = None
        
        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted"
        )
        # ==============================================================
        # PROBABILITY-SENSITIVE VALIDATION METRICS
        # ==============================================================

        roc_auc = None
        logloss_value = None

        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    roc_auc = roc_auc_score(
                        y_true,
                        y_pred_proba[:, 1]
                    )
                else:
                    roc_auc = roc_auc_score(
                        y_true,
                        y_pred_proba,
                        multi_class="ovr",
                        average="weighted"
                    )
            except Exception as e:
                print(
                    "ROC-AUC calculation failed:",
                    str(e)
                )
                roc_auc = None

            try:
                logloss_value = log_loss(
                    y_true,
                    y_pred_proba
                )
            except Exception as e:
                print(
                    "Log Loss calculation failed:",
                    str(e)
                )
                logloss_value = None
        

      
        emissions = tracker.stop_task()

        co2_emissions_micro = (
            emissions.emissions * 1_000_000
        )

        energy_micro_wh = (
            emissions.energy_consumed * 1_000_000
        )
        result = _json_safe({
            "framework": framework,
            "algorithm": algorithm,
            "Accuracy": float(accuracy),
            "F1 Score": float(f1),
            "CO2 Emission": float(co2_emissions_micro),
            "Energy Consumption": float(energy_micro_wh),
            "time_budget": int(time_budget),
            "requested_hyperparameters": requested_params,
            "effective_hyperparameters": effective_params,
            "seed": int(seed),
            "ROC-AUC": (
                float(roc_auc)
                if roc_auc is not None
                else None
            ),

            "Log Loss": ( 
                float(logloss_value)
                if logloss_value is not None
                else None
            ),        
        })

        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:
        print(
            "Recommendation validation error:",
            str(e)
        )

        return jsonify({
            "status": "error",
            "message": f"{type(e).__name__}: {str(e)}"
        }), 500

    finally:
        if tracker is not None:
            try:
                tracker.stop()
            except Exception:
                pass
@app.route('/fairness_analysis', methods=['POST'])
def fairness_analysis():
    """Run fairness analysis on the held-out validation set using the actual AutoML winner."""
    try:
        if not CLASSIFICATION_RUNTIME.get("available"):
            return jsonify({"status": "error", "message": "No completed classification AutoML model is available. Run AutoML first."}), 400

        data = request.json or {}
        demographic_col = data.get("demographic_col")
        positive_label = data.get("positive_label", None)
        alpha = float(data.get("alpha", 0.05))

        raw_eval = CLASSIFICATION_RUNTIME["raw_eval"]
        if demographic_col not in raw_eval.columns:
            return jsonify({
                "status": "error",
                "message": f"Demographic column '{demographic_col}' is not available in the original evaluation data."
            }), 400

        # JSON converts labels to strings; map back to the actual target dtype where possible.
        y_eval = CLASSIFICATION_RUNTIME["y_eval"]
        if positive_label is not None:
            candidates = list(pd.unique(y_eval))
            matched = next((v for v in candidates if str(v) == str(positive_label)), None)
            positive_label = matched if matched is not None else positive_label

        model = CLASSIFICATION_RUNTIME["model"]
        X_eval = CLASSIFICATION_RUNTIME["X_eval"]

        # Get predictions and probabilities in ONE block
        if CLASSIFICATION_RUNTIME["model_type"] == "H2O":
            eval_hf = h2o.H2OFrame(pd.concat([X_eval, y_eval], axis=1))
            response_col = y_eval.name
            eval_hf[response_col] = eval_hf[response_col].asfactor()
            pred_df = model.predict(eval_hf).as_data_frame()
            y_pred = pred_df["predict"].values
            # GET PROBABILITIES:
            try:
                y_pred_proba = pred_df.iloc[:, 1:].values  # Get probability columns
            except:
                y_pred_proba = None
        else:
            y_pred = model.predict(X_eval)
            # GET PROBABILITIES:
            try:
                y_pred_proba = model.predict_proba(X_eval)
            except:
                y_pred_proba = None

        groups = raw_eval[demographic_col].reset_index(drop=True)
        y_eval_reset = pd.Series(y_eval).reset_index(drop=True)

        # Single call with probabilities
        fairness_metrics = calculate_fairness_metrics(
            y_eval_reset, y_pred, groups, positive_label=positive_label, y_pred_proba=y_pred_proba
        )
        bias_info = detect_bias(fairness_metrics)
        significance = test_fairness_significance(y_eval_reset, y_pred, groups, alpha=alpha)

        # Prepare metrics for heatmap display
        fairness_metrics_summary = {
            'Demographic Parity': fairness_metrics.get('demographic_parity', 0),
            'Equalized Odds': fairness_metrics.get('equalized_odds', 0),
            'Predictive Parity': fairness_metrics.get('predictive_parity', 0),
            'Calibration': fairness_metrics.get('calibration', 0),
            'Accuracy Parity': fairness_metrics.get('accuracy_parity', 0),
            'FNR/FPR Parity': fairness_metrics.get('fnr_fpr_parity', 0),
        }

        # Do not silently turn statistical significance into a bias claim; report it separately.
        return jsonify(_json_safe({
            "status": "success",
            "model": {
                "key": CLASSIFICATION_RUNTIME["best_key"],
                "framework": CLASSIFICATION_RUNTIME["framework"],
                "algorithm": CLASSIFICATION_RUNTIME["algorithm"],
                "accuracy": CLASSIFICATION_RUNTIME["best_accuracy"],
            },
            "data_type": "held-out validation",
            "sample_count": int(len(X_eval)),
            "demographic_col": demographic_col,
            "positive_label": positive_label,
            "metrics": fairness_metrics,
            "bias": bias_info,
            "significance": significance,
            "fairness_metrics_summary": fairness_metrics_summary,
        }))
    except Exception as e:
        import traceback

        print("\n" + "=" * 80)
        print("ERROR IN /fairness_analysis")
        print("=" * 80)
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        traceback.print_exc()
        print("=" * 80 + "\n")

        return jsonify({
            "status": "error",
            "message": f"{type(e).__name__}: {str(e)}"
        }), 500


def fit_with_budget(model, X_train, y_train, time_budget):
    import time
    
    from sklearn.metrics import accuracy_score, f1_score
    start = time.time()
    if hasattr(model, "partial_fit"):
        classes = np.unique(y_train)
        first = True
        while time.time() - start < time_budget:
            model.partial_fit(X_train, y_train, classes=classes if first else None)
            first = False
    elif hasattr(model, "warm_start") and model.get_params().get("warm_start", False):
        n = model.get_params().get("n_estimators", 10)
        while time.time() - start < time_budget:
            n += 5
            model.set_params(n_estimators=n)
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    return model


# Error handling for warnings and exceptions during feature extraction
def safe_feature_extraction(X_only):
    """Extract features safely, suppressing any warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # Suppress warnings
        mfe = MFE(groups=["general", "statistical", "info-theory", "model-based", "landmarking"])
        try:
            mfe.fit(X_only.values, [0] * len(X_only))  # Assuming binary or multiclass labels
            names, values = mfe.extract()
            return pd.DataFrame([values], columns=names)
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Returns
        { "accuracy_recommendations": { fw → {algorithm, time_budget}, … } }
    """
    try:
        # ---------- parse incoming DataFrame ----------
        df = pd.read_json(StringIO(request.json["data"]))
        X = df.iloc[:, :-1] if df.shape[1] > 1 else df.copy()

        # basic NA handling so MFE doesn’t crash
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.columns.difference(num_cols)
        X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
        for c in cat_cols:
            X[c] = X[c].fillna(X[c].mode().iloc[0])

        # ---------- meta-feature extraction ----------
        mfe = MFE(groups=["general", "statistical", "info-theory"])
        mfe.fit(X.values, np.zeros(len(X)))           # dummy y
        names, feats = mfe.extract()
        meta_vec = pd.Series(feats, index=names).reindex(
                    SCALER.feature_names_in_, fill_value=0
                  ).values.reshape(1, -1)

        X_scaled = SCALER.transform(meta_vec)

        # ---------- query TabPFN models ----------
        recs = {}
        for fw, models in META_MODELS.items():
            alg  = models["algo"].predict(X_scaled)[0]
            tsec = int(models["time"].predict(X_scaled)[0])
            recs[fw] = {"algorithm": alg, "time_budget": tsec}

        return jsonify({"accuracy_recommendations": recs})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
def _cluster_score(X_, labels, which):
    import numpy as _np
    # need at least 2 clusters
    if _np.unique(labels).size < 2:
        # for silhouette & CH: invalid → very bad; for DB: higher-is-worse so return +inf before negation
        return -np.inf if which in ("silhouette", "calinski_harabasz") else np.inf
    if which == "silhouette":
        return float(silhouette_score(X_, labels))
    if which == "calinski_harabasz":
        return float(calinski_harabasz_score(X_, labels))
    if which == "davies_bouldin":
        # we maximize in the search loop, so negate DB (lower is better)
        return float(-davies_bouldin_score(X_, labels))
    return float(silhouette_score(X_, labels))
def _fit_one_candidate(X, framework, algo_name, metric_name, k_fixed=None, k_min=2, k_max=8,
                       time_budget_sec=60, n_trials=30, allow_hdbscan=True, seed=42, feature_names=None):
    """
    Run exactly one candidate and return a result dict compatible with the frontend.
    For frameworks:
      - "SklearnAutoCluster": run ONLY the requested algo_name (KMeans, MiniBatchKMeans, GaussianMixture, Agglomerative-*, Birch, HDBSCAN).
        If k_fixed is None and algo needs k, we pick the best within [k_min, k_max].
      - "Clusteval": run CE(cluster=algo_name) (algo_name in {"kmeans","agglomerative","dbscan","hdbscan"}).
      - "OptunaAutoCluster": run the full optuna search; algo_name is ignored (treated as "best").
    """
    from codecarbon import OfflineEmissionsTracker
    import numpy as _np

    # ---- emissions tracking per candidate ----
    tracker = OfflineEmissionsTracker(country_iso_code="EST", log_level="critical", allow_multiple_runs=True)
    tracker.start()
    tracker.start_task()

    labels = None
    chosen_algo = algo_name
    chosen_k = None

    try:
        if framework == "SklearnAutoCluster":
            # Map name variations to estimators or wrapped search
            # If algo needs k:
            needs_k = False
            def _fit_predict(est):
                est.fit(X)
                lab = getattr(est, "labels_", None)
                return est.predict(X) if lab is None else lab

            if algo_name in ("KMeans", "MiniBatchKMeans", "GaussianMixture", "Agglomerative", "Birch"):
                needs_k = True

            if needs_k and k_fixed is None:
                # choose best k within range, but only for THIS algo
                best = {"score": -_np.inf, "k": None, "labels": None}
                for k in range(int(k_min), int(k_max) + 1):
                    try:
                        if algo_name == "KMeans":
                            est = KMeans(n_clusters=k, n_init="auto", random_state=seed)
                            lab = _fit_predict(est)
                            algo_show = "KMeans"
                        elif algo_name == "MiniBatchKMeans":
                            est = MiniBatchKMeans(n_clusters=k, random_state=seed)
                            lab = _fit_predict(est)
                            algo_show = "MiniBatchKMeans"
                        elif algo_name == "GaussianMixture":
                            est = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed)
                            est.fit(X); lab = est.predict(X)
                            algo_show = "GaussianMixture-diag"
                        elif algo_name == "Agglomerative":
                            est = AgglomerativeClustering(n_clusters=k, linkage="ward")
                            lab = est.fit_predict(X)
                            algo_show = "Agglomerative-ward"
                        elif algo_name == "Birch":
                            est = Birch(n_clusters=k)
                            lab = est.fit_predict(X)
                            algo_show = "BIRCH"
                        else:
                            continue
                        s = _cluster_score(X, lab, metric_name)
                        if s > best["score"]:
                            best = {"score": s, "k": k, "labels": lab, "algo_show": algo_show}
                    except Exception:
                        continue
                if best["labels"] is None:
                    raise RuntimeError(f"{algo_name} failed in k-scan.")
                labels = best["labels"]; chosen_k = int(best["k"]); chosen_algo = best["algo_show"]
            else:
                # use fixed k or k-independent algo
                if algo_name == "KMeans":
                    est = KMeans(n_clusters=int(k_fixed), n_init="auto", random_state=42)
                    labels = _fit_predict(est); chosen_algo = "KMeans"; chosen_k = int(k_fixed)
                elif algo_name == "MiniBatchKMeans":
                    est = MiniBatchKMeans(n_clusters=int(k_fixed), random_state=42)
                    labels = _fit_predict(est); chosen_algo = "MiniBatchKMeans"; chosen_k = int(k_fixed)
                elif algo_name == "GaussianMixture":
                    est = GaussianMixture(n_components=int(k_fixed), covariance_type="diag", random_state=42)
                    est.fit(X); labels = est.predict(X); chosen_algo = "GaussianMixture-diag"; chosen_k = int(k_fixed)
                elif algo_name == "Agglomerative":
                    est = AgglomerativeClustering(n_clusters=int(k_fixed), linkage="ward")
                    labels = est.fit_predict(X); chosen_algo = "Agglomerative-ward"; chosen_k = int(k_fixed)
                elif algo_name == "Birch":
                    est = Birch(n_clusters=int(k_fixed))
                    labels = est.fit_predict(X); chosen_algo = "BIRCH"; chosen_k = int(k_fixed)
                elif algo_name == "HDBSCAN":
                    if _HAS_HDBSCAN:
                        est = _hdbscan.HDBSCAN()
                        labels = est.fit_predict(X); chosen_algo = "HDBSCAN"; chosen_k = None
                    else:
                        raise RuntimeError("HDBSCAN not available.")
                else:
                    raise RuntimeError(f"Unknown sklearn algo: {algo_name}")

        elif framework == "Clusteval":
            # algo_name: 'kmeans','agglomerative','dbscan','hdbscan'
            ce = CE(cluster=algo_name, evaluate="silhouette")
            ce.fit(X)
            labels = ce.results.get("labx")
            if labels is None:
                raise RuntimeError("clusteval returned no labels.")
            chosen_k = int(ce.results.get("optk", len(set(labels))))
            chosen_algo = algo_name

        elif framework == "OptunaAutoCluster":
            # full best-of search as a single competitor
            best = _optuna_autocluster(
                X, metric_name=metric_name, k_min=int(k_min), k_max=int(k_max),
                n_trials=int(n_trials), time_budget_sec=int(time_budget_sec), allow_hdbscan=True,  seed=seed,

            )
            if not best or best["labels"] is None:
                raise RuntimeError("OptunaAutoCluster failed to find clustering.")
            labels = _np.array(best["labels"])
            chosen_algo = best["algo"]
            chosen_k = int(best["params"]["k"]) if best["params"]["k"] is not None else None

        else:
            raise RuntimeError(f"Unknown framework {framework}")

        # metrics + embedding
        uniq = _np.unique(labels)
        if uniq.size >= 2:
            sil = float(silhouette_score(X, labels))
            ch  = float(calinski_harabasz_score(X, labels))
            db  = float(davies_bouldin_score(X, labels))
        else:
            sil = ch = db = -1.0

        pca2 = PCA(n_components=2, random_state=seed)
        emb2 = pca2.fit_transform(X)

        # ---- stop tracker and compute CO2/energy ----
        em = tracker.stop_task()
        try: tracker.stop()
        except: pass

        co2_micro = float(em.emissions) * 1_000_000
        energy_micro_wh = float(em.energy_consumed) * 1_000_000
        # Calculate feature importance for clustering
        feature_imp = calculate_clustering_feature_importance(X, labels, feature_names)
        
        return {
            "framework": framework,
            "algo": chosen_algo,
            "best_k": (int(chosen_k) if chosen_k is not None else None),
            "time_budget_sec": time_budget_sec,  
            "labels": labels.astype(int).tolist(),
            "embedding_2d": emb2.tolist(),
            "dr": "PCA(2)",
            "metrics": {
                "Silhouette": round(sil, 4),
                "Calinski-Harabasz": round(ch, 1),
                "Davies-Bouldin": round(db, 4),
            },
            "CO2 Emission": co2_micro,
            "Energy Consumption": energy_micro_wh,
            "feature_importance": feature_imp,
        }
    except Exception as e:
        # ensure tracker stops on error
        try:
            em = tracker.stop_task()
            tracker.stop()
        except:
            pass
        return {"error": str(e)}

def _try_configs(X, metric_name, k_min, k_max, time_budget_sec, allow_hdbscan, seed=42):
    start_ts = time.time()
    best = {"score": -np.inf, "details": None}

    def _timeout():
        return (time.time() - start_ts) > time_budget_sec

    algos = []
    for k in range(int(k_min), int(k_max) + 1):
        algos.extend([
            ("KMeans", k, KMeans(n_init="auto", random_state=seed)),
            ("MiniBatchKMeans", k, MiniBatchKMeans(random_state=seed)),
            ("GaussianMixture-full", k, GaussianMixture(covariance_type="full", random_state=seed)),
            ("GaussianMixture-diag", k, GaussianMixture(covariance_type="diag", random_state=seed)),
            ("Agglomerative-ward", k, AgglomerativeClustering(n_clusters=k, linkage="ward")),
            ("Agglomerative-average", k, AgglomerativeClustering(n_clusters=k, linkage="average")),
            ("BIRCH", k, Birch(n_clusters=k)),
        ])

    if allow_hdbscan and _HAS_HDBSCAN:
        algos.append(("HDBSCAN", None, _hdbscan.HDBSCAN()))

    for name, k, est in algos:
        if _timeout():
            break
        try:
            if name.startswith("GaussianMixture"):
                est.fit(X)
                labels = est.predict(X)
            else:
                est.fit(X)
                labels = getattr(est, "labels_", None)
                if labels is None:
                    labels = est.predict(X)

            s = _cluster_score(X, labels, metric_name)  # uses your existing _score in the route
            if s > best["score"]:
                best = {"score": s, "details": {"algo": name, "k": (int(k) if k is not None else None),
                                                 "labels": labels.astype(int).tolist()}}
        except Exception:
            continue

    return best
def _optuna_autocluster( X, metric_name, k_min, k_max, n_trials, time_budget_sec, allow_hdbscan, seed=42):
    start_ts = time.time()
    best = {"score": -np.inf, "algo": None, "params": None, "labels": None}

    def objective(trial):
        # Early stop on time
        if (time.time() - start_ts) > time_budget_sec:
            raise optuna.TrialPruned()

        algo = trial.suggest_categorical("algo", [
            "KMeans", "MiniBatchKMeans", "GaussianMixture", "Agglomerative", "Birch"
        ] + (["HDBSCAN"] if allow_hdbscan and _HAS_HDBSCAN else []))

        labels = None
        k = None

        try:
            if algo in ("KMeans", "MiniBatchKMeans", "GaussianMixture", "Agglomerative", "Birch"):
                k = trial.suggest_int("k", int(k_min), int(k_max))
            # --- Instantiate per algo ---
            if algo == "KMeans":
                est = KMeans(
                    n_clusters=k,
                    n_init="auto",
                    random_state=seed,
                )
                est.fit(X); labels = est.labels_
            elif algo == "MiniBatchKMeans":
                est = MiniBatchKMeans(
                    n_clusters=k,
                    random_state=seed,
                    batch_size=trial.suggest_categorical("mb_batch", [64, 128, 256, 512]),
                )
                est.fit(X); labels = est.labels_
            elif algo == "GaussianMixture":
                cov = trial.suggest_categorical("gm_cov", ["full", "diag"])
                est = GaussianMixture(
                    n_components=k,
                    covariance_type=cov,
                    random_state=seed
                )
                est.fit(X); labels = est.predict(X)
            elif algo == "Agglomerative":
                link = trial.suggest_categorical("agg_link", ["ward", "average", "complete"])
                # ward requires euclidean + no affinity arg in recent sklearn
                est = AgglomerativeClustering(n_clusters=k, linkage=link)
                labels = est.fit_predict(X)
            elif algo == "Birch":
                thresh = trial.suggest_float("birch_threshold", 0.3, 1.5, step=0.1)
                est = Birch(n_clusters=k, threshold=thresh)
                labels = est.fit_predict(X)
            elif algo == "HDBSCAN":
                mcs = trial.suggest_int("min_cluster_size", 5, 50)
                mns = trial.suggest_int("min_samples", 1, 20)
                est = _hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=mns)
                labels = est.fit_predict(X)
            else:
                raise optuna.TrialPruned()

            score = _cluster_score(X, labels, metric_name)
            # Track best
            nonlocal best
            if score > best["score"]:
                best = {
                    "score": float(score),
                    "algo": algo,
                    "params": {"k": int(k) if k is not None else None},
                    "labels": labels.astype(int).tolist(),
                }
            return float(score)
        except Exception:
            raise optuna.TrialPruned()

    study = optuna.create_study(direction="maximize")
    for t in range(int(n_trials)):
        if (time.time() - start_ts) > time_budget_sec:
            break
        try:
            study.optimize(objective, n_trials=1, catch=(Exception,))
        except optuna.TrialPruned:
            pass
        except Exception:
            pass

    return best
@app.route("/last_runs", methods=["GET"])
def last_runs():
    task = request.args.get("task", "clustering")
    return jsonify(RUN_LOG.get(task, []))
@app.route("/interpretability_eval", methods=["POST"])
def interpretability_eval():
    """
    Compare multiple explanation techniques across interpretability dimensions.

    Returns JSON:
      {
        "techniques": [
          {
            "name": "permutation",
            "dimensions": {"fidelity":0.7, "stability":0.6, ...},
            "iem": 0.65
          },
          ...
        ],
        "meta": {"n_features": 12, "n_eval": 50}
      }
    """
    try:
        import numpy as np
        import pandas as pd
        from io import StringIO

        from sklearn.model_selection import train_test_split
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import r2_score
        from sklearn.linear_model import Ridge

        data = request.json or {}
        dataset_json = data.get("data")
        framework = data.get("framework", "FLAML")
        algorithm = data.get("algorithm", None)

        n_eval = int(data.get("n_eval", 50))
        n_perturb = int(data.get("n_perturb", 200))

        weights = data.get("weights", {}) or {}
        default_dims = ["fidelity", "stability", "compactness", "correctness", "simplicity", "completeness"]
        w = {d: float(weights.get(d, 1.0)) for d in default_dims}
        w_sum = sum(w.values()) if sum(w.values()) > 0 else 1.0

        if dataset_json is None:
            return jsonify({"error": "Missing 'data' in request."}), 400
        if algorithm is None:
            return jsonify({"error": "Missing 'algorithm' in request."}), 400
        if framework not in framework_algorithms:
            return jsonify({"error": f"Unknown framework '{framework}'."}), 400
        if algorithm not in framework_algorithms[framework]:
            return jsonify({"error": f"Unknown algorithm '{algorithm}' for framework '{framework}'."}), 400

        df = pd.read_json(StringIO(dataset_json))
        if df.shape[1] < 2:
            return jsonify({"error": "Dataset must have at least 1 feature and 1 target column."}), 400

        response_col = df.columns[-1]
        y_series = df[response_col]
        X_df = df.drop(columns=[response_col]).copy()

        numeric_cols = X_df.select_dtypes(include=["float64", "int64"]).columns
        categorical_cols = X_df.select_dtypes(include=["object"]).columns

        imputer = SimpleImputer(strategy="mean")
        if len(numeric_cols) > 0:
            X_df[numeric_cols] = imputer.fit_transform(X_df[numeric_cols])
            scaler = StandardScaler()
            X_df[numeric_cols] = scaler.fit_transform(X_df[numeric_cols])

        if len(categorical_cols) > 0:
            cat_imp = SimpleImputer(strategy="most_frequent")
            X_df[categorical_cols] = cat_imp.fit_transform(X_df[categorical_cols])
            for col in categorical_cols:
                X_df[col] = LabelEncoder().fit_transform(X_df[col])

        df2 = pd.concat([X_df, y_series], axis=1).drop_duplicates()
        X = df2.iloc[:, :-1]
        y = df2.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if y.nunique() > 1 else None
        )
        model = framework_algorithms[
            framework
        ][algorithm]

        model.fit(
            X_train,
            y_train
        )

        n_features = X.shape[1]
        feat_means = X_train.mean(axis=0).values

        rng = np.random.RandomState(42)
        idx = np.arange(X_test.shape[0])
        if X_test.shape[0] > n_eval:
            idx = rng.choice(idx, size=n_eval, replace=False)
        X_eval = X_test.iloc[idx].copy()

        def _safe_predict_proba(m, X_):
            if hasattr(m, "predict_proba"):
                return m.predict_proba(X_)
            preds = m.predict(X_)
            classes = np.unique(y_train)
            proba = np.zeros((len(preds), len(classes)))
            class_to_i = {c: i for i, c in enumerate(classes)}
            for i, p in enumerate(preds):
                proba[i, class_to_i.get(p, 0)] = 1.0
            return proba

        def _get_expl_perm():
            pi = permutation_importance(model, X_eval, y_test.iloc[idx], n_repeats=5, random_state=42, scoring="accuracy")
            v = np.abs(pi.importances_mean)
            if v.max() > 0:
                v = v / v.max()
            return v

        def _get_expl_local_surrogate():
            X_eval_np = X_eval.values
            expls = []
            r2s = []
            for row in X_eval_np:
                base = row.reshape(1, -1)
                base_proba = _safe_predict_proba(model, base)[0]
                base_class = int(np.argmax(base_proba))

                Z = rng.normal(loc=base, scale=0.5, size=(n_perturb, n_features))
                Z_df = pd.DataFrame(Z, columns=X_eval.columns)
                y_hat = _safe_predict_proba(model, Z_df)[:, base_class]

                reg = Ridge(alpha=1.0, random_state=42)
                reg.fit(Z, y_hat)
                pred = reg.predict(Z)
                r2s.append(float(r2_score(y_hat, pred)))

                coefs = np.abs(reg.coef_)
                expls.append(coefs)

            v = np.mean(np.vstack(expls), axis=0)
            if v.max() > 0:
                v = v / v.max()
            return v, float(np.mean(r2s))

        def _entropy(p):
            p = p[p > 0]
            return float(-(p * np.log(p)).sum())

        def _simplicity_score(v):
            s = v.sum()
            if s <= 0:
                return 0.0
            p = v / s
            H = _entropy(p)
            Hmax = np.log(len(v))
            if Hmax <= 0:
                return 0.0
            return float(1.0 - (H / Hmax))

        def _compactness_score(v, thresh=0.9):
            s = v.sum()
            if s <= 0:
                return 0.0
            p = np.sort(v / s)[::-1]
            c = np.cumsum(p)
            k = int(np.searchsorted(c, thresh) + 1)
            if len(v) <= 1:
                return 1.0
            return float(1.0 - (k - 1) / (len(v) - 1))

        def _fidelity_aopc(v, topk=5):
            if hasattr(model, "predict_proba"):
                base = model.predict_proba(X_eval)
                base_cls = np.argmax(base, axis=1)
                base_score = base[np.arange(base.shape[0]), base_cls]
            else:
                base_score = np.ones(len(X_eval))

            topk = min(topk, len(v))
            order = np.argsort(v)[::-1][:topk]

            X_mask = X_eval.values.copy()
            X_mask[:, order] = feat_means[order]
            X_mask_df = pd.DataFrame(X_mask, columns=X_eval.columns)

            if hasattr(model, "predict_proba"):
                masked = model.predict_proba(X_mask_df)
                masked_score = masked[np.arange(masked.shape[0]), base_cls]
            else:
                masked_score = np.ones(len(X_eval))

            drop = np.maximum(0.0, base_score - masked_score)
            denom = (np.maximum(1e-9, np.mean(base_score)))
            return float(np.clip(np.mean(drop) / denom, 0.0, 1.0))

        def _stability_cosine(v, noise_std=0.05, reps=10):
            X0 = X_eval.values
            sims = []
            for _ in range(reps):
                Xn = X0 + rng.normal(0.0, noise_std, size=X0.shape)
                Xn_df = pd.DataFrame(Xn, columns=X_eval.columns)

                pi = permutation_importance(model, Xn_df, y_test.iloc[idx], n_repeats=3, random_state=42, scoring="accuracy")
                vn = np.abs(pi.importances_mean)
                if vn.max() > 0:
                    vn = vn / vn.max()

                a = v / (np.linalg.norm(v) + 1e-12)
                b = vn / (np.linalg.norm(vn) + 1e-12)
                sims.append(float(np.clip(np.dot(a, b), -1.0, 1.0)))
            return float(np.clip(np.mean(sims), 0.0, 1.0))

        def _correctness_infid(v, m=20):
            X0 = X_eval.values
            errs = []
            for row in X0:
                base = row.reshape(1, -1)
                base_proba = _safe_predict_proba(model, base)[0]
                base_class = int(np.argmax(base_proba))
                fx = float(base_proba[base_class])

                for _ in range(m):
                    delta = rng.normal(0.0, 0.2, size=(n_features,))
                    x2 = row - delta
                    x2_df = pd.DataFrame([x2], columns=X_eval.columns)
                    fx2 = float(_safe_predict_proba(model, x2_df)[0, base_class])

                    approx = float(np.dot(v, delta))
                    errs.append((fx - fx2 - approx) ** 2)

            infid = float(np.mean(errs)) if errs else 1.0
            score = 1.0 / (1.0 + infid)
            return float(np.clip(score, 0.0, 1.0))

        techniques = []
        def _top_features(v, feature_names, top_k=10):
            v = np.array(v).reshape(-1)
            top_k = int(min(top_k, len(v)))
            order = np.argsort(v)[::-1][:top_k]
            out = []
            for j in order:
                out.append({
                    "feature": str(feature_names[j]),
                    "importance": float(v[j]),
                })
            return out

        v_perm = _get_expl_perm()
        dims_perm = {
            "fidelity": _fidelity_aopc(v_perm),
            "stability": _stability_cosine(v_perm),
            "compactness": _compactness_score(v_perm),
            "correctness": _correctness_infid(v_perm),
            "simplicity": _simplicity_score(v_perm),
            "completeness": 0.5,
        }
        iem_perm = sum(w[d] * dims_perm[d] for d in default_dims) / w_sum
        techniques.append({
            "name": "permutation",
            "dimensions": dims_perm,
            "iem": float(iem_perm),
            "artifacts": {
                "top_features": _top_features(v_perm, X_eval.columns, top_k=10)
            }
        })


        v_surr, surr_r2 = _get_expl_local_surrogate()
        dims_surr = {
            "fidelity": _fidelity_aopc(v_surr),
            "stability": 0.5,
            "compactness": _compactness_score(v_surr),
            "correctness": _correctness_infid(v_surr),
            "simplicity": _simplicity_score(v_surr),
            "completeness": float(np.clip((surr_r2 + 1.0) / 2.0, 0.0, 1.0)),
        }
        iem_surr = sum(w[d] * dims_surr[d] for d in default_dims) / w_sum
                # SHAP technique (optional). Uses shap if installed.
        # References:
        # - Lundberg and Lee, 2017, "A Unified Approach to Interpreting Model Predictions" (SHAP)
        # - SHAP "local accuracy" (additive completeness) is used for the completeness proxy below.
        try:
            import shap

            # Build an explainer. This works for many sklearn models.
            # For some models, shap may fall back to a model-agnostic approach.
            explainer = shap.Explainer(model, X_train)

            shap_out = explainer(X_eval)

            # shap_out.values can be:
            # - (n_samples, n_features) for single output
            # - (n_samples, n_features, n_classes) for multiclass
            values = getattr(shap_out, "values", None)
            base_values = getattr(shap_out, "base_values", None)

            if values is None:
                raise RuntimeError("SHAP output has no values.")

            values = np.array(values)

            # Convert to a global importance vector v_shap in [0, 1]
            if values.ndim == 2:
                # Binary with single output or regression-like output
                v_shap = np.mean(np.abs(values), axis=0)
            elif values.ndim == 3:
                # Multiclass. Take predicted class per sample and use corresponding SHAP vector.
                proba = _safe_predict_proba(model, X_eval)
                pred_cls = np.argmax(proba, axis=1)
                picked = values[np.arange(values.shape[0]), :, pred_cls]
                v_shap = np.mean(np.abs(picked), axis=0)
            else:
                raise RuntimeError(f"Unexpected SHAP values shape: {values.shape}")

            if v_shap.max() > 0:
                v_shap = v_shap / v_shap.max()

            # Completeness proxy for SHAP.
            # We measure how well base_value + sum(shap) reconstructs the model score (local accuracy idea).
            # Convert reconstruction error into a bounded score in [0, 1].
            completeness_score = 0.5
            try:
                if hasattr(model, "predict_proba"):
                    proba = _safe_predict_proba(model, X_eval)
                    pred_cls = np.argmax(proba, axis=1)
                    fx = proba[np.arange(proba.shape[0]), pred_cls]

                    if values.ndim == 2:
                        # Single output. Use base_values as scalar or per sample if provided.
                        if base_values is None:
                            base = np.mean(fx)
                        else:
                            base = np.array(base_values).reshape(-1)
                            if base.shape[0] == 1:
                                base = np.repeat(base[0], len(fx))
                        recon = base + np.sum(values, axis=1)
                    else:
                        # Multiclass. Base and values per class.
                        base = np.array(base_values)
                        if base.ndim == 1:
                            base = base.reshape(-1, 1)

                        # Take base for predicted class
                        base_pred = base[np.arange(base.shape[0]), pred_cls]
                        picked = values[np.arange(values.shape[0]), :, pred_cls]
                        recon = base_pred + np.sum(picked, axis=1)

                    mae = float(np.mean(np.abs(fx - recon)))
                    completeness_score = float(np.clip(1.0 / (1.0 + mae), 0.0, 1.0))
            except Exception:
                completeness_score = 0.5

            # Stability proxy for SHAP.
            # Quick robustness check: compare SHAP global vector under small noise once.
            stability_score = 0.5
            try:
                Xn = X_eval.values + rng.normal(0.0, 0.05, size=X_eval.values.shape)
                Xn_df = pd.DataFrame(Xn, columns=X_eval.columns)
                shap_out_n = explainer(Xn_df)
                v_n = np.array(getattr(shap_out_n, "values", None))
                if v_n is None:
                    raise RuntimeError("No SHAP values on noisy input.")

                v_n = np.array(v_n)
                if v_n.ndim == 2:
                    v_n = np.mean(np.abs(v_n), axis=0)
                elif v_n.ndim == 3:
                    proba_n = _safe_predict_proba(model, Xn_df)
                    pred_cls_n = np.argmax(proba_n, axis=1)
                    picked_n = v_n[np.arange(v_n.shape[0]), :, pred_cls_n]
                    v_n = np.mean(np.abs(picked_n), axis=0)

                if v_n.max() > 0:
                    v_n = v_n / v_n.max()

                a = v_shap / (np.linalg.norm(v_shap) + 1e-12)
                b = v_n / (np.linalg.norm(v_n) + 1e-12)
                stability_score = float(np.clip(np.dot(a, b), 0.0, 1.0))
            except Exception:
                stability_score = 0.5

            dims_shap = {
                "fidelity": _fidelity_aopc(v_shap),
                "stability": stability_score,
                "compactness": _compactness_score(v_shap),
                "correctness": _correctness_infid(v_shap),
                "simplicity": _simplicity_score(v_shap),
                "completeness": completeness_score,
            }
            iem_shap = sum(w[d] * dims_shap[d] for d in default_dims) / w_sum
            techniques.append({
                "name": "shap",
                "dimensions": dims_shap,
                "iem": float(iem_shap),
                "artifacts": {
                    "top_features": _top_features(v_shap, X_eval.columns, top_k=10)
                }
            })


        except ImportError:
            # SHAP not installed. Skip without failing the endpoint.
            pass
        except Exception:
            # Any SHAP runtime issue. Skip without failing the endpoint.
            pass

        techniques.append({
            "name": "local_surrogate",
            "dimensions": dims_surr,
            "iem": float(iem_surr),
            "artifacts": {
                "top_features": _top_features(v_surr, X_eval.columns, top_k=10),
                "surrogate_r2": float(surr_r2),
                "n_perturb": int(n_perturb)
            }
        })


        response_payload = {
            "framework": framework,
            "algorithm": algorithm,
            "techniques": techniques,
            "meta": {
                "n_features": int(n_features),
                "n_eval": int(len(X_eval))
            }
        }

        _log_run(
            "interpretability",
            framework,
            {
                "algorithm": algorithm,
                "n_eval": n_eval,
                "n_perturb": n_perturb,
                "weights": w
            },
            response_payload
        )

        return jsonify(response_payload)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/explain_instance", methods=["POST"])
def explain_instance():
    """
    Local explanation for one row in the uploaded dataset.

    Input JSON:
      {
        "data": "<df.to_json()>",
        "framework": "FLAML",
        "algorithm": "Extra Trees",
        "row_index": 0,
        "n_perturb": 200,
        "top_k": 10
      }

    Output JSON:
      {
        "meta": {...},
        "prediction": {"pred_class": ..., "proba": [...]},
        "local": {
          "shap": [{"feature":..., "contribution":...}, ...],
          "occlusion": [{"feature":..., "drop":...}, ...],
          "local_surrogate": [{"feature":..., "weight":...}, ...],
          "top_feature_values": [{"feature":..., "value":...}, ...]
        }
      }
    """
    try:
        import numpy as np
        import pandas as pd
        from io import StringIO

        from sklearn.model_selection import train_test_split
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.linear_model import Ridge

        data = request.json or {}
        dataset_json = data.get("data")
        framework = data.get("framework", "FLAML")
        algorithm = data.get("algorithm", None)

        row_index = int(data.get("row_index", 0))
        n_perturb = int(data.get("n_perturb", 200))
        top_k = int(data.get("top_k", 10))

        if dataset_json is None:
            return jsonify({"error": "Missing 'data' in request."}), 400
        if algorithm is None:
            return jsonify({"error": "Missing 'algorithm' in request."}), 400
        if framework not in framework_algorithms:
            return jsonify({"error": f"Unknown framework '{framework}'."}), 400
        if algorithm not in framework_algorithms[framework]:
            return jsonify({"error": f"Unknown algorithm '{algorithm}' for framework '{framework}'."}), 400

        df = pd.read_json(StringIO(dataset_json))
        if df.shape[1] < 2:
            return jsonify({"error": "Dataset must have at least 1 feature and 1 target column."}), 400

        # Use last column as target, consistent with interpretability_eval
        response_col = df.columns[-1]
        y_series = df[response_col]
        X_df = df.drop(columns=[response_col]).copy()

        if row_index < 0 or row_index >= len(X_df):
            return jsonify({"error": f"row_index out of range: {row_index}"}), 400

        # Preprocess exactly as in interpretability_eval
        numeric_cols = X_df.select_dtypes(include=["float64", "int64"]).columns
        categorical_cols = X_df.select_dtypes(include=["object"]).columns

        imputer = SimpleImputer(strategy="mean")
        if len(numeric_cols) > 0:
            X_df[numeric_cols] = imputer.fit_transform(X_df[numeric_cols])
            scaler = StandardScaler()
            X_df[numeric_cols] = scaler.fit_transform(X_df[numeric_cols])

        if len(categorical_cols) > 0:
            cat_imp = SimpleImputer(strategy="most_frequent")
            X_df[categorical_cols] = cat_imp.fit_transform(X_df[categorical_cols])
            for col in categorical_cols:
                X_df[col] = LabelEncoder().fit_transform(X_df[col])

        df2 = pd.concat([X_df, y_series], axis=1).drop_duplicates()
        X = df2.iloc[:, :-1]
        y = df2.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
        )

        model = framework_algorithms[framework][algorithm]
        model.fit(X_train, y_train)

        # Instance vector
        x0 = X_df.iloc[row_index:row_index + 1].copy()

        # Helper predict_proba fallback
        def _safe_predict_proba(m, X_):
            if hasattr(m, "predict_proba"):
                return m.predict_proba(X_)
            preds = m.predict(X_)
            classes = np.unique(y_train)
            proba = np.zeros((len(preds), len(classes)))
            class_to_i = {c: i for i, c in enumerate(classes)}
            for i, p in enumerate(preds):
                proba[i, class_to_i.get(p, 0)] = 1.0
            return proba

        proba0 = _safe_predict_proba(model, x0)[0]
        pred_class_idx = int(np.argmax(proba0))
        pred_class = int(np.unique(y_train)[pred_class_idx]) if len(np.unique(y_train)) == len(proba0) else pred_class_idx
        base_score = float(proba0[pred_class_idx])

        # Compute SHAP local contributions (top_k)
        shap_items = []
        try:
            import shap
            explainer = shap.Explainer(model, X_train)
            shap_out = explainer(x0)
            values = np.array(getattr(shap_out, "values", None))

            if values is None:
                raise RuntimeError("No SHAP values.")

            # Handle multiclass shapes
            if values.ndim == 2:
                v_local = values[0]
            elif values.ndim == 3:
                v_local = values[0, :, pred_class_idx]
            else:
                raise RuntimeError(f"Unexpected SHAP values shape: {values.shape}")

            order = np.argsort(np.abs(v_local))[::-1][:min(top_k, len(v_local))]
            for j in order:
                shap_items.append({
                    "feature": str(x0.columns[j]),
                    "contribution": float(v_local[j]),
                })
        except Exception:
            shap_items = []

        # Occlusion sensitivity (mask-one-feature) using top features from SHAP if available
        feat_means = X_train.mean(axis=0).values
        occlusion_items = []
        if shap_items:
            top_feats = [it["feature"] for it in shap_items]
            for fname in top_feats:
                j = list(x0.columns).index(fname)
                xm = x0.copy()
                xm.iloc[0, j] = feat_means[j]
                pm = _safe_predict_proba(model, xm)[0]
                masked_score = float(pm[pred_class_idx])
                occlusion_items.append({
                    "feature": str(fname),
                    "drop": float(max(0.0, base_score - masked_score)),
                })

            occlusion_items = sorted(occlusion_items, key=lambda z: z["drop"], reverse=True)
        else:
            occlusion_items = []

        # Local surrogate (LIME-style) around x0
        rng = np.random.RandomState(42)
        n_features = x0.shape[1]
        base = x0.values.reshape(1, -1)

        Z = rng.normal(loc=base, scale=0.5, size=(n_perturb, n_features))
        Z_df = pd.DataFrame(Z, columns=x0.columns)
        y_hat = _safe_predict_proba(model, Z_df)[:, pred_class_idx]

        reg = Ridge(alpha=1.0, random_state=42)
        reg.fit(Z, y_hat)
        coefs = reg.coef_

        order = np.argsort(np.abs(coefs))[::-1][:min(top_k, len(coefs))]
        surrogate_items = []
        for j in order:
            surrogate_items.append({
                "feature": str(x0.columns[j]),
                "weight": float(coefs[j]),
            })

        # Return top feature values for quick inspection (match SHAP top if available, else surrogate top)
        top_feat_names = [it["feature"] for it in shap_items] if shap_items else [it["feature"] for it in surrogate_items]
        top_values = []
        for fname in top_feat_names:
            j = list(x0.columns).index(fname)
            top_values.append({
                "feature": str(fname),
                "value": float(x0.iloc[0, j]),
            })

        return jsonify({
            "meta": {
                "framework": framework,
                "algorithm": algorithm,
                "row_index": int(row_index),
                "top_k": int(top_k),
                "n_perturb": int(n_perturb)
            },
            "prediction": {
                "pred_class_index": int(pred_class_idx),
                "pred_class": pred_class,
                "proba": [float(x) for x in proba0.tolist()]
            },
            "local": {
                "shap": shap_items,
                "occlusion": occlusion_items,
                "local_surrogate": surrogate_items,
                "top_feature_values": top_values
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run_clustering_automl', methods=['POST'])
def run_clustering_automl():
    """Clustering AutoML: Clusteval-only."""
    try:
        data = request.json or {}
        framework = data.get("framework", "Clusteval")
        k_min, k_max = data.get("k_range", [2, 8])
        metric = data.get("metric", "silhouette")
        time_budget_sec = int(data.get("time_budget_sec", 60))
        seed = int(data.get("seed", 42))         
        np.random.seed(seed)
        random.seed(seed)
        # Load data
        df = pd.read_json(StringIO(data["data"]))
        # Basic preprocessing: OHE categoricals, scale numerics
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", StandardScaler(), num_cols),
            ],
            remainder="drop"
        )
        X = pre.fit_transform(df)
        if X.shape[0] < 3:
            return jsonify({"message": "Need at least 3 rows for clustering."}), 400

        # Metric function
        def _score(X_, labels, which):
            # need at least 2 clusters
            if np.unique(labels).size < 2:
                return -np.inf if which in ("silhouette", "calinski_harabasz") else np.inf
            if which == "silhouette":
                return float(silhouette_score(X_, labels))
            if which == "calinski_harabasz":
                return float(calinski_harabasz_score(X_, labels))
            if which == "davies_bouldin":
                return float(-davies_bouldin_score(X_, labels))  # negate so higher=better
            return float(silhouette_score(X_, labels))

        # ---- Framework switch ----
        if framework == "Clusteval":
            try:
                # Start tracker for the whole clusteval search
                tracker = OfflineEmissionsTracker(
                    country_iso_code="EST",
                    log_level="critical",
                    allow_multiple_runs=True
                )
                tracker.start()
                tracker.start_task()

                # We'll try a small pool of algorithms and pick the best by your chosen metric.
                algos = ["kmeans", "agglomerative"]
                if _HAS_HDBSCAN:
                    algos += ["dbscan", "hdbscan"]

                best = {"score": -np.inf, "labels": None, "algo": None, "k": None}
                for algo in algos:
                    ce = CE(cluster=algo, evaluate="silhouette")  # ce chooses k for k-based algos
                    ce.fit(X)
                    labels = ce.results.get("labx")
                    if labels is None:
                        continue
                    score = _cluster_score(X, labels, metric)
                    if score > best["score"]:
                        best["score"] = score
                        best["labels"] = labels
                        best["algo"] = algo
                        best["k"] = int(ce.results.get("optk", len(set(labels))))

                if best["labels"] is None:
                    # always stop tracker before returning
                    em = tracker.stop_task()
                    try: tracker.stop()
                    except: pass
                    return jsonify({"message": "Clusteval could not form valid clusters."}), 400

                labels = best["labels"]
                sil = silhouette_score(X, labels) if len(set(labels)) >= 2 else -1
                ch  = calinski_harabasz_score(X, labels) if len(set(labels)) >= 2 else -1
                db  = davies_bouldin_score(X, labels) if len(set(labels)) >= 2 else -1
                pca = PCA(n_components=2, random_state=seed).fit_transform(X)

                # Stop tracker and compute sustainability metrics
                em = tracker.stop_task()
                try: tracker.stop()
                except: pass

                co2_micro = float(em.emissions) * 1_000_000           # kg -> µkg
                energy_micro_wh = float(em.energy_consumed) * 1_000_000  # kWh -> µWh

                out = {
                    "labels": list(map(int, labels)),
                    "best_k": int(best["k"]),
                    "framework": "Clusteval",
                    "algo": best["algo"],
                    "dr": "auto",
                    "metrics": {
                        "Silhouette": round(float(sil), 4),
                        "Calinski-Harabasz": round(float(ch), 1),
                        "Davies-Bouldin": round(float(db), 4)
                    },
                    "embedding_2d": pca.tolist(),
                    "CO2 Emission": co2_micro,
                    "Energy Consumption": energy_micro_wh,
                }
                _log_run(
                    "clustering",
                    "Clusteval",
                    {"metric": metric, "k_range": [int(k_min), int(k_max)], "time_budget_sec": time_budget_sec},
                    out
                )
                return jsonify(out)

            except Exception as e:
                return jsonify({"message": f"Clusteval error: {e}"}), 500
        elif framework == "SklearnAutoCluster":
            # Start tracking for the whole sklearn search
            tracker = OfflineEmissionsTracker(
                country_iso_code="EST",
                log_level="critical",
                allow_multiple_runs=True
            )
            tracker.start()
            tracker.start_task()

            best = _try_configs(
                X,
                metric_name=metric,
                k_min=int(k_min),
                k_max=int(k_max),
                time_budget_sec=int(time_budget_sec),
                allow_hdbscan=True, 
                seed=seed, 
            )
            if not best or not best["details"]:
                em = tracker.stop_task()
                try: tracker.stop()
                except: pass
                return jsonify({"message": "No valid clustering found."}), 400

            labels = np.array(best["details"]["labels"])
            pca2 = PCA(n_components=2, random_state=seed)
            emb2 = pca2.fit_transform(X)

            if np.unique(labels).size >= 2:
                sil = float(silhouette_score(X, labels))
                ch  = float(calinski_harabasz_score(X, labels))
                db  = float(davies_bouldin_score(X, labels))
            else:
                sil = ch = db = -1.0

            # Stop tracker and compute sustainability
            em = tracker.stop_task()
            try: tracker.stop()
            except: pass

            co2_micro = float(em.emissions) * 1_000_000
            energy_micro_wh = float(em.energy_consumed) * 1_000_000

            out = {
                "framework": "SklearnAutoCluster",
                "algo": best["details"]["algo"],
                "best_k": (int(best["details"]["k"]) if best["details"]["k"] is not None else None),
                "dr": "PCA(2)",
                "metrics": {
                    "Silhouette": round(sil, 4),
                    "Calinski-Harabasz": round(ch, 1),
                    "Davies-Bouldin": round(db, 4)
                },
                "labels": labels.astype(int).tolist(),
                "embedding_2d": emb2.tolist(),
                "CO2 Emission": co2_micro,
                "Energy Consumption": energy_micro_wh,
            }
            _log_run(
                "clustering",
                "SklearnAutoCluster",
                {"metric": metric, "k_range": [int(k_min), int(k_max)], "time_budget_sec": time_budget_sec},
                out
            )
            return jsonify(out)


        elif framework == "OptunaAutoCluster":
            # Track the whole Optuna search
            tracker = OfflineEmissionsTracker(
                country_iso_code="EST",
                log_level="critical",
                allow_multiple_runs=True
            )
            tracker.start()
            tracker.start_task()

            n_trials = int(data.get("n_trials", 30))
            best = _optuna_autocluster(
                X,
                metric_name=metric,
                k_min=int(k_min),
                k_max=int(k_max),
                n_trials=n_trials,
                time_budget_sec=int(time_budget_sec),
                allow_hdbscan=True, 
                seed=seed,
            )
            if not best or best["labels"] is None:
                em = tracker.stop_task()
                try: tracker.stop()
                except: pass
                return jsonify({"message": "OptunaAutoCluster failed to find a valid clustering."}), 400

            labels = np.array(best["labels"])
            pca2 = PCA(n_components=2, random_state=seed)
            emb2 = pca2.fit_transform(X)

            if np.unique(labels).size >= 2:
                sil = float(silhouette_score(X, labels))
                ch  = float(calinski_harabasz_score(X, labels))
                db  = float(davies_bouldin_score(X, labels))
            else:
                sil = ch = db = -1.0

            # Stop tracker and compute sustainability
            em = tracker.stop_task()
            try: tracker.stop()
            except: pass

            co2_micro = float(em.emissions) * 1_000_000
            energy_micro_wh = float(em.energy_consumed) * 1_000_000
            out = {
                "framework": "OptunaAutoCluster",
                "algo": best["algo"],
                "best_k": (int(best["params"]["k"]) if best["params"]["k"] is not None else None),
                "dr": "PCA(2)",
                "metrics": {
                    "Silhouette": round(sil, 4),
                    "Calinski-Harabasz": round(ch, 1),
                    "Davies-Bouldin": round(db, 4)
                },
                "labels": labels.astype(int).tolist(),
                "embedding_2d": emb2.tolist(),
                "CO2 Emission": co2_micro,
                "Energy Consumption": energy_micro_wh,
            }
            _log_run(
                "clustering",
                "OptunaAutoCluster",
                {"metric": metric, "k_range": [int(k_min), int(k_max)], "time_budget_sec": time_budget_sec},
                out
            )
            return jsonify(out)
    except Exception as e:
        return jsonify({"message": f"Clustering AutoML error: {e}"}), 500
@app.route("/run_clustering_compare", methods=["POST"])
def run_clustering_compare():
    """
    Body:
      {
        "data": "<df.to_json()>",
        "frameworks": ["SklearnAutoCluster","Clusteval","OptunaAutoCluster"],
        "algorithms": {
            "SklearnAutoCluster": {"KMeans": true, "Agglomerative": true, "HDBSCAN": true},
            "Clusteval": {"kmeans": true, "agglomerative": true},
            "OptunaAutoCluster": {"best": true}
        },
        "metric": "silhouette",
        "k_fixed": 4,               # optional; if omitted we search k for that algo
        "k_range": [2, 8],
        "time_budget_sec": 60,
        "n_trials": 30
      }
    Returns: {"results": { "<fw>_<algo>": result_dict, ... }}
    """
    try:
        data = request.json
        df = pd.read_json(StringIO(data["data"]))
        X = df.select_dtypes(include=[np.number])
        if X.shape[1] == 0:
            return jsonify({"message": "No numeric columns for clustering."}), 400

        frameworks = data.get("frameworks", [])
        algos_map  = data.get("algorithms", {})
        metric     = data.get("metric", "silhouette")
        k_fixed    = data.get("k_fixed", None)
        k_min, k_max = (data.get("k_range") or [2, 8])
        time_budget_sec = int(data.get("time_budget_sec", 60))
        n_trials  = int(data.get("n_trials", 30))
        seed = int(data.get("seed", 42))          # ← NEW
        np.random.seed(seed)
        random.seed(seed)
        results = {}
        for fw in frameworks:
            selected = algos_map.get(fw, {})
            for algo_name, enabled in selected.items():
                if not enabled:
                    continue
                key = f"{fw}_{algo_name}"
                res = _fit_one_candidate(
                    X.values, fw, algo_name, metric_name=metric,
                    k_fixed=k_fixed, k_min=int(k_min), k_max=int(k_max),
                    time_budget_sec=time_budget_sec, n_trials=n_trials, seed=seed,
                    feature_names=list(X.columns),
                )
                results[key] = res

        # Store compact run metadata
        try:
            _log_run("clustering", "compare",
                     {"metric": metric, "k_range": [int(k_min), int(k_max)], "k_fixed": k_fixed,
                      "time_budget_sec": time_budget_sec, "frameworks": frameworks},
                     {"results": list(results.keys())})
        except Exception:
            pass

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"message": f"compare failed: {e}"}), 500

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

