from __future__ import annotations

import json
import pickle
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from ablation_common import (
    align_columns,
    compute_derived_features,
    load_base_feature_names,
    load_label_encoder_34,
    project_root,
    validate_columns,
    workspace_root,
)


class AblationReviewer3InferenceEngine:
    """Reviewer-3 artifact loader for raw 46-feature hierarchical inference."""

    def __init__(self, project_dir: Path | None = None, workspace_dir: Path | None = None):
        self.project_dir = Path(project_dir) if project_dir is not None else project_root()
        self.workspace_dir = Path(workspace_dir) if workspace_dir is not None else workspace_root()
        self.label_encoder_34 = load_label_encoder_34()
        self.label_names_34 = list(self.label_encoder_34.classes_)
        self.base_features = load_base_feature_names()
        self._load_binary()
        self._load_multiclass()
        self._load_subclass()
        self._force_cpu()
        self._validate_contracts()

    def _load_binary(self) -> None:
        config_path = self.project_dir / "results" / "reviewer3_binary" / "xgboost_final_operational_config.json"
        protocol_path = self.project_dir / "results" / "reviewer3_binary" / "binary_stage_feature_protocol.json"
        model_path = self.project_dir / "models" / "reviewer3_binary" / "xgboost_optuna40.joblib"
        scaler_path = self.workspace_dir / "processed_data" / "scaler.joblib"

        with config_path.open("r", encoding="utf-8") as f:
            self.binary_config = json.load(f)
        with protocol_path.open("r", encoding="utf-8") as f:
            self.binary_protocol = json.load(f)

        self.binary_model = joblib.load(model_path)
        self.binary_scaler = joblib.load(scaler_path)
        self.binary_features = list(self.binary_protocol["feature_names"])
        self.binary_threshold = float(self.binary_config["selected_threshold"])

    def _load_multiclass(self) -> None:
        pkg_path = self.project_dir / "models" / "reviewer3_multiclass" / "multiclass_xgb_threshold_tuned.pkl"
        protocol_path = self.project_dir / "results" / "reviewer3_multiclass" / "multiclass_xgb_protocol.json"
        with pkg_path.open("rb") as f:
            self.multiclass_pkg = pickle.load(f)
        with protocol_path.open("r", encoding="utf-8") as f:
            self.multiclass_protocol = json.load(f)

        self.multiclass_model = self.multiclass_pkg["model"]
        self.multiclass_scaler = self.multiclass_pkg["scaler"]
        self.multiclass_feature_columns = list(self.multiclass_pkg["feature_columns"])
        self.multiclass_model_features = list(self.multiclass_pkg["model_features"])
        self.multiclass_thresholds = np.asarray(self.multiclass_pkg["thresholds"], dtype=float)
        self.attack_names = list(self.multiclass_pkg["attack_names"])
        self.reverse_map = {int(k): int(v) for k, v in self.multiclass_pkg["reverse_map"].items()}
        # The saved package metadata says model_input_scaled=True, but the
        # Reviewer-3 notebook trains xgb_tuned on X_train_atk[top_features].
        # Using raw model_features exactly reproduces the saved test predictions.
        self.multiclass_model_input_scaled = False

    def _load_subclass(self) -> None:
        # Source of truth: SubClass_Classification_Experiments.ipynb changes CWD to
        # project/output and saves consolidated subclass artifacts under output/models.
        sub_models_path = self.project_dir / "output" / "models" / "all_sub_models.pkl"
        sub_encoders_path = self.project_dir / "output" / "models" / "all_label_encoders.pkl"
        unified_path = self.project_dir / "Inference" / "models" / "03_subclass_models" / "unified_inference_pipeline.pkl"

        self.sub_models_path = sub_models_path
        self.sub_encoders_path = sub_encoders_path
        with sub_models_path.open("rb") as f:
            self.sub_models = pickle.load(f)
        with sub_encoders_path.open("rb") as f:
            self.sub_encoders = pickle.load(f)
        with unified_path.open("rb") as f:
            self.unified_subclass_pkg = pickle.load(f)

        self.sub_scaler = self.unified_subclass_pkg["scaler"]
        self.sub_expected_features = list(self.unified_subclass_pkg["expected_features"])
        namespace: dict = {}
        exec(self.unified_subclass_pkg["feature_engineering_code"], namespace)
        self.add_features = namespace["add_features"]
        self.add_features_web = namespace["add_features_web_v2"]

    def _force_cpu(self) -> None:
        models = [self.binary_model, self.multiclass_model, *self.sub_models.values()]
        for model in models:
            try:
                model.set_params(device="cpu", n_jobs=-1)
            except Exception:
                pass

    def _validate_contracts(self) -> None:
        assert getattr(self.binary_model, "n_features_in_", None) == 46
        assert self.binary_threshold == 0.454
        assert self.binary_features == self.base_features
        assert getattr(self.binary_scaler, "n_features_in_", None) == 46

        assert self.multiclass_pkg["feature_selection_used"] is True
        assert self.multiclass_feature_columns == list(self.multiclass_protocol["feature_columns"])
        assert self.multiclass_model_features == list(self.multiclass_protocol["model_features"])
        assert len(self.multiclass_feature_columns) == 40
        assert len(self.multiclass_model_features) == 30
        assert getattr(self.multiclass_model, "n_features_in_", None) == 30
        assert np.allclose(self.multiclass_thresholds, np.array([1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7]))

        expected_families = {"DDoS", "DoS", "Mirai", "Spoofing", "Recon", "Web"}
        assert set(self.sub_models.keys()) == expected_families
        assert set(self.sub_encoders.keys()) == expected_families
        assert set(self.unified_subclass_pkg["sub_models"].keys()) == expected_families
        assert "BruteForce" not in self.sub_models
        for family, model in self.sub_models.items():
            assert getattr(model, "n_features_in_", None) == 110, family

        dummy = pd.DataFrame(np.ones((2, len(self.sub_expected_features))), columns=self.sub_expected_features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PerformanceWarning)
            standard_cols = set(self.add_features(dummy).columns)
        for family, model in self.sub_models.items():
            # The Reviewer-3 subclass notebook saved output/models/all_sub_models.pkl.
            # Those models were trained on output/processed 40-feature frames with
            # add_features(X) directly; no subclass scaler or Web-v2 expansion was used.
            missing = set(model.feature_names_in_) - standard_cols
            assert not missing, (family, sorted(missing)[:10])

    @staticmethod
    def _apply_probability_multipliers(proba: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
        adjusted = proba * multipliers
        return np.argmax(adjusted, axis=1)

    @staticmethod
    def _scaler_transform(scaler, frame: pd.DataFrame) -> np.ndarray:
        if hasattr(scaler, "feature_names_in_"):
            return scaler.transform(frame)
        return scaler.transform(frame.values)

    def _predict_family(self, df_attack_rich: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        # Match Multi_Class_7Attack_reviewer3.ipynb cell 19:
        # X_test_tree = X_test_atk[top_features].reset_index(drop=True).
        # The RobustScaler artifact is retained for audit compatibility only.
        x_model = align_columns(df_attack_rich, self.multiclass_model_features).reset_index(drop=True)
        proba = self.multiclass_model.predict_proba(x_model)
        pred_attack_0_6 = self._apply_probability_multipliers(proba, self.multiclass_thresholds)
        family_labels = [self.attack_names[int(pred)] for pred in pred_attack_0_6]
        return pred_attack_0_6, proba, family_labels

    def _sub_features(self, df_family_rich: pd.DataFrame, family: str) -> pd.DataFrame:
        base = align_columns(df_family_rich, self.sub_expected_features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PerformanceWarning)
            engineered = self.add_features(base)
        expected = list(self.sub_models[family].feature_names_in_)
        return align_columns(engineered, expected)

    def predict_raw(self, df_raw_46: pd.DataFrame, row_id=None) -> tuple[pd.DataFrame, dict]:
        validate_columns(df_raw_46, self.base_features, "df_raw_46")
        raw = df_raw_46[self.base_features].copy().reset_index(drop=True)
        n_rows = len(raw)
        if row_id is None:
            row_id = np.arange(n_rows, dtype=np.int64)
        row_id = np.asarray(row_id)
        if len(row_id) != n_rows:
            raise ValueError("row_id length does not match df_raw_46 length.")

        timings: list[dict] = []
        t0_total = time.perf_counter()

        t0 = time.perf_counter()
        x_binary = self._scaler_transform(self.binary_scaler, raw[self.binary_features])
        binary_prob = self.binary_model.predict_proba(x_binary)[:, 1]
        binary_pred = (binary_prob >= self.binary_threshold).astype(np.int64)
        timings.append(
            {
                "stage": "binary",
                "rows": int(n_rows),
                "seconds": float(time.perf_counter() - t0),
            }
        )

        benign_code = int(self.label_encoder_34.transform(["BenignTraffic"])[0])
        final_names = np.array(["BenignTraffic"] * n_rows, dtype=object)
        final_codes = np.full(n_rows, benign_code, dtype=np.int64)
        family_pred = np.array(["Benign"] * n_rows, dtype=object)
        family_proba_max = np.zeros(n_rows, dtype=float)
        subtype_pred = np.array(["BenignTraffic"] * n_rows, dtype=object)
        confidence = 1.0 - binary_prob
        route_status = np.array(["binary_benign"] * n_rows, dtype=object)

        attack_mask = binary_pred == 1
        attack_indices = np.where(attack_mask)[0]
        if len(attack_indices) > 0:
            df_attack_raw = raw.iloc[attack_indices].reset_index(drop=True)
            df_attack_rich = compute_derived_features(df_attack_raw)

            t0 = time.perf_counter()
            _, family_proba, family_labels = self._predict_family(df_attack_rich)
            family_labels_arr = np.asarray(family_labels, dtype=object)
            timings.append(
                {
                    "stage": "multiclass_family",
                    "rows": int(len(attack_indices)),
                    "seconds": float(time.perf_counter() - t0),
                }
            )

            family_pred[attack_indices] = family_labels_arr
            family_proba_max[attack_indices] = family_proba.max(axis=1)
            route_status[attack_indices] = "family_predicted"
            confidence[attack_indices] = family_proba.max(axis=1)

            brute_local = np.where(family_labels_arr == "BruteForce")[0]
            if len(brute_local) > 0:
                global_idx = attack_indices[brute_local]
                subtype_pred[global_idx] = "DictionaryBruteForce"
                final_names[global_idx] = "DictionaryBruteForce"
                final_codes[global_idx] = self.label_encoder_34.transform(["DictionaryBruteForce"])[0]
                route_status[global_idx] = "bruteforce_single_subtype"

            for family, model in self.sub_models.items():
                local_idx = np.where(family_labels_arr == family)[0]
                if len(local_idx) == 0:
                    continue
                t0 = time.perf_counter()
                x_sub = self._sub_features(df_attack_rich.iloc[local_idx].reset_index(drop=True), family)
                y_sub = model.predict(x_sub)
                sub_proba = model.predict_proba(x_sub).max(axis=1)
                labels = self.sub_encoders[family].inverse_transform(y_sub)
                global_idx = attack_indices[local_idx]
                subtype_pred[global_idx] = labels
                final_names[global_idx] = labels
                final_codes[global_idx] = self.label_encoder_34.transform(labels)
                confidence[global_idx] = sub_proba
                route_status[global_idx] = "subtype_predicted"
                timings.append(
                    {
                        "stage": f"subtype_{family}",
                        "rows": int(len(local_idx)),
                        "seconds": float(time.perf_counter() - t0),
                    }
                )

        timings.append(
            {
                "stage": "total_end_to_end",
                "rows": int(n_rows),
                "seconds": float(time.perf_counter() - t0_total),
            }
        )
        out = pd.DataFrame(
            {
                "row_id": row_id,
                "binary_prob": binary_prob,
                "binary_pred": binary_pred,
                "family_pred": family_pred,
                "family_proba_max": family_proba_max,
                "subtype_pred": subtype_pred,
                "y_pred_34class_name": final_names,
                "y_pred_34class": final_codes,
                "route_status": route_status,
                "confidence": confidence,
            }
        )
        return out, {"timings": timings, "rows": int(n_rows)}
