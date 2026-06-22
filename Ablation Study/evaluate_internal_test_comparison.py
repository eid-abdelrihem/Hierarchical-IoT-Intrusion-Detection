from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from ablation_common import ensure_dir, evaluate_34class, load_base_feature_names, load_label_encoder_34, write_json
from ablation_hierarchical_engine import AblationReviewer3InferenceEngine


WORKSPACE_ROOT = Path("D:/Ml Project")
PROJECT_ROOT = THIS_DIR.parent
PROCESSED_DIR = WORKSPACE_ROOT / "processed_data"
FLAT_DIR = PROJECT_ROOT / "results" / "ablation" / "flat"
OUTPUT_DIR = ensure_dir(PROJECT_ROOT / "results" / "ablation" / "internal_test")

TEST_NPZ_PATH = PROCESSED_DIR / "test_data.npz"
SCALER_PATH = PROCESSED_DIR / "scaler.joblib"
BINARY_ARTIFACT_PATH = PROJECT_ROOT / "results" / "reviewer3_binary" / "binary_stage_test_predictions_threshold_0454.npz"
FLAT_PRED_PATH = FLAT_DIR / "flat_34class_test_predictions.npz"
FLAT_METRICS_PATH = FLAT_DIR / "flat_34class_test_metrics.json"

HIER_PRED_PATH = OUTPUT_DIR / "hierarchical_internal_test_predictions.npz"
SUMMARY_PATH = OUTPUT_DIR / "internal_test_flat_vs_hierarchical_summary.csv"
PER_CLASS_PATH = OUTPUT_DIR / "internal_test_per_class_comparison.csv"
BINARY_GATE_PATH = OUTPUT_DIR / "internal_test_binary_gate_summary.csv"
MANIFEST_PATH = OUTPUT_DIR / "internal_test_comparison_manifest.json"

CHUNK_SIZE = 100_000


def require_paths() -> None:
    missing = [
        path
        for path in [TEST_NPZ_PATH, SCALER_PATH, BINARY_ARTIFACT_PATH, FLAT_PRED_PATH, FLAT_METRICS_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required internal-test inputs:\n" + "\n".join(map(str, missing)))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str], prefix: str) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(label_names)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    frame = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    frame = frame[frame["label"].isin(label_names)].copy()
    frame = frame.rename(
        columns={
            "precision": f"{prefix}_precision",
            "recall": f"{prefix}_recall",
            "f1-score": f"{prefix}_f1",
            "support": f"{prefix}_support",
        }
    )
    return frame


def binary_summary(y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return pd.DataFrame(
        [
            {
                "threshold": 0.454,
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "accuracy": accuracy_score(y_true_binary, y_pred_binary),
                "macro_f1": f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0),
                "attack_precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
                "attack_recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
                "attack_miss_rate": fn / max(fn + tp, 1),
                "false_alarm_rate": fp / max(fp + tn, 1),
                "mcc": matthews_corrcoef(y_true_binary, y_pred_binary),
            }
        ]
    )


def main() -> None:
    require_paths()

    label_encoder_34 = load_label_encoder_34()
    label_names = list(label_encoder_34.classes_)
    feature_names = load_base_feature_names()
    scaler = joblib.load(SCALER_PATH)
    engine = AblationReviewer3InferenceEngine(PROJECT_ROOT, WORKSPACE_ROOT)
    flat_metrics = load_json(FLAT_METRICS_PATH)

    test_npz = np.load(TEST_NPZ_PATH)
    x_test_scaled = test_npz["X"]
    y_true = test_npz["y_34class"].astype(np.int64)
    y_true_binary = test_npz["y_binary"].astype(np.int64)
    n_rows = len(y_true)

    flat_npz = np.load(FLAT_PRED_PATH, allow_pickle=True)
    flat_y_true = flat_npz["y_true_34class"].astype(np.int64)
    flat_y_pred = flat_npz["y_pred_34class"].astype(np.int64)
    flat_row_id = flat_npz["test_row_id"].astype(np.int64)

    assert x_test_scaled.shape == (n_rows, len(feature_names))
    assert np.array_equal(flat_y_true, y_true)
    assert np.array_equal(flat_row_id, np.arange(n_rows, dtype=np.int64))

    binary_npz = np.load(BINARY_ARTIFACT_PATH)
    binary_saved_pred = binary_npz["y_pred_binary"].astype(np.int64)
    binary_saved_prob = binary_npz["y_prob_binary"].astype(np.float64)
    assert np.array_equal(binary_npz["y_true_binary"].astype(np.int64), y_true_binary)

    hier_y_pred = np.empty(n_rows, dtype=np.int64)
    hier_binary_pred = np.empty(n_rows, dtype=np.int8)
    hier_binary_prob = np.empty(n_rows, dtype=np.float32)
    hier_confidence = np.empty(n_rows, dtype=np.float32)
    hier_family = np.empty(n_rows, dtype=object)
    hier_route = np.empty(n_rows, dtype=object)

    start = time.perf_counter()
    for start_idx in range(0, n_rows, CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, n_rows)
        x_scaled_chunk = x_test_scaled[start_idx:end_idx]
        raw_chunk = scaler.inverse_transform(x_scaled_chunk)
        raw_df = pd.DataFrame(raw_chunk, columns=feature_names)
        pred_df, _ = engine.predict_raw(raw_df, row_id=np.arange(start_idx, end_idx, dtype=np.int64))

        hier_y_pred[start_idx:end_idx] = pred_df["y_pred_34class"].to_numpy(np.int64)
        hier_binary_pred[start_idx:end_idx] = pred_df["binary_pred"].to_numpy(np.int8)
        hier_binary_prob[start_idx:end_idx] = pred_df["binary_prob"].to_numpy(np.float32)
        hier_confidence[start_idx:end_idx] = pred_df["confidence"].to_numpy(np.float32)
        hier_family[start_idx:end_idx] = pred_df["family_pred"].astype(str).to_numpy(object)
        hier_route[start_idx:end_idx] = pred_df["route_status"].astype(str).to_numpy(object)

        print(f"[internal] rows={end_idx:,}/{n_rows:,} ({end_idx / n_rows:.1%})", flush=True)

    hier_seconds = time.perf_counter() - start

    binary_mismatch = int(np.count_nonzero(hier_binary_pred.astype(np.int64) != binary_saved_pred))
    binary_prob_max_abs_diff = float(np.max(np.abs(hier_binary_prob.astype(np.float64) - binary_saved_prob)))

    flat_summary = {
        "model": "Flat 34-class XGBoost",
        "rows": int(n_rows),
        "accuracy": float(flat_metrics["accuracy"]),
        "balanced_accuracy": float(flat_metrics["balanced_accuracy"]),
        "macro_precision": float(flat_metrics["macro_precision"]),
        "macro_recall": float(flat_metrics["macro_recall"]),
        "macro_f1": float(flat_metrics["macro_f1"]),
        "weighted_f1": float(flat_metrics["weighted_f1"]),
        "mcc": float(flat_metrics["mcc"]),
        "elapsed_seconds": float(flat_metrics["elapsed_seconds"]),
        "us_per_flow": float(flat_metrics["us_per_flow"]),
    }
    hier_summary, _ = evaluate_34class(
        y_true,
        hier_y_pred,
        None,
        label_names,
        hier_seconds,
        "Hierarchical IDS",
    )

    summary_df = pd.DataFrame([flat_summary, hier_summary])
    metric_cols = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "mcc"]
    for col in metric_cols:
        summary_df[col] = summary_df[col].astype(float)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    flat_report = per_class_report(y_true, flat_y_pred, label_names, "flat")
    hier_report = per_class_report(y_true, hier_y_pred, label_names, "hier")
    per_class = flat_report.merge(hier_report, on="label", how="inner")
    per_class["delta_f1"] = per_class["hier_f1"] - per_class["flat_f1"]
    per_class["winner"] = np.select(
        [per_class["delta_f1"] > 1e-12, per_class["delta_f1"] < -1e-12],
        ["hierarchical", "flat"],
        default="tie",
    )
    per_class.to_csv(PER_CLASS_PATH, index=False)

    binary_df = binary_summary(y_true_binary, hier_binary_pred.astype(np.int64))
    binary_df["saved_binary_prediction_mismatches"] = binary_mismatch
    binary_df["saved_binary_prob_max_abs_diff"] = binary_prob_max_abs_diff
    binary_df.to_csv(BINARY_GATE_PATH, index=False)

    np.savez_compressed(
        HIER_PRED_PATH,
        y_true_34class=y_true,
        y_pred_34class=hier_y_pred,
        y_true_binary=y_true_binary,
        binary_pred=hier_binary_pred,
        binary_prob=hier_binary_prob,
        confidence=hier_confidence,
        family_pred=hier_family,
        route_status=hier_route,
        test_row_id=np.arange(n_rows, dtype=np.int64),
        source_protocol=np.array(["processed_data/test_data.npz + scaler.inverse_transform"], dtype=object),
    )

    manifest = {
        "comparison_name": "internal_test_flat_vs_hierarchical",
        "test_npz": str(TEST_NPZ_PATH),
        "raw_reconstruction": "RobustScaler.inverse_transform(test_data.npz['X'])",
        "inverse_transform_validation": "scaler.transform(scaler.inverse_transform(X)) reproduces X to floating precision",
        "flat_predictions": str(FLAT_PRED_PATH),
        "hierarchical_predictions": str(HIER_PRED_PATH),
        "binary_artifact": str(BINARY_ARTIFACT_PATH),
        "binary_prediction_mismatches_vs_saved_artifact": binary_mismatch,
        "binary_prob_max_abs_diff_vs_saved_artifact": binary_prob_max_abs_diff,
        "chunk_size": CHUNK_SIZE,
        "rows": int(n_rows),
    }
    write_json(MANIFEST_PATH, manifest)

    print("\nInternal test comparison complete.")
    print(summary_df.to_string(index=False))
    print("\nBinary gate:")
    print(binary_df.to_string(index=False))
    print("\nTop hierarchical gains:")
    print(per_class.sort_values("delta_f1", ascending=False).head(10).to_string(index=False))
    print("\nTop flat gains:")
    print(per_class.sort_values("delta_f1", ascending=True).head(10).to_string(index=False))
    print(f"\nOutput dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
