from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


RANDOM_STATE = 42


def ablation_dir() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return ablation_dir().parent


def workspace_root() -> Path:
    candidates = [
        Path("D:/Ml Project"),
        project_root(),
        *project_root().parents,
    ]
    for candidate in candidates:
        if (candidate / "processed_data" / "feature_columns.joblib").exists() and (
            candidate / "data"
        ).exists():
            return candidate
    raise FileNotFoundError("Could not locate workspace root with data/ and processed_data/.")


def results_root() -> Path:
    path = project_root() / "results" / "ablation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_base_feature_names() -> list[str]:
    return list(joblib.load(workspace_root() / "processed_data" / "feature_columns.joblib"))


def load_label_encoder_34():
    return joblib.load(workspace_root() / "processed_data" / "label_encoder_34.joblib")


def load_label_encoder_8():
    return joblib.load(workspace_root() / "processed_data" / "label_encoder_8.joblib")


def validate_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing[:20]}")


def align_columns(df: pd.DataFrame, columns: Iterable[str], fill_value: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = fill_value
    return out[list(columns)].copy()


def label_to_family(label: str) -> str:
    if label == "BenignTraffic":
        return "Benign"
    if label == "DictionaryBruteForce":
        return "BruteForce"
    if label.startswith("DDoS-"):
        return "DDoS"
    if label.startswith("DoS-"):
        return "DoS"
    if label.startswith("Mirai-"):
        return "Mirai"
    if label in {"DNS_Spoofing", "MITM-ArpSpoofing"}:
        return "Spoofing"
    if label.startswith("Recon-") or label == "VulnerabilityScan":
        return "Recon"
    if label in {
        "Backdoor_Malware",
        "BrowserHijacking",
        "CommandInjection",
        "SqlInjection",
        "Uploading_Attack",
        "XSS",
    }:
        return "Web"
    raise ValueError(f"Unknown label family for label={label!r}")


def add_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column.")
    out = df.copy()
    le34 = load_label_encoder_34()
    le8 = load_label_encoder_8()
    labels = out["label"].astype(str)
    unknown = sorted(set(labels) - set(le34.classes_))
    if unknown:
        raise ValueError(f"Labels not present in label_encoder_34: {unknown}")
    out["label_binary"] = (labels != "BenignTraffic").astype(np.int64)
    out["label_8"] = labels.map(label_to_family)
    out["label_34_encoded"] = le34.transform(labels)
    out["label_8_encoded"] = le8.transform(out["label_8"])
    return out


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Match the EDA_Feature_Engineering notebook exactly; the 40-feature
    # multiclass and subclass artifacts were trained on these definitions.
    out["bytes_per_packet"] = out.get("Tot sum", 0) / (out.get("Number", 0) + 1)
    out["rate_ratio"] = out.get("Drate", 0) / (out.get("Srate", 0) + 1e-10)
    out["header_payload_ratio"] = out.get("Header_Length", 0) / (out.get("Tot sum", 0) + 1)
    flags = out.get("syn_count", 0) + out.get("ack_count", 0) + out.get("urg_count", 0)
    out["flag_density"] = flags / (out.get("Number", 0) + 1)
    return out


def stable_row_hashes(
    df: pd.DataFrame,
    columns: list[str],
    decimals: int = 10,
    sep: str = "\x1f",
) -> np.ndarray:
    validate_columns(df, columns, "row-hash dataframe")
    normalized = pd.DataFrame(index=df.index)
    for column in columns:
        series = df[column]
        if column == "label":
            normalized[column] = series.astype(str)
            continue
        numeric = pd.to_numeric(series, errors="coerce").astype("float64").round(decimals)
        normalized[column] = numeric.map(
            lambda value: "" if pd.isna(value) else f"{float(value):.{decimals}g}"
        )
    joined = normalized[columns].agg(sep.join, axis=1)
    return joined.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()).to_numpy()


def evaluate_34class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    label_names: list[str],
    elapsed_seconds: float,
    model_name: str,
) -> tuple[dict, pd.DataFrame]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(label_names)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = pd.DataFrame(report).T
    per_class.index.name = "label"
    macro_recall = float(
        recall_score(
            y_true,
            y_pred,
            labels=np.arange(len(label_names)),
            average="macro",
            zero_division=0,
        )
    )
    metrics = {
        "model": model_name,
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": macro_recall,
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": macro_recall,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "elapsed_seconds": float(elapsed_seconds),
        "us_per_flow": float(elapsed_seconds / max(len(y_true), 1) * 1_000_000),
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if y_proba.ndim == 2 and len(y_proba) == len(y_true):
            metrics["mean_top1_confidence"] = float(np.max(y_proba, axis=1).mean())
    return metrics, per_class.reset_index()


def save_npz_compressed(path: Path, **arrays) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)
