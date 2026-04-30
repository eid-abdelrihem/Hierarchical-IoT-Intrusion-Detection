"""
run_inference.py
================
CLI runner for the IoT IDS Inference Engine.

Usage:
    python run_inference.py --input data_samples/sample_traffic.csv
    python run_inference.py --input data_samples/sample_traffic.csv --output results/output.csv
"""

import argparse
import os
import sys
import time
import pandas as pd

# Add Inference folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_engine import IDSInferenceEngine


def main():
    parser = argparse.ArgumentParser(description='IoT IDS Inference Engine')
    parser.add_argument('--input',  required=True,   help='Path to input CSV file (raw network features)')
    parser.add_argument('--output', default=None,    help='Path to save results CSV (optional)')
    parser.add_argument('--show',   type=int, default=20, help='Number of rows to display (default: 20)')
    args = parser.parse_args()

    # ── Load Data ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  IoT IDS Inference System")
    print(f"{'='*60}")
    print(f"  Input : {args.input}")

    df_raw = pd.read_csv(args.input)
    # Drop label columns if present (they're only there during testing)
    label_cols = [c for c in df_raw.columns if 'label' in c.lower() or 'class' in c.lower()]
    y_true = df_raw[label_cols].copy() if label_cols else None
    df = df_raw.drop(columns=label_cols, errors='ignore')
    print(f"  Samples: {len(df):,} | Features: {len(df.columns)}")
    print(f"{'='*60}\n")

    # ── Run Inference ─────────────────────────────────────────────────────────
    engine = IDSInferenceEngine()
    t0 = time.time()
    results = engine.predict(df)
    elapsed = time.time() - t0

    # ── Merge results with original data ──────────────────────────────────────
    final_df = pd.concat([df, results], axis=1)
    if y_true is not None:
        final_df = pd.concat([final_df, y_true], axis=1)

    # ── Summary Report ────────────────────────────────────────────────────────
    n = len(results)
    n_atk = results['is_attack'].sum()
    n_ben = n - n_atk

    print(f"\n{'='*60}")
    print(f"  INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"  Total samples   : {n:,}")
    print(f"  Benign          : {n_ben:,}  ({n_ben/n*100:.1f}%)")
    print(f"  Attacks         : {n_atk:,}  ({n_atk/n*100:.1f}%)")
    print(f"  Latency         : {elapsed/n*1000:.4f} ms/sample")
    print(f"  Total time      : {elapsed:.2f}s")

    if n_atk > 0:
        print(f"\n  Attack Families Detected:")
        fam_counts = results[results['is_attack']]['stage1_label'].value_counts()
        for fam, cnt in fam_counts.items():
            print(f"    {fam:<20}: {cnt:,}")

        print(f"\n  Top Attack Subtypes:")
        sub_counts = results[results['is_attack']]['stage2_subtype'].value_counts().head(10)
        for sub, cnt in sub_counts.items():
            print(f"    {sub:<35}: {cnt:,}")

    # ── Compare with ground truth if available ────────────────────────────────
    if y_true is not None and 'true_binary_label' in y_true.columns:
        from sklearn.metrics import accuracy_score, recall_score, f1_score
        y_true_bin = y_true['true_binary_label'].values
        y_pred_bin = results['is_attack'].astype(int).values
        print(f"\n  Validation vs Ground Truth:")
        print(f"    Accuracy : {accuracy_score(y_true_bin, y_pred_bin)*100:.2f}%")
        print(f"    Recall   : {recall_score(y_true_bin, y_pred_bin)*100:.2f}%")
        print(f"    F1-Score : {f1_score(y_true_bin, y_pred_bin)*100:.2f}%")

    print(f"{'='*60}")

    # ── Sample Preview ────────────────────────────────────────────────────────
    print(f"\nSample Results (first {args.show} rows):")
    display_cols = ['is_attack', 'stage1_label', 'stage2_subtype', 'confidence']
    if y_true is not None:
        display_cols += [c for c in y_true.columns if c in results.columns]
    display_cols = [c for c in display_cols if c in results.columns]
    print(results[display_cols].head(args.show).to_string(index=True))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = args.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', f'{base}_results.csv')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    final_df.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")


if __name__ == '__main__':
    main()
