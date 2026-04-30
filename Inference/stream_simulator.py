"""
stream_simulator.py
===================
Simulates a live IoT network traffic stream by reading rows from a CSV
and writing them to a shared JSON file one batch at a time.

Realistic simulation features:
  - 80% Benign / 20% Attack traffic ratio (mirrors real network distributions)
  - Calibrated feature drift noise (2% of each feature's real std dev)
    to simulate real sensor variance without corrupting the signal.

Run in a SEPARATE terminal:
    python Inference/stream_simulator.py

Then run the dashboard in another terminal:
    streamlit run Inference/dashboard.py
"""

import os, time, json, pickle
import pandas as pd
import numpy as np

BASE        = os.path.dirname(os.path.abspath(__file__))
SOURCE_CSV  = os.path.join(BASE, 'data_samples', 'live_stream.json').replace('live_stream.json', '') + 'sample_traffic.csv'
SOURCE_CSV  = os.path.join(BASE, 'data_samples', 'sample_traffic.csv')
STREAM_FILE = os.path.join(BASE, 'data_samples', 'live_stream.json')
BINARY_PKG  = os.path.join(BASE, 'models', '01_binary', 'binary_inference_package.pkl')

BATCH_SIZE  = 10     # rows per tick (bigger batch = smoother metrics)
TICK_SEC    = 1.5    # seconds between ticks

# Realistic traffic ratio: 80% benign, 20% attack
BENIGN_RATIO = 0.80

# Noise level: 2% of each feature's real standard deviation (realistic sensor drift)
NOISE_LEVEL  = 0       # Disabled: this security model is sensitive; even tiny noise creates false positives


def main():
    print(f"Loading source data from {SOURCE_CSV}...")
    df = pd.read_csv(SOURCE_CSV)

    # Keep label columns for ground truth validation on the dashboard
    label_cols = [c for c in df.columns if 'label' in c.lower() or 'true' in c.lower()]
    print(f"  Found label columns: {label_cols}")

    # Split into benign and attack pools
    benign_df = df[df['true_binary_label'] == 0].copy()
    attack_df = df[df['true_binary_label'] == 1].copy()
    print(f"  Benign pool: {len(benign_df)} rows | Attack pool: {len(attack_df)} rows")
    print(f"  Traffic ratio → Benign: {BENIGN_RATIO*100:.0f}% / Attack: {(1-BENIGN_RATIO)*100:.0f}%")

    # Load feature std devs from the binary model scaler for calibrated noise
    try:
        with open(BINARY_PKG, 'rb') as f:
            pkg = pickle.load(f)
        feature_names = pkg['feature_names']
        feature_stds  = pkg['scaler'].scale_   # shape: (46,)
        print(f"  Loaded scaler: {len(feature_names)} features. Noise = {NOISE_LEVEL*100:.0f}% × std")
    except Exception as e:
        feature_names = None
        feature_stds  = None
        print(f"  Could not load scaler ({e}), using fixed noise instead.")

    n  = len(df)
    print(f"\n  Batch size: {BATCH_SIZE} rows/tick | Interval: {TICK_SEC}s")
    print(f"  Stream file → {STREAM_FILE}")
    print(f"  [Ctrl+C to stop]\n")

    # Initialize the stream state file
    state = {'rows': [], 'tick': 0, 'total_sent': 0}
    with open(STREAM_FILE, 'w') as f:
        json.dump(state, f)

    tick = 0
    benign_idx = 0
    attack_idx = 0

    while True:
        # Determine how many benign vs attack rows to pick this tick
        n_benign = round(BATCH_SIZE * BENIGN_RATIO)
        n_attack = BATCH_SIZE - n_benign

        # Pick rows from each pool (looping around)
        b_idx = [benign_idx % len(benign_df) + i for i in range(n_benign)]
        a_idx = [attack_idx % len(attack_df) + i for i in range(n_attack)]

        b_batch = benign_df.iloc[[i % len(benign_df) for i in b_idx]].copy()
        a_batch = attack_df.iloc[[i % len(attack_df) for i in a_idx]].copy()

        batch = pd.concat([b_batch, a_batch], ignore_index=True)
        # Shuffle so benign/attack aren't always grouped
        batch = batch.sample(frac=1, random_state=tick).reset_index(drop=True)

        # Apply calibrated feature drift noise (2% of each feature's real std)
        if feature_names is not None and feature_stds is not None:
            for i, feat in enumerate(feature_names):
                if feat in batch.columns:
                    drift = np.random.normal(0, NOISE_LEVEL * feature_stds[i], size=len(batch))
                    batch[feat] = batch[feat].values + drift
        else:
            # Fallback: tiny proportional noise
            num_cols = [c for c in batch.select_dtypes(include=[np.number]).columns if c not in label_cols]
            for col in num_cols:
                noise = np.random.normal(0, 0.002 * (batch[col].std() + 1e-8), size=len(batch))
                batch[col] = batch[col].values + noise

        state = {
            'rows':       batch.to_dict(orient='records'),
            'tick':       tick,
            'total_sent': (benign_idx + n_benign) + (attack_idx + n_attack),
            'timestamp':  time.strftime('%H:%M:%S'),
            'n_benign_sent': n_benign,
            'n_attack_sent': n_attack,
        }
        with open(STREAM_FILE, 'w') as f:
            json.dump(state, f)

        print(
            f"[Tick {tick:4d}] Sent {n_benign} Benign + {n_attack} Attack "
            f"(noise={NOISE_LEVEL*100:.0f}% std) @ {state['timestamp']}",
            flush=True
        )

        benign_idx = (benign_idx + n_benign) % len(benign_df)
        attack_idx = (attack_idx + n_attack) % len(attack_df)
        tick += 1
        time.sleep(TICK_SEC)


if __name__ == '__main__':
    main()
