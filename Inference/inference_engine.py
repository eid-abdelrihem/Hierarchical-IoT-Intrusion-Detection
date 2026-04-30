"""
inference_engine.py
===================
The core inference engine for the IoT IDS.

Pipeline:
  Raw CSV → Binary → [if Attack] → 8-Class → Sub-Class → Final Result

Usage:
    from inference_engine import IDSInferenceEngine

    engine = IDSInferenceEngine()
    results = engine.predict(df)  # df is a pandas DataFrame
"""

import pickle
import warnings
import threading
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Global lock — serializes GPU calls across Streamlit's concurrent threads.
# Without this, multiple browser tabs cause simultaneous CUDA access → segfault.
_GPU_LOCK = threading.Lock()

# ── Model Paths ───────────────────────────────────────────────────────────────
import os
_BASE = os.path.dirname(os.path.abspath(__file__))
BINARY_PKG      = os.path.join(_BASE, "models", "01_binary",       "binary_inference_package.pkl")
MULTICLASS_PKG  = os.path.join(_BASE, "models", "02_multiclass_8", "MultiClass_7Attack_Inference_Package.pkl")
SUBCLASS_PKG    = os.path.join(_BASE, "models", "03_subclass_models", "unified_inference_pipeline.pkl")


class IDSInferenceEngine:
    """
    Hierarchical IoT Intrusion Detection System.

    Stage 1  – Binary:     Benign vs Attack
    Stage 2  – 8-Class:    Which attack family?  (DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce)
    Stage 3  – Sub-Class:  Exact attack subtype  (e.g. DDoS-SYN_Flood)
    """

    def __init__(self):
        print("Loading models…")
        self._load_binary()
        self._load_multiclass()
        self._load_subclass()
        print("✅ All models loaded and ready.\n")

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_binary(self):
        with open(BINARY_PKG, 'rb') as f:
            pkg = pickle.load(f)
        self.bin_model    = pkg['model']
        self.bin_scaler   = pkg['scaler']
        self.bin_features = pkg['feature_names']   # 46 features
        self.bin_labels   = pkg['label_map']       # {0: 'BenignTraffic', 1: 'Attack'}
        print(f"  Binary  : {pkg['model_name']} | {len(self.bin_features)} features")

    def _load_multiclass(self):
        with open(MULTICLASS_PKG, 'rb') as f:
            pkg = pickle.load(f)
        self.mc_model      = pkg['model']
        self.mc_scaler     = pkg['scaler']
        self.mc_scaler_features = list(pkg['scaler'].feature_names_in_)  # 40 features
        self.mc_features   = pkg['top_features']   # 30 features (subset of scaled 40)
        self.mc_thresholds = pkg['thresholds']
        self.mc_rev_map    = pkg['reverse_map']    # {0→1, 1→2 …}
        self.mc_names      = pkg['attack_names']   # ['BruteForce','DDoS',…]
        print(f"  8-Class : {pkg['model_name']} | scaler={len(self.mc_scaler_features)} features | model={len(self.mc_features)} features | {len(self.mc_names)} classes")

    def _load_subclass(self):
        with open(SUBCLASS_PKG, 'rb') as f:
            pkg = pickle.load(f)
        self.sub_scaler        = pkg['scaler']            # 40 features
        self.sub_stage1        = pkg['stage1_model']      # used internally (already done by mc)
        self.sub_models        = pkg['sub_models']        # dict: family → XGBClassifier
        self.sub_encoders      = pkg['sub_encoders']      # dict: family → LabelEncoder
        self.sub_cat_map       = pkg['category_map']      # dict: family → [subtypes]
        self.sub_feat_eng_code = pkg['feature_engineering_code']
        self.sub_expected      = pkg['expected_features'] # 40 base features
        self.sub_stage1_labels = pkg['stage1_labels_map'] # {0:'BenignTraffic', 1:'DDoS'…}

        # Compile the feature-engineering function from the stored code string
        _ns = {}
        exec(self.sub_feat_eng_code, _ns)
        self._add_features     = _ns['add_features']
        self._add_features_web = _ns['add_features_web_v2']

        # ✅ Force CPU mode — XGBoost CUDA is NOT thread-safe with Streamlit's
        # concurrent sessions, causing unrecoverable segfaults.
        # n_jobs=-1 uses ALL CPU cores for maximum speed.
        try:
            self.bin_model.set_params(device='cpu', n_jobs=-1)
            self.mc_model.set_params(device='cpu', n_jobs=-1)
            for m in self.sub_models.values():
                m.set_params(device='cpu', n_jobs=-1)
        except Exception:
            pass

        print(f"  SubClass: {len(self.sub_models)} families | base={len(self.sub_expected)} features")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _align(self, df, feature_list):
        """Select and order columns; fill missing ones with 0."""
        df = df.copy()
        missing = [c for c in feature_list if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = 0.0
        return df[feature_list].copy()

    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the 4 derived features that Stage-2 and Stage-3 scalers expect
        but are NOT present in the raw 46-feature binary input.
            bytes_per_packet     = Tot size / (Number + 1)
            rate_ratio           = Rate / (Drate + 1e-8)
            header_payload_ratio = Header_Length / (Tot size + 1e-8)
            flag_density         = (syn_flag_number + ack_flag_number + psh_flag_number
                                    + fin_count + urg_count) / (Tot sum + 1e-8)
        """
        df = df.copy()
        df['bytes_per_packet']     = df.get('Tot size', 0) / (df.get('Number', 0) + 1)
        df['rate_ratio']           = df.get('Rate', 0)          / (df.get('Drate', 0) + 1e-8)
        df['header_payload_ratio'] = df.get('Header_Length', 0) / (df.get('Tot size', 0) + 1e-8)
        flags = (df.get('syn_flag_number', 0) + df.get('ack_flag_number', 0)
                 + df.get('psh_flag_number', 0) + df.get('fin_count', 0)
                 + df.get('urg_count', 0))
        df['flag_density'] = flags / (df.get('Tot sum', 0) + 1e-8)
        return df

    def _apply_thresholds(self, proba, thresholds):
        """Apply per-class thresholds to probability matrix → class indices."""
        adjusted = proba * (1 / np.clip(thresholds, 1e-8, 1.0))
        return np.argmax(adjusted, axis=1)

    def _sub_features(self, df_raw, family):
        """Build the engineered feature set expected by a sub-class model."""
        sub_model = self.sub_models[family]
        expected_cols = list(sub_model.feature_names_in_)

        # Scale the 40 base features first
        base_df = self._align(df_raw, self.sub_expected)
        scaled_arr = self.sub_scaler.transform(base_df)
        scaled_df  = pd.DataFrame(scaled_arr, columns=self.sub_expected)

        # Apply feature engineering
        if family == 'Web':
            eng_df = self._add_features_web(scaled_df)
        else:
            eng_df = self._add_features(scaled_df)

        return self._align(eng_df, expected_cols)

    # ── Main Predict ──────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full hierarchical pipeline on a DataFrame of raw network features.

        Returns a DataFrame with the original data plus:
            is_attack       – bool
            stage1_label    – 'BenignTraffic' or attack family
            stage2_subtype  – exact attack subtype (or same as stage1 if no sub-model)
            confidence      – probability of the final prediction (%)
        """
        df = df.copy().reset_index(drop=True)
        n  = len(df)

        results = pd.DataFrame({
            'is_attack':    False,
            'stage1_label': 'BenignTraffic',
            'stage2_subtype': 'BenignTraffic',
            'confidence':   0.0
        }, index=range(n))

        # ── Stage 1: Binary ───────────────────────────────────────────────────
        print(f"[Stage 1] Binary classification on {n:,} samples…", end=' ')
        X_bin        = self._align(df, self.bin_features)
        X_bin_scaled = self.bin_scaler.transform(X_bin)
        bin_pred  = self.bin_model.predict(X_bin_scaled)
        bin_proba = self.bin_model.predict_proba(X_bin_scaled)[:, 1]

        atk_mask = bin_pred == 1
        results.loc[~atk_mask, 'is_attack']      = False
        results.loc[~atk_mask, 'stage1_label']   = 'BenignTraffic'
        results.loc[~atk_mask, 'stage2_subtype'] = 'BenignTraffic'
        results.loc[~atk_mask, 'confidence']     = (1 - bin_proba[~atk_mask]) * 100
        results.loc[atk_mask,  'is_attack']      = True
        results.loc[atk_mask,  'confidence']     = bin_proba[atk_mask] * 100

        n_atk = atk_mask.sum()
        print(f"✓  Attacks: {n_atk:,} / {n:,}")

        if n_atk == 0:
            return results

        df_atk = df[atk_mask].reset_index(drop=True)

        # Enrich with derived features for both Stage 2 and Stage 3
        df_atk_rich = self._compute_derived_features(df_atk)

        # ── Stage 2: 8-Class (Attack Family) ─────────────────────────────────
        print(f"[Stage 2] 8-Class on {n_atk:,} attack samples…", end=' ')
        X_mc_raw    = self._align(df_atk_rich, self.mc_scaler_features)
        X_mc_scaled = self.mc_scaler.transform(X_mc_raw)
        X_mc_scaled_df = pd.DataFrame(X_mc_scaled, columns=self.mc_scaler_features)
        X_mc = X_mc_scaled_df[self.mc_features].values
        mc_proba = self.mc_model.predict_proba(X_mc)
        mc_pred  = self._apply_thresholds(mc_proba, self.mc_thresholds)
        mc_labels = [self.mc_names[self.mc_rev_map.get(p, p)] if self.mc_rev_map.get(p, p) < len(self.mc_names) else 'Unknown' for p in mc_pred]
        results.loc[atk_mask, 'stage1_label'] = mc_labels
        print("✓")

        # ── Stage 3: Sub-Class (Exact Subtype) ───────────────────────────────
        print(f"[Stage 3] Sub-class classification per family…")
        for family in self.sub_models.keys():
            fam_mask_local = np.array(mc_labels) == family
            if fam_mask_local.sum() == 0:
                continue
            print(f"          {family}: {fam_mask_local.sum():,} samples…", end=' ')

            df_fam = df_atk_rich[fam_mask_local].reset_index(drop=True)
            X_sub  = self._sub_features(df_fam, family)
            sub_pred  = self.sub_models[family].predict(X_sub)
            sub_proba = self.sub_models[family].predict_proba(X_sub).max(axis=1)
            sub_labels = self.sub_encoders[family].inverse_transform(sub_pred)
            print(f"✓")

            # Map back to the original DataFrame index
            global_idx = np.where(atk_mask)[0][fam_mask_local]
            results.loc[global_idx, 'stage2_subtype'] = sub_labels
            results.loc[global_idx, 'confidence']     = sub_proba * 100
            print(f"✓")

        # BruteForce / families without a sub-model: keep family name as subtype
        atk_global_idx = np.where(atk_mask)[0]
        for i, lbl in enumerate(mc_labels):
            if lbl not in self.sub_models:
                g = atk_global_idx[i]
                if results.loc[g, 'stage2_subtype'] == 'BenignTraffic':
                    results.loc[g, 'stage2_subtype'] = lbl

        return results


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    sample = os.path.join(_BASE, 'data_samples', 'sample_traffic.csv')
    if os.path.exists(sample):
        df = pd.read_csv(sample).drop(columns=['true_binary_label', 'true_8class_label'], errors='ignore')
        engine = IDSInferenceEngine()
        out = engine.predict(df)
        print("\n=== Sample Results ===")
        print(out.head(10).to_string())
        print(f"\nSummary:\n{out['stage1_label'].value_counts()}")
    else:
        print("No sample data found. Create one in data_samples/")
