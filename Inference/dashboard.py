"""
dashboard.py
============
Streamlit Live IoT IDS Dashboard

Modes:
  1. Live Stream  — reads from stream_simulator output (live_stream.json) and runs inference in real-time
  2. File Upload  — upload any CSV and run inference immediately

Run:
    streamlit run Inference/dashboard.py
"""

import os, sys, json, time

# ✅ CUDA_LAUNCH_BLOCKING=1 forces all GPU kernel launches to be synchronous.
# This prevents the silent CUDA segfault caused by multiple Streamlit sessions
# (browser tabs) running GPU inference concurrently on the same CUDA context.
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import streamlit as st
import pandas as pd
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
.attack-card  { background:#ff4b4b22; border-left:4px solid #ff4b4b; padding:10px; border-radius:6px; margin:5px 0; }
.benign-card  { background:#21c35422; border-left:4px solid #21c354; padding:10px; border-radius:6px; margin:5px 0; }
.stAlert      { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

BASE        = os.path.dirname(os.path.abspath(__file__))
STREAM_FILE = os.path.join(BASE, 'data_samples', 'live_stream.json')
ATTACK_COLORS = {
    'DoS':      '#FF6B6B',
    'DDoS':     '#FF4500',
    'Mirai':    '#FFA500',
    'Recon':    '#FFD700',
    'Spoofing': '#9B59B6',
    'Web':      '#3498DB',
    'BruteForce':'#E74C3C',
    'BenignTraffic': '#2ECC71',
}

# ── Load Engine (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models… (one-time)")
def load_engine():
    from inference_engine import IDSInferenceEngine
    return IDSInferenceEngine()

# ── Session State Init ────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history     = []   # list of result dicts
if 'last_tick' not in st.session_state:
    st.session_state.last_tick   = -1
if 'total_processed' not in st.session_state:
    st.session_state.total_processed = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/firewall.png", width=80)
    st.title("IoT IDS Control")
    st.divider()

    mode = st.radio("Mode", ["🔴 Live Stream", "📂 File Upload", "🔍 Explainability (SHAP)"], index=0)
    st.divider()

    if mode == "🔴 Live Stream":
        refresh_sec = st.sidebar.slider("Refresh interval (s)", 1, 5, 2)
        max_history = st.sidebar.slider("Max history rows", 50, 500, 200)
        st.sidebar.info("Start the stream simulator in a terminal:\n```\npython Inference/stream_simulator.py\n```")
        auto_refresh = st.sidebar.toggle("Auto-refresh", value=True)
    elif mode == "📂 File Upload":
        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        run_btn  = st.button("▶ Run Inference", type="primary", use_container_width=True)
    else:
        st.sidebar.info("SHAP Explainability — understand WHY the model flags each sample as attack or benign.")
        shap_n_samples = st.sidebar.slider("Samples for SHAP analysis", 20, 200, 50, step=10)
    st.divider()
    st.caption("IoT IDS — Ahmed Sherif")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ IoT Intrusion Detection System")
st.caption("Real-time hierarchical attack detection: Binary → Attack Family → Exact Subtype")

# ── Helper: run inference on a df ────────────────────────────────────────────
def infer(engine, df_raw):
    df_clean = df_raw.drop(
        columns=[c for c in df_raw.columns if 'label' in c.lower() or 'true' in c.lower()],
        errors='ignore'
    )
    results = engine.predict(df_clean)
    return results

# ── Helper: build summary from history ───────────────────────────────────────
def build_summary(history_df):
    n          = len(history_df)
    n_attack   = int(history_df['is_attack'].sum())
    n_benign   = n - n_attack
    attack_rate = n_attack / n * 100 if n > 0 else 0
    fam_counts  = history_df[history_df['is_attack']]['family'].value_counts() if n_attack > 0 else pd.Series(dtype=int)
    sub_counts  = history_df[history_df['is_attack']]['subtype'].value_counts().head(10) if n_attack > 0 else pd.Series(dtype=int)
    return n, n_attack, n_benign, attack_rate, fam_counts, sub_counts

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: LIVE STREAM
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🔴 Live Stream":
    engine = load_engine()

    # ── Create PERSISTENT containers in the main page scope ──────────────────
    # These are created ONCE. The fragment only fills their content.
    # This prevents the whole page from flashing/resetting on each refresh.
    ph_status  = st.empty()
    ph_metrics = st.empty()
    ph_charts  = st.empty()
    ph_valid   = st.empty()
    ph_timeline = st.empty()
    ph_alerts  = st.empty()
    ph_table   = st.empty()

    # ── Fragment: only refreshes data, fills the containers above ────────────
    @st.fragment(run_every=refresh_sec if auto_refresh else None)
    def render():
        if not os.path.exists(STREAM_FILE):
            ph_status.warning("⏳ Waiting for stream simulator to start…")
            return

        try:
            with open(STREAM_FILE, 'r') as f:
                state = json.load(f)
        except Exception:
            return

        tick = state.get('tick', -1)
        if tick == st.session_state.last_tick:
            ph_status.info(f"⏸ Waiting for new data… (tick {tick})")
            return

        rows = pd.DataFrame(state['rows'])
        if rows.empty:
            return

        results = infer(engine, rows)
        st.session_state.last_tick = tick
        st.session_state.total_processed += len(rows)

        for i in range(len(results)):
            r = results.iloc[i]
            orig_row = rows.iloc[i]
            st.session_state.history.append({
                'time':        state.get('timestamp', ''),
                'is_attack':   r['is_attack'],
                'family':      r['stage1_label'],
                'subtype':     r['stage2_subtype'],
                'confidence':  r['confidence'],
                'true_binary': orig_row.get('true_binary_label', None),
            })

        if len(st.session_state.history) > max_history:
            st.session_state.history = st.session_state.history[-max_history:]

        history_df = pd.DataFrame(st.session_state.history)
        n, n_attack, n_benign, attack_rate, fam_counts, sub_counts = build_summary(history_df)

        # ── Status ────────────────────────────────────────────────────────────
        ph_status.success(
            f"🟢 Live | Tick {tick} | {state.get('timestamp','')} | "
            f"Total processed: {st.session_state.total_processed:,}"
        )

        # ── Metrics ───────────────────────────────────────────────────────────
        with ph_metrics.container():
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("📦 Samples",  f"{n:,}")
            c2.metric("✅ Benign",   f"{n_benign:,}", delta=f"{100-attack_rate:.1f}%")
            c3.metric("🚨 Attacks",  f"{n_attack:,}", delta=f"{attack_rate:.1f}%", delta_color="inverse")
            c4.metric("⚡ Latency",  "< 1ms / sample")
            c5.metric("🎯 Families", f"{len(fam_counts)} detected")

        # ── Charts ────────────────────────────────────────────────────────────
        with ph_charts.container():
            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("Attack Family Distribution")
                if not fam_counts.empty:
                    chart_df = pd.DataFrame({'Family': fam_counts.index, 'Count': fam_counts.values})
                    st.bar_chart(chart_df.set_index('Family'), color='#FF4B4B')
                else:
                    st.success("✅ No attacks detected!")
            with col_right:
                st.subheader("Top Attack Subtypes")
                if not sub_counts.empty:
                    sub_df = pd.DataFrame({'Subtype': sub_counts.index, 'Count': sub_counts.values})
                    st.bar_chart(sub_df.set_index('Subtype'), color='#9B59B6')
                else:
                    st.success("✅ Network is clean!")

        # ── Validation ────────────────────────────────────────────────────────
        valid_history = history_df.dropna(subset=['true_binary'])
        if not valid_history.empty:
            from sklearn.metrics import accuracy_score, recall_score, f1_score
            y_pred_bin = valid_history['is_attack'].astype(int).values
            y_true_bin = valid_history['true_binary'].astype(int).values
            with ph_valid.container():
                st.subheader("🎯 Real-time Validation (vs Ground Truth)")
                v1, v2, v3 = st.columns(3)
                v1.metric("Live Accuracy",       f"{accuracy_score(y_true_bin, y_pred_bin)*100:.2f}%")
                v2.metric("Live Recall (Attack)", f"{recall_score(y_true_bin, y_pred_bin, zero_division=0)*100:.2f}%")
                v3.metric("Live F1-Score",        f"{f1_score(y_true_bin, y_pred_bin, zero_division=0)*100:.2f}%")

        # ── Timeline ──────────────────────────────────────────────────────────
        with ph_timeline.container():
            st.subheader("📈 Live Timeline")
            timeline = history_df.copy()
            timeline['Attack'] = timeline['is_attack'].astype(int)
            timeline['Benign'] = (~timeline['is_attack']).astype(int)
            st.area_chart(
                timeline[['Attack', 'Benign']].rename(
                    columns={'Attack': '🔴 Attack', 'Benign': '🟢 Benign'}
                )
            )

        # ── Alerts ────────────────────────────────────────────────────────────
        with ph_alerts.container():
            st.subheader("🔔 Recent Alerts")
            recent_attacks = history_df[history_df['is_attack']].tail(5) if 'family' in history_df.columns else pd.DataFrame()
            if recent_attacks.empty:
                st.success("No recent attacks!")
            else:
                for _, row in recent_attacks.iterrows():
                    st.markdown(f"""
                    <div class="attack-card">
                        🚨 <b>{row.get('subtype','?')}</b> &nbsp;|&nbsp;
                        Family: <b>{row.get('family','?')}</b> &nbsp;|&nbsp;
                        Confidence: <b>{row.get('confidence',0):.1f}%</b> &nbsp;|&nbsp;
                        Time: {row.get('time','')}
                    </div>""", unsafe_allow_html=True)

        # ── Table ─────────────────────────────────────────────────────────────
        with ph_table.container():
            st.subheader("📋 Last 20 Records")
            show_cols = [c for c in ['time','family','subtype','confidence','is_attack'] if c in history_df.columns]
            show = history_df.tail(20)[show_cols].copy()
            if 'is_attack' in show.columns:
                show['is_attack'] = show['is_attack'].map({True: '🔴 Attack', False: '🟢 Benign'})
            st.dataframe(show, use_container_width=True, height=300)

    try:
        render()
    except Exception as e:
        import traceback
        st.error(f"Error rendering dashboard: {e}")
        st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📂 File Upload":
    if uploaded and run_btn:
        engine = load_engine()
        df_raw = pd.read_csv(uploaded)
        y_true = df_raw[[c for c in df_raw.columns if 'label' in c.lower() or 'true' in c.lower()]].copy()

        with st.spinner("Running inference…"):
            t0 = time.time()
            results = infer(engine, df_raw)
            elapsed = time.time() - t0

        n, n_attack, n_benign, attack_rate, fam_counts, sub_counts = build_summary(results)

        # Metrics
        st.success(f"✅ Inference complete in {elapsed:.2f}s | {elapsed/len(results)*1000:.2f} ms/sample")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📦 Total",    f"{n:,}")
        c2.metric("✅ Benign",   f"{n_benign:,}",  delta=f"{100-attack_rate:.1f}% safe")
        c3.metric("🚨 Attacks",  f"{n_attack:,}",  delta=f"{attack_rate:.1f}%", delta_color="inverse")
        c4.metric("⚡ Latency",  f"{elapsed/n*1000:.2f} ms/sample")

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Attack Families")
            if not fam_counts.empty:
                st.bar_chart(pd.DataFrame({'Count': fam_counts}), color='#FF4B4B')
        with col2:
            st.subheader("Top Subtypes")
            if not sub_counts.empty:
                st.bar_chart(pd.DataFrame({'Count': sub_counts}), color='#9B59B6')

        # Validation
        if not y_true.empty and 'true_binary_label' in y_true.columns:
            from sklearn.metrics import accuracy_score, recall_score, f1_score
            y_pb  = results['is_attack'].astype(int).values
            y_tb  = y_true['true_binary_label'].values
            st.divider()
            st.subheader("📊 Validation vs Ground Truth")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy",  f"{accuracy_score(y_tb, y_pb)*100:.2f}%")
            m2.metric("Recall",    f"{recall_score(y_tb, y_pb)*100:.2f}%")
            m3.metric("F1-Score",  f"{f1_score(y_tb, y_pb)*100:.2f}%")

        # Full results table
        st.subheader("📋 Full Results")
        display = results[['is_attack','stage1_label','stage2_subtype','confidence']].copy()
        display['is_attack'] = display['is_attack'].map({True: '🔴 Attack', False: '🟢 Benign'})
        st.dataframe(display, use_container_width=True, height=400)

        # Download button
        csv_out = results.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results CSV", csv_out, "ids_results.csv", "text/csv")

    elif not uploaded:
        st.info("👈 Upload a CSV file from the sidebar to get started.")
        st.markdown("""
        ### Expected CSV format
        The CSV should contain the same **46 network traffic features** used during training.
        You can use `Inference/data_samples/sample_traffic.csv` as a reference.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3: SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "🔍 Explainability (SHAP)":
    import pickle
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import shap

    st.title("🔍 Model Explainability — SHAP Analysis")
    st.caption("Understand WHY the binary IDS model flags traffic as Attack or Benign")

    BINARY_PKG = os.path.join(BASE, 'models', '01_binary', 'binary_inference_package.pkl')
    SAMPLE_CSV = os.path.join(BASE, 'data_samples', 'sample_traffic.csv')

    @st.cache_resource(show_spinner="Building SHAP explainer… (one-time)")
    def get_shap_explainer():
        with open(BINARY_PKG, 'rb') as f:
            pkg = pickle.load(f)
        mdl    = pkg['model']
        scaler = pkg['scaler']
        feats  = pkg['feature_names']
        # Force CPU for SHAP — prevents any CUDA context issues
        try:
            mdl.set_params(device='cpu', n_jobs=-1)
        except Exception:
            pass
        return mdl, scaler, feats

    @st.cache_data(show_spinner="Computing SHAP values…")
    def compute_shap(_model, _scaler, feature_names, n_samples):
        df_raw = pd.read_csv(SAMPLE_CSV)
        df_feat = df_raw[feature_names].copy()
        sample = df_feat.sample(n=min(n_samples, len(df_feat)), random_state=42)
        X_scaled = _scaler.transform(sample.values)

        # ✅ TreeExplainer — works on CPU, stable with Streamlit threads
        explainer = shap.TreeExplainer(_model)
        shap_values = explainer(X_scaled)
        return shap_values, sample, X_scaled, df_raw.iloc[sample.index]

    try:
        model, scaler, feature_names = get_shap_explainer()
        shap_values, sample_df, X_scaled, raw_with_labels = compute_shap(
            model, scaler, feature_names, shap_n_samples
        )

        st.success(f"✅ SHAP analysis on **{len(sample_df)}** samples | Binary Model: Benign vs Attack")
        st.divider()

        # ── Tab layout ────────────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs([
            "📊 Global Feature Importance",
            "🌊 SHAP Beeswarm",
            "🔎 Individual Prediction"
        ])

        # ── Tab 1: Global bar chart of mean |SHAP| per feature ────────────────
        with tab1:
            st.subheader("Top 20 Features — Mean |SHAP| Impact")
            st.caption(
                "Higher bar = this feature has more influence on the model's attack/benign decision overall."
            )
            mean_shap = np.abs(shap_values.values).mean(axis=0)
            feat_df = pd.DataFrame({
                'Feature': feature_names,
                'Mean |SHAP|': mean_shap
            }).sort_values('Mean |SHAP|', ascending=False).head(20)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(feat_df['Feature'][::-1], feat_df['Mean |SHAP|'][::-1],
                           color='#FF4B4B')
            ax.set_xlabel("Mean |SHAP value| (impact on model output)", fontsize=11)
            ax.set_title("Global Feature Importance (SHAP)", fontsize=13, fontweight='bold')
            ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.divider()
            st.subheader("📋 Full SHAP Feature Ranking")
            all_feat_df = pd.DataFrame({
                'Rank': range(1, len(feature_names)+1),
                'Feature': [feature_names[i] for i in np.argsort(mean_shap)[::-1]],
                'Mean |SHAP| Impact': sorted(mean_shap, reverse=True)
            })
            st.dataframe(all_feat_df, use_container_width=True, height=400)

        # ── Tab 2: Beeswarm plot ───────────────────────────────────────────────
        with tab2:
            st.subheader("SHAP Beeswarm — Feature Value vs Impact")
            st.caption(
                "Each dot is one sample. Red = high feature value, Blue = low. "
                "Dots on the right = pushed toward Attack prediction."
            )
            fig2, ax2 = plt.subplots(figsize=(10, 7))
            shap.plots.beeswarm(shap_values, max_display=20, show=False)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close('all')

        # ── Tab 3: Individual prediction waterfall ────────────────────────────
        with tab3:
            st.subheader("Individual Sample Explanation — Waterfall Chart")
            st.caption(
                "For one sample: shows which features PUSHED the decision toward Attack (red) "
                "or Benign (blue), and by how much."
            )

            # Let user pick sample index
            has_labels = 'true_binary_label' in raw_with_labels.columns
            if has_labels:
                labels = raw_with_labels['true_binary_label'].values
                label_display = [
                    f"Sample {i} — {'🔴 Attack' if labels[i]==1 else '🟢 Benign'}"
                    for i in range(len(sample_df))
                ]
            else:
                label_display = [f"Sample {i}" for i in range(len(sample_df))]

            selected = st.selectbox("Choose a sample to explain:", label_display)
            idx = int(selected.split()[1])

            pred_class = "🔴 Attack" if model.predict(X_scaled[idx:idx+1])[0] == 1 else "🟢 Benign"
            pred_prob  = model.predict_proba(X_scaled[idx:idx+1])[0][1] * 100
            if has_labels:
                true_label = "🔴 Attack" if labels[idx] == 1 else "🟢 Benign"
                correct = "✅ Correct" if pred_class[:2] == true_label[:2] else "❌ Wrong"
                st.info(f"**Prediction:** {pred_class} ({pred_prob:.1f}% confidence)  |  **Ground Truth:** {true_label}  |  {correct}")
            else:
                st.info(f"**Prediction:** {pred_class} ({pred_prob:.1f}% confidence)")

            shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
            fig3 = plt.gcf()
            fig3.set_size_inches(10, 7)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            st.divider()
            st.subheader("📋 Top Contributing Features for this Sample")
            sample_shap = shap_values.values[idx]
            contrib_df = pd.DataFrame({
                'Feature': feature_names,
                'Raw Value': sample_df.iloc[idx].values,
                'SHAP Impact': sample_shap,
                'Direction': ['⬆ Attack' if v > 0 else '⬇ Benign' for v in sample_shap]
            }).sort_values('SHAP Impact', key=abs, ascending=False).head(15)
            st.dataframe(contrib_df, use_container_width=True)

    except Exception as e:
        import traceback
        st.error(f"SHAP Error: {e}")
        st.code(traceback.format_exc())

