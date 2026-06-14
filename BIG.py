import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
    /* Background color ya bluu ya giza na gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #f0f0f0;
    }

    /* Animated heading inayotiririka */
    @keyframes shimmer {
        0%   { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    .animated-title {
        font-size: 3em;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #f7971e, #ffd200, #f7971e, #ff6b6b, #f7971e);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 3s linear infinite;
        letter-spacing: 2px;
        margin-bottom: 0.2em;
        font-family: 'Georgia', serif;
    }

    .subtitle {
        text-align: center;
        color: #aed6f1;
        font-size: 1.1em;
        margin-bottom: 1.5em;
        letter-spacing: 1px;
    }

    /* Card style kwa sections */
    .card {
        background: rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1.5em;
        margin: 1em 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    /* Section headers */
    h2, h3 {
        color: #ffd200 !important;
        font-family: 'Georgia', serif !important;
    }

    /* Dataframe background */
    .stDataFrame {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 10px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #f7971e, #ffd200);
        color: #1a1a2e;
        font-weight: 800;
        border: none;
        border-radius: 25px;
        padding: 0.6em 2em;
        font-size: 1em;
        letter-spacing: 1px;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(255, 210, 0, 0.5);
    }

    /* File uploader */
    .stFileUploader {
        border: 2px dashed #ffd200 !important;
        border-radius: 12px;
        padding: 1em;
    }

    /* Info / warning boxes */
    .stInfo, .stWarning {
        border-radius: 10px;
    }

    /* Metric labels */
    .metric-label {
        color: #aed6f1;
        font-size: 0.9em;
    }
    .metric-value {
        color: #ffd200;
        font-size: 1.6em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ===================== ANIMATED HEADING =====================
st.markdown('<div class="animated-title">🏠 HOUSE PRICE PREDICTION</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">✨ Pakia faili lako la CSV · Angalia takwimu · Pata bei ya nyumba ✨</div>', unsafe_allow_html=True)
st.markdown("---")

# ===================== LOAD MODEL =====================
MODEL_PATH = "DATA_SIMPLE.joblib"

model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Model imepakiwa vizuri!")
    except Exception as e:
        st.error(f"❌ Hitilafu kupakia model: {e}")
else:
    st.warning("⚠️ Faili la model halijapatikana.")
st.markdown("---")

# ===================== FILE UPLOAD =====================
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.subheader("📂 Pakia Faili Lako")
    file = st.file_uploader("Chagua faili la CSV", type="csv", label_visibility="collapsed")

with col_info:
    st.markdown("""
    <div class="card">
        <div class="metric-label">Jinsi ya kutumia:</div>
        <ol style="color:#aed6f1; font-size:0.9em;">
            <li>Pakia faili la CSV</li>
            <li>Angalia takwimu na grafu</li>
            <li>Bonyeza <b style='color:#ffd200'>Predict</b></li>
            <li>Pakua matokeo</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

button = st.button("🔮 Predict House Price")

# ===================== MAIN LOGIC =====================
if file is not None:
    df = pd.read_csv(file)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # --- Data Preview ---
    st.markdown("---")
    st.subheader("📊 Muhtasari wa Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="card"><div class="metric-label">Safu (Rows)</div><div class="metric-value">{df.shape[0]:,}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><div class="metric-label">Nguzo (Columns)</div><div class="metric-value">{df.shape[1]}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><div class="metric-label">Nguzo za Nambari</div><div class="metric-value">{len(numeric_cols)}</div></div>', unsafe_allow_html=True)

    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📈 Maelezo ya Takwimu (Describe)"):
        st.write(df.describe())

    # --- GRAPHS ---
    st.markdown("---")
    st.subheader("📊 Michoro ya Data")

    if len(numeric_cols) > 0:
        tab1, tab2, tab3 = st.tabs(["🥧 Pie Chart", "📊 Histogram", "🌡️ Heatmap"])

        # ====== PIE CHART ======
        with tab1:
            st.markdown("#### 🥧 Mgawanyo wa Nguzo za Nambari")
            st.caption("Grafu hii inaonyesha jumla ya kila nguzo ya nambari kwa asilimia")

            col_pie, col_pie_info = st.columns([2, 1])
            with col_pie:
                # Pie chart using absolute sums of each numeric column
                pie_values = df[numeric_cols].abs().sum()
                pie_values = pie_values[pie_values > 0]  # Remove zero columns

                if len(pie_values) > 0:
                    COLORS = [
                        "#ffd200", "#f7971e", "#ff6b6b", "#4ecdc4",
                        "#45b7d1", "#96e6a1", "#dda0dd", "#87ceeb",
                        "#ffb347", "#b19cd9"
                    ]
                    colors = (COLORS * ((len(pie_values) // len(COLORS)) + 1))[:len(pie_values)]

                    fig_pie, ax_pie = plt.subplots(figsize=(7, 7), facecolor='none')
                    fig_pie.patch.set_facecolor('#0f2027')
                    ax_pie.set_facecolor('#0f2027')

                    wedges, texts, autotexts = ax_pie.pie(
                        pie_values,
                        labels=None,
                        autopct='%1.1f%%',
                        colors=colors,
                        startangle=140,
                        pctdistance=0.75,
                        wedgeprops=dict(width=0.6, edgecolor='#0f2027', linewidth=2),
                        shadow=True
                    )

                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontsize(9)
                        autotext.set_fontweight('bold')

                    # Center text (donut style)
                    ax_pie.text(0, 0, 'Data\nGawanyo', ha='center', va='center',
                                fontsize=12, color='#ffd200', fontweight='bold')

                    # Legend
                    legend_patches = [
                        mpatches.Patch(color=colors[i], label=f"{name}")
                        for i, name in enumerate(pie_values.index)
                    ]
                    ax_pie.legend(
                        handles=legend_patches,
                        loc='lower center',
                        bbox_to_anchor=(0.5, -0.15),
                        ncol=3,
                        fontsize=8,
                        labelcolor='white',
                        facecolor='#203a43',
                        edgecolor='#ffd200',
                        framealpha=0.8
                    )

                    ax_pie.set_title("Mgawanyo wa Nguzo za Nambari", color='#ffd200',
                                     fontsize=14, fontweight='bold', pad=20)
                    plt.tight_layout()
                    st.pyplot(fig_pie)
                else:
                    st.warning("Hakuna data ya kutosha kwa pie chart")

            with col_pie_info:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**Kila rangi inawakilisha nguzo moja:**")
                for i, col_name in enumerate(pie_values.index):
                    pct = pie_values[col_name] / pie_values.sum() * 100
                    st.markdown(f"• **{col_name}**: {pct:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)

        # ====== HISTOGRAM ======
        with tab2:
            st.markdown("#### 📊 Histogram ya Nguzo")
            selected_col = st.selectbox("Chagua nguzo:", numeric_cols)

            fig_hist, ax_hist = plt.subplots(figsize=(9, 4), facecolor='none')
            fig_hist.patch.set_facecolor('#0f2027')
            ax_hist.set_facecolor('#1a2a3a')

            n, bins, patches = ax_hist.hist(df[selected_col].dropna(), bins=25,
                                             edgecolor='#0f2027', linewidth=0.5)

            # Color gradient on bars
            norm_vals = (n - n.min()) / (n.max() - n.min() + 1e-9)
            cmap = plt.cm.YlOrRd
            for patch, val in zip(patches, norm_vals):
                patch.set_facecolor(cmap(val))

            ax_hist.set_title(f"Mgawanyo wa '{selected_col}'", color='#ffd200',
                              fontsize=13, fontweight='bold')
            ax_hist.set_xlabel(selected_col, color='#aed6f1')
            ax_hist.set_ylabel("Idadi", color='#aed6f1')
            ax_hist.tick_params(colors='#aed6f1')
            for spine in ax_hist.spines.values():
                spine.set_edgecolor('#2c5364')

            plt.tight_layout()
            st.pyplot(fig_hist)

        # ====== HEATMAP ======
        with tab3:
            if len(numeric_cols) > 1:
                st.markdown("#### 🌡️ Heatmap ya Mahusiano (Correlation)")

                fig_hm, ax_hm = plt.subplots(figsize=(9, 6), facecolor='none')
                fig_hm.patch.set_facecolor('#0f2027')
                ax_hm.set_facecolor('#0f2027')

                corr = df[numeric_cols].corr()
                im = ax_hm.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect='auto')

                cbar = plt.colorbar(im, ax=ax_hm)
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

                ax_hm.set_xticks(range(len(numeric_cols)))
                ax_hm.set_yticks(range(len(numeric_cols)))
                ax_hm.set_xticklabels(numeric_cols, rotation=45, ha='right',
                                      color='#aed6f1', fontsize=8)
                ax_hm.set_yticklabels(numeric_cols, color='#aed6f1', fontsize=8)

                # Add correlation values
                for i in range(len(numeric_cols)):
                    for j in range(len(numeric_cols)):
                        val = corr.iloc[i, j]
                        ax_hm.text(j, i, f"{val:.2f}", ha='center', va='center',
                                   color='black' if abs(val) > 0.5 else 'white', fontsize=7)

                ax_hm.set_title("Mahusiano kati ya Nguzo", color='#ffd200',
                                fontsize=13, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_hm)
            else:
                st.info("Heatmap inahitaji nguzo 2 au zaidi za nambari")
    else:
        st.warning("⚠️ Hakuna nguzo za nambari kwenye faili lako")

    # ===================== PREDICTION =====================
    st.markdown("---")
    if button:
        if model is None:
            st.error("❌ Mfano haujapatikana. Hakikisha MODEL_PATH ni sahihi.")
        else:
            try:
                with st.spinner("⏳ Inakokotoa bei..."):
                    prediction = model.predict(df)

                st.subheader("🔮 Matokeo ya Utabiri")
                result = df.copy()
                result["💰 Bei Iliyotabiriwa"] = prediction

                st.dataframe(result, use_container_width=True)

                # Summary stats for predictions
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.markdown(f'<div class="card"><div class="metric-label">Bei ya Chini</div><div class="metric-value">{prediction.min():,.0f}</div></div>', unsafe_allow_html=True)
                with col_s2:
                    st.markdown(f'<div class="card"><div class="metric-label">Bei ya Wastani</div><div class="metric-value">{prediction.mean():,.0f}</div></div>', unsafe_allow_html=True)
                with col_s3:
                    st.markdown(f'<div class="card"><div class="metric-label">Bei ya Juu</div><div class="metric-value">{prediction.max():,.0f}</div></div>', unsafe_allow_html=True)

                # Download
                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Pakua Matokeo (CSV)",
                    csv,
                    "matokeo_ya_bei.csv",
                    "text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ Hitilafu: {e}")
    elif not button:
        st.info("👆 Bonyeza kitufe cha **Predict House Price** kupata matokeo ya utabiri")

else:
    # Welcome screen when no file uploaded
    st.markdown("""
    <div class="card" style="text-align:center; padding: 3em;">
        <div style="font-size:4em;">📂</div>
        <div style="font-size:1.4em; color:#ffd200; font-weight:bold; margin:0.5em 0;">
            Pakia Faili Lako la CSV
        </div>
        <div style="color:#aed6f1; font-size:1em;">
            Faili lako la data linapaswa kuwa na nguzo zinazofanana na mfano uliofunzwa.<br>
            Matokeo yataonekana hapa baada ya kupakia.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a6278; font-size:0.8em; margin-top:1em;">
    🏠 House Price Prediction App · Imetengenezwa na Streamlit · 
    <span style="color:#ffd200;">Powered by Machine Learning</span>
</div>
""", unsafe_allow_html=True)