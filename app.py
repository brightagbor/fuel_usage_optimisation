"""
AutoMPG Fuel Intelligence Dashboard
Federal Polytechnic Nekede, Owerri — Department of Computer Engineering
Production-ready app.py with Bayesian optimisation, theme switch, PDF/CSV export.
"""

import io
import csv
import datetime
import numpy as np
import joblib
import streamlit as st
from skopt import gp_minimize

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoMPG · Fuel Intelligence",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# THEME  (sidebar toggle stored in session_state)
# ──────────────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

with st.sidebar:
    st.markdown("### ⚙️ Display")
    mode_label = "🌙 Dark Mode" if not st.session_state["dark_mode"] else "🌞 Light Mode"
    if st.button(mode_label, use_container_width=True):
        st.session_state["dark_mode"] = not st.session_state["dark_mode"]
        st.rerun()
    dark = st.session_state["dark_mode"]

# ──────────────────────────────────────────────────────────────────────────────
# THEME TOKENS
# ──────────────────────────────────────────────────────────────────────────────
if dark:
    T = dict(
        bg        = "#0f1117",
        surface   = "#1c1e26",
        surface2  = "#252833",
        border    = "#2e3040",
        ink       = "#e8eaf0",
        mid       = "#8b8fa8",
        faint     = "#4a4e65",
        accent    = "#4ade80",      # green-400
        accent2   = "#22c55e",      # green-500
        danger    = "#f87171",      # red-400
        blue      = "#60a5fa",
        card_hi   = "#162d1e",
        badge_bg  = "#166534",
        badge_neg = "#7f1d1d",
        sidebar_bg= "#13151f",
    )
else:
    T = dict(
        bg        = "#f5f7fa",
        surface   = "#ffffff",
        surface2  = "#eef1f6",
        border    = "#dde1eb",
        ink       = "#0f172a",
        mid       = "#64748b",
        faint     = "#cbd5e1",
        accent    = "#16a34a",      # green-600
        accent2   = "#15803d",      # green-700
        danger    = "#dc2626",
        blue      = "#2563eb",
        card_hi   = "#f0fdf4",
        badge_bg  = "#dcfce7",
        badge_neg = "#fee2e2",
        sidebar_bg= "#f8fafc",
    )

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Sora:wght@700;800&display=swap');

/* ── Reset & base ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {{
    background: {T['bg']} !important;
    color: {T['ink']} !important;
    font-family: 'Inter', sans-serif !important;
}}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
footer {{ display: none !important; }}

/* Main container */
[data-testid="block-container"] {{
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1200px !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {T['sidebar_bg']} !important;
    border-right: 1px solid {T['border']} !important;
}}
[data-testid="stSidebar"] * {{
    color: {T['ink']} !important;
}}

/* ── Masthead ── */
.masthead {{
    background: linear-gradient(135deg, {T['surface']} 0%, {T['surface2']} 100%);
    border: 1px solid {T['border']};
    border-left: 5px solid {T['accent']};
    border-radius: 12px;
    padding: 2.2rem 2.5rem 1.8rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: flex-start;
    gap: 1.5rem;
}}
.masthead-icon {{
    font-size: 3rem;
    line-height: 1;
    flex-shrink: 0;
}}
.masthead-body {{ flex: 1; }}
.masthead-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {T['accent']};
    margin-bottom: 0.4rem;
}}
.masthead-title {{
    font-family: 'Sora', sans-serif;
    font-size: clamp(1rem, 2vw, 2rem);
    font-weight: 400;
    color: {T['ink']};
    line-height: 1;
    margin-bottom: 0.4rem;
}}
.masthead-sub {{
    font-size: 0.9rem;
    color: {T['mid']};
    line-height: 1.5;
}}
.masthead-pills {{
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.9rem;
}}
.pill {{
    background: {T['surface2']};
    border: 1px solid {T['border']};
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: {T['mid']};
    letter-spacing: 0.05em;
}}

/* ── Section heading ── */
.sec-head {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.1rem;
    padding-bottom: 0.55rem;
    border-bottom: 2px solid {T['border']};
}}
.sec-head-icon {{
    width: 28px; height: 28px;
    background: {T['accent']};
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem;
    flex-shrink: 0;
}}
.sec-head-text {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {T['mid']};
}}

/* ── Card ── */
.card {{
    background: {T['surface']};
    border: 1px solid {T['border']};
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}}

/* ── Metric cards ── */
.metric-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 1.75rem;
}}
.metric-card {{
    background: {T['surface']};
    border: 1px solid {T['border']};
    border-radius: 10px;
    padding: 1.5rem 1.75rem;
    text-align: center;
    transition: box-shadow 0.2s;
}}
.metric-card:hover {{
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}}
.metric-card.optimal {{
    border-color: {T['accent']};
    background: {T['card_hi']};
}}
.metric-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {T['mid']};
    margin-bottom: 0.75rem;
}}
.metric-value {{
    font-family: 'Sora', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: {T['ink']};
    line-height: 1;
    margin-bottom: 0.3rem;
}}
.metric-value.green {{ color: {T['accent']}; }}
.metric-unit {{
    font-size: 0.75rem;
    color: {T['mid']};
    margin-bottom: 0.75rem;
}}
.metric-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: {T['badge_bg']};
    color: {T['accent']};
    border: 1px solid {T['accent']};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
}}
.metric-badge.neg {{
    background: {T['badge_neg']};
    color: {T['danger']};
    border-color: {T['danger']};
}}

/* ── Comparison table ── */
.cmp-wrap {{
    background: {T['surface']};
    border: 1px solid {T['border']};
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}}
.cmp-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}}
.cmp-table thead {{
    background: {T['surface2']};
}}
.cmp-table th {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {T['mid']};
    padding: 0.85rem 1rem;
    text-align: left;
    border-bottom: 1px solid {T['border']};
}}
.cmp-table td {{
    padding: 0.75rem 1rem;
    border-bottom: 1px solid {T['border']};
    color: {T['ink']};
    vertical-align: middle;
}}
.cmp-table tbody tr:last-child td {{ border-bottom: none; }}
.cmp-table tbody tr:hover td {{ background: {T['surface2']}; }}
.feat-name {{
    font-weight: 500;
    color: {T['ink']};
}}
.feat-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
}}
.bar-track {{
    height: 5px;
    background: {T['faint']};
    border-radius: 3px;
    margin-top: 5px;
    overflow: hidden;
}}
.bar-fill-curr {{ height: 5px; border-radius: 3px; background: {T['faint']}; }}
.bar-fill-opt  {{ height: 5px; border-radius: 3px; background: {T['accent']}; }}
.chg-up   {{ color: {T['accent']};  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600; }}
.chg-down {{ color: {T['danger']};  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600; }}
.chg-same {{ color: {T['mid']};     font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }}

/* ── Sidebar inputs ── */
[data-testid="stSidebar"] [data-testid="stNumberInput"] label,
[data-testid="stSidebar"] [data-testid="stSelectbox"] label,
[data-testid="stSidebar"] [data-testid="stSlider"] label {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: {T['mid']} !important;
    font-weight: 600 !important;
}}
[data-testid="stNumberInput"] input {{
    background: {T['surface']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 6px !important;
    color: {T['ink']} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}
[data-testid="stNumberInput"] input:focus {{
    border-color: {T['accent']} !important;
    outline: none !important;
}}
[data-testid="stSelectbox"] > div > div {{
    background: {T['surface']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 6px !important;
    color: {T['ink']} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] {{
    background: {T['accent']} !important;
    border: 2px solid {T['surface']} !important;
}}

/* ── Button ── */
.stButton > button {{
    background: {T['accent']} !important;
    border: none !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 8px rgba(22,163,74,0.3) !important;
    letter-spacing: 0.01em !important;
}}
.stButton > button:hover {{
    background: {T['accent2']} !important;
    box-shadow: 0 4px 16px rgba(22,163,74,0.4) !important;
    transform: translateY(-1px) !important;
}}
.stButton > button:active {{ transform: translateY(0) !important; }}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {{
    background: {T['surface2']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    color: {T['ink']} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.25rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    border-color: {T['accent']} !important;
    color: {T['accent']} !important;
    background: {T['card_hi']} !important;
}}

/* ── Progress ── */
.stProgress > div > div > div > div {{
    background: {T['accent']} !important;
    border-radius: 4px !important;
}}

/* ── Alerts ── */
.stAlert {{ border-radius: 8px !important; }}

/* ── Results wrapper animation ── */
.results-wrap {{
    animation: fadeUp 0.4s ease both;
}}
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* ── Divider ── */
.rule {{ border: none; border-top: 1px solid {T['border']}; margin: 1.75rem 0; }}

/* ── Footer ── */
.footer {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    color: {T['faint']};
    text-align: center;
    margin-top: 3rem;
    padding-top: 1.25rem;
    border-top: 1px solid {T['border']};
}}

/* ── Live opt progress card ── */
.opt-progress-card {{
    background: {T['surface']};
    border: 1px solid {T['accent']};
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}}
.opt-progress-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {T['accent']};
    margin-bottom: 0.5rem;
}}

/* ── Summary info row ── */
.info-row {{
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}}
.info-chip {{
    background: {T['surface2']};
    border: 1px solid {T['border']};
    border-radius: 6px;
    padding: 0.45rem 0.9rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: {T['mid']};
}}
.info-chip strong {{ color: {T['ink']}; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (100% preserved)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    import os, pickle

    model = joblib.load("results/ensemble1_best_model.pkl")

    if hasattr(model, "named_steps") or hasattr(model, "steps"):
        return model, None

    scaler = None
    for path in ["results/scaler.pkl"]:
        if os.path.exists(path):
            try:
                scaler = joblib.load(path)
                break
            except Exception:
                pass

    if scaler is None:
        for path in ["results/scaler_scaler.pkl"]:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        scaler = pickle.load(f)
                    break
                except Exception:
                    pass

    return model, scaler

try:
    model, scaler = load_artifacts()
    model_ok = True
except Exception as e:
    model_ok = False
    load_err = str(e)
    model, scaler = None, None

# ──────────────────────────────────────────────────────────────────────────────
# SAFE PREDICT  (100% preserved)
# ──────────────────────────────────────────────────────────────────────────────
def safe_predict(X_raw):
    X = np.array(X_raw, dtype=float).reshape(1, -1)
    if hasattr(model, "steps") or hasattr(model, "named_steps"):
        return float(model.predict(X)[0])
    if scaler is not None:
        X = scaler.transform(X)
    return float(model.predict(X)[0])

ORIGIN_MAP = {1: "🇺🇸 USA", 2: "🇪🇺 Europe", 3: "🇯🇵 Japan"}
ORIGIN_SHORT = {1: "USA", 2: "Europe", 3: "Japan"}

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR DIAGNOSTIC
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔍 Model Diagnostics")
    if model_ok:
        st.success(f"Model loaded: `{type(model).__name__}`")
        if scaler is not None:
            st.info(f"Scaler: `{type(scaler).__name__}`")
        else:
            st.info("No companion scaler")
        try:
            p_low  = safe_predict([4, 90.0,  65.0, 1800.0, 20.0, 82, 3])
            p_high = safe_predict([8, 400.0, 200.0, 5000.0, 10.0, 70, 1])
            st.caption(f"Probe (efficient): `{p_low:.2f} MPG`")
            st.caption(f"Probe (heavy):     `{p_high:.2f} MPG`")
            if abs(p_low - p_high) < 0.5:
                st.warning("⚠️ Outputs nearly identical — scaler may be missing.")
        except Exception as ex:
            st.caption(f"Probe error: {ex}")
    else:
        st.error(f"Load error: {load_err}")

# ──────────────────────────────────────────────────────────────────────────────
# MASTHEAD
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
  <div class="masthead-icon">⛽</div>
  <div class="masthead-body">
    <div class="masthead-eyebrow">Bayesian Optimisation · Ensemble ML · UCI Auto-MPG</div>
    <div class="masthead-title">An Intelligent System for Fuel Efficiency Prediction and Optimal Vehicle Design</div>
    <div class="masthead-sub">
      Department of Computer Engineering &nbsp;·&nbsp; Federal Polytechnic Nekede, Owerri<br>
      Predict fuel efficiency and discover the optimal vehicle configuration.
    </div>
    <div class="masthead-pills">
      <span class="pill">Gaussian Process (gp_minimize)</span>
      <span class="pill">80 evaluations</span>
      <span class="pill">EI acquisition</span>
      <span class="pill">Ensemble Regressor</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not model_ok:
    st.error(f"**Cannot load model.** `{load_err}`  \nPlace `ensemble1_best_model.pkl` in `results/`.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# INPUT PANEL  — two-column layout in main body
# ──────────────────────────────────────────────────────────────────────────────
col_inp, col_bounds = st.columns([1, 1], gap="large")

with col_inp:
    st.markdown("""
    <div class="sec-head">
      <div class="sec-head-icon">🚗</div>
      <span class="sec-head-text">Vehicle Specifications</span>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        cylinders    = st.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
        displacement = st.number_input("Displacement (cu·in)", 50.0, 500.0, 200.0, step=1.0)
        horsepower   = st.number_input("Horsepower (hp)", 40.0, 250.0, 100.0, step=1.0)
        weight       = st.number_input("Weight (lbs)", 1500.0, 5500.0, 3000.0, step=10.0)
        acceleration = st.number_input("Acceleration (0–60 s)", 8.0, 25.0, 15.0, step=0.5)
        model_year   = st.selectbox(
            "Model Year",
            list(range(70, 83)),
            format_func=lambda y: f"19{y:02d}",
            index=8,
        )
        origin       = st.selectbox(
            "Origin",
            [1, 2, 3],
            format_func=lambda v: ORIGIN_MAP[v],
        )

input_vec = [cylinders, displacement, horsepower, weight, acceleration, model_year, origin]

with col_bounds:
    st.markdown("""
    <div class="sec-head">
      <div class="sec-head-icon">🎯</div>
      <span class="sec-head-text">Optimisation Search Bounds</span>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Constrain the search space for Bayesian optimisation.")

    cyl_r  = st.slider("Cylinders",             3,      8,      (3,   8))
    disp_r = st.slider("Displacement (cu·in)",  50.0,  500.0,  (68.0, 455.0))
    hp_r   = st.slider("Horsepower (hp)",        40.0,  250.0,  (46.0, 230.0))
    wt_r   = st.slider("Weight (lbs)",           1500.0,5500.0, (1600.0, 5000.0))
    acc_r  = st.slider("Acceleration (0–60 s)",  8.0,   25.0,   (8.0,  25.0))
    yr_r   = st.slider("Model Year",             70,    82,     (70,   82))
    org_r  = st.slider("Origin",                 1,     3,      (1,    3))

# ──────────────────────────────────────────────────────────────────────────────
# ACTION BUTTON
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
go = st.button("⛽  Predict MPG & Find Optimal Configuration", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# OPTIMISATION  (100% preserved gp_minimize workflow)
# ──────────────────────────────────────────────────────────────────────────────
if go:
    pred_mpg = safe_predict(input_vec)

    def safe_bound(lo, hi, pad=1.0):
        lo, hi = float(lo), float(hi)
        return (lo, hi) if hi - lo > 0.5 else (lo, lo + pad)

    bounds = [
        safe_bound(cyl_r[0],  cyl_r[1]),
        safe_bound(disp_r[0], disp_r[1], 10.0),
        safe_bound(hp_r[0],   hp_r[1],   10.0),
        safe_bound(wt_r[0],   wt_r[1],   100.0),
        safe_bound(acc_r[0],  acc_r[1],  1.0),
        safe_bound(yr_r[0],   yr_r[1],   1.0),
        safe_bound(org_r[0],  org_r[1],  1.0),
    ]

    N_CALLS = 80

    def predict_mpg_from_vec(x):
        cyl, disp, hp, wt, acc, year, org = x
        raw = [
            int(round(cyl)), float(disp), float(hp),
            float(wt), float(acc),
            int(round(year)), max(1, min(3, int(round(org)))),
        ]
        return safe_predict(raw)

    def objective(x):
        return -predict_mpg_from_vec(x)

    # Live progress display
    st.markdown('<div class="opt-progress-card">', unsafe_allow_html=True)
    st.markdown('<div class="opt-progress-title">🔄 Bayesian Optimisation Running</div>', unsafe_allow_html=True)

    pbar        = st.progress(0, text="Initialising…")
    live_best   = st.empty()
    live_eval   = st.empty()
    counter     = [0]
    best_so_far = [pred_mpg]

    def cb(res):
        counter[0] += 1
        best_now = -min(res.func_vals)
        best_so_far[0] = best_now
        pbar.progress(
            min(counter[0] / N_CALLS, 1.0),
            text=f"Evaluation {counter[0]} / {N_CALLS}",
        )
        live_best.markdown(
            f"<div style='font-family:JetBrains Mono,monospace;font-size:0.8rem;"
            f"color:{T['accent']};margin-top:0.25rem;'>"
            f"Best so far: <strong>{best_now:.2f} MPG</strong></div>",
            unsafe_allow_html=True,
        )

    result = gp_minimize(
        func             = objective,
        dimensions       = bounds,
        n_calls          = N_CALLS,
        n_initial_points = 20,
        acq_func         = "EI",
        random_state     = 0,
        callback         = cb,
        noise            = 1e-10,
        xi               = 0.05,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    best_x    = list(result.x)
    best_x[0] = int(round(best_x[0]))
    best_x[5] = int(round(best_x[5]))
    best_x[6] = max(1, min(3, int(round(best_x[6]))))
    best_mpg  = predict_mpg_from_vec(best_x)

    # Persist
    st.session_state["pred_mpg"]   = pred_mpg
    st.session_state["best_mpg"]   = best_mpg
    st.session_state["best_x"]     = best_x
    st.session_state["input_vec"]  = list(input_vec)
    st.session_state["bounds_r"]   = (cyl_r, disp_r, hp_r, wt_r, acc_r, yr_r, org_r)
    st.session_state["timestamp"]  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ──────────────────────────────────────────────────────────────────────────────
# RESULTS  (only shown when session state has data)
# ──────────────────────────────────────────────────────────────────────────────
if "pred_mpg" in st.session_state:
    pred_mpg  = st.session_state["pred_mpg"]
    best_mpg  = st.session_state["best_mpg"]
    best_x    = st.session_state["best_x"]
    orig      = st.session_state["input_vec"]
    cyl_r, disp_r, hp_r, wt_r, acc_r, yr_r, org_r = st.session_state["bounds_r"]
    ts        = st.session_state.get("timestamp", "")

    gain     = best_mpg - pred_mpg
    pct      = (gain / pred_mpg * 100) if pred_mpg else 0
    pos      = gain >= 0
    badge_cl = "metric-badge" if pos else "metric-badge neg"
    arrow    = "▲" if pos else "▼"
    badge_tx = f"{arrow} {abs(gain):.2f} MPG  ({pct:+.1f}%)"

    st.markdown('<div class="results-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-head">
      <div class="sec-head-icon">📊</div>
      <span class="sec-head-text">Analysis Results</span>
    </div>
    """, unsafe_allow_html=True)

    # Timestamp / context info
    st.markdown(f"""
    <div class="info-row">
      <div class="info-chip">Run: <strong>{ts}</strong></div>
      <div class="info-chip">Model: <strong>{type(model).__name__}</strong></div>
      <div class="info-chip">Evaluations: <strong>80</strong></div>
      <div class="info-chip">Gain: <strong>{badge_tx}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Your Configuration</div>
        <div class="metric-value">{pred_mpg:.1f}</div>
        <div class="metric-unit">miles per gallon</div>
      </div>
      <div class="metric-card optimal">
        <div class="metric-label">Optimal Configuration</div>
        <div class="metric-value green">{best_mpg:.1f}</div>
        <div class="metric-unit">miles per gallon</div>
        <span class="{badge_cl}">{badge_tx}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Comparison table
    feat_meta = [
        ("Cylinders",     orig[0], best_x[0], cyl_r[0],  cyl_r[1],  ".0f"),
        ("Displacement",  orig[1], best_x[1], disp_r[0], disp_r[1], ".1f"),
        ("Horsepower",    orig[2], best_x[2], hp_r[0],   hp_r[1],   ".1f"),
        ("Weight",        orig[3], best_x[3], wt_r[0],   wt_r[1],   ".0f"),
        ("Acceleration",  orig[4], best_x[4], acc_r[0],  acc_r[1],  ".1f"),
        ("Model Year",    orig[5], best_x[5], yr_r[0],   yr_r[1],   ".0f"),
        ("Origin",        orig[6], best_x[6], org_r[0],  org_r[1],  ".0f"),
    ]

    def fmt_val(val, i, fmt):
        if i == 5:
            return f"19{int(val):02d}"
        if i == 6:
            return ORIGIN_SHORT.get(int(val), str(int(val)))
        return f"{val:{fmt}}"

    rows_html = ""
    for i, (name, o, op, lo, hi, fmt) in enumerate(feat_meta):
        rng  = max(float(hi) - float(lo), 1e-9)
        w_o  = max(0, min(100, int((float(o)  - float(lo)) / rng * 100)))
        w_op = max(0, min(100, int((float(op) - float(lo)) / rng * 100)))
        o_d  = fmt_val(o,  i, fmt)
        op_d = fmt_val(op, i, fmt)
        delta = float(op) - float(o)
        if abs(delta) < 0.05:
            chg = '<span class="chg-same">—</span>'
        elif delta > 0:
            chg = f'<span class="chg-up">▲ +{delta:.1f}</span>'
        else:
            chg = f'<span class="chg-down">▼ {delta:.1f}</span>'

        rows_html += f"""
        <tr>
          <td class="feat-name">{name}</td>
          <td class="feat-val">{o_d}
            <div class="bar-track"><div class="bar-fill-curr" style="width:{w_o}%"></div></div>
          </td>
          <td class="feat-val">{op_d}
            <div class="bar-track"><div class="bar-fill-opt" style="width:{w_op}%"></div></div>
          </td>
          <td>{chg}</td>
        </tr>"""

    st.markdown(f"""
    <div class="cmp-wrap">
      <table class="cmp-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>Your Value</th>
            <th>Optimal Value</th>
            <th>Change</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── EXPORTS ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-head" style="margin-top:1.5rem;">
      <div class="sec-head-icon">💾</div>
      <span class="sec-head-text">Export Results</span>
    </div>
    """, unsafe_allow_html=True)

    ex1, ex2 = st.columns(2, gap="medium")

    # CSV
    with ex1:
        feat_names = ["Cylinders","Displacement","Horsepower",
                      "Weight","Acceleration","Model Year","Origin"]
        csv_buf = io.StringIO()
        writer  = csv.writer(csv_buf)
        writer.writerow(["Metric", "Your Config", "Optimal Config", "Change"])
        writer.writerow(["MPG", f"{pred_mpg:.2f}", f"{best_mpg:.2f}", f"{gain:+.2f}"])
        writer.writerow([])
        writer.writerow(["Feature", "Your Value", "Optimal Value", "Delta"])
        for i, (name, o, op, lo, hi, fmt) in enumerate(feat_meta):
            writer.writerow([name, fmt_val(o, i, fmt), fmt_val(op, i, fmt),
                             f"{float(op)-float(o):+.2f}"])
        writer.writerow([])
        writer.writerow([f"Generated: {ts}"])
        writer.writerow(["Model: Ensemble Regressor | Optimiser: gp_minimize | Dataset: UCI Auto-MPG"])
        csv_bytes = csv_buf.getvalue().encode()

        st.download_button(
            label     = "⬇️  Download CSV Report",
            data      = csv_bytes,
            file_name = f"autompg_results_{ts.replace(' ','_').replace(':','-')}.csv",
            mime      = "text/csv",
            use_container_width=True,
        )

    # PDF (via ReportLab if available, otherwise fallback to plain-text PDF)
    with ex2:
        def build_pdf() -> bytes:
            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.lib import colors
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import cm
                from reportlab.platypus import (
                    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
                )

                buf = io.BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=A4,
                                        leftMargin=2*cm, rightMargin=2*cm,
                                        topMargin=2*cm, bottomMargin=2*cm)
                styles = getSampleStyleSheet()
                story  = []

                accent_hex = "#16a34a"
                accent_rl  = colors.HexColor(accent_hex)

                h1 = ParagraphStyle("h1", parent=styles["Heading1"],
                                    fontSize=18, textColor=colors.HexColor("#0f172a"),
                                    spaceAfter=4)
                h2 = ParagraphStyle("h2", parent=styles["Heading2"],
                                    fontSize=10, textColor=accent_rl,
                                    spaceBefore=14, spaceAfter=4,
                                    fontName="Helvetica-Bold")
                body = ParagraphStyle("body", parent=styles["Normal"],
                                      fontSize=9, textColor=colors.HexColor("#334155"),
                                      leading=14)
                mono = ParagraphStyle("mono", parent=styles["Code"],
                                      fontSize=8.5, textColor=colors.HexColor("#1e293b"),
                                      leading=13)

                story.append(Paragraph("AutoMPG Fuel Intelligence", h1))
                story.append(Paragraph(
                    "Department of Computer Engineering · Federal Polytechnic Nekede, Owerri", body))
                story.append(HRFlowable(width="100%", thickness=2,
                                         color=accent_rl, spaceAfter=12))
                story.append(Paragraph(f"Report generated: {ts}", body))
                story.append(Spacer(1, 10))

                story.append(Paragraph("MPG Summary", h2))
                summ_data = [
                    ["Metric", "Your Config", "Optimal Config", "Gain"],
                    ["MPG (miles/gallon)",
                     f"{pred_mpg:.2f}",
                     f"{best_mpg:.2f}",
                     f"{gain:+.2f} ({pct:+.1f}%)"],
                ]
                summ_tbl = Table(summ_data, colWidths=[5.5*cm, 3.5*cm, 3.5*cm, 4*cm])
                summ_tbl.setStyle(TableStyle([
                    ("BACKGROUND",  (0, 0), (-1, 0), accent_rl),
                    ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
                    ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE",    (0, 0), (-1, 0), 9),
                    ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE",    (0, 1), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.HexColor("#f8fafc"), colors.white]),
                    ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
                    ("TOPPADDING",  (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(summ_tbl)
                story.append(Spacer(1, 8))

                story.append(Paragraph("Feature Comparison", h2))
                feat_data = [["Feature", "Your Value", "Optimal Value", "Delta"]]
                for i, (name, o, op, lo, hi, fmt) in enumerate(feat_meta):
                    delta = float(op) - float(o)
                    feat_data.append([
                        name,
                        fmt_val(o,  i, fmt),
                        fmt_val(op, i, fmt),
                        f"{delta:+.2f}",
                    ])
                feat_tbl = Table(feat_data, colWidths=[5*cm, 3.5*cm, 3.5*cm, 4.5*cm])
                feat_tbl.setStyle(TableStyle([
                    ("BACKGROUND",  (0, 0), (-1, 0), accent_rl),
                    ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
                    ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE",    (0, 0), (-1, 0), 9),
                    ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE",    (0, 1), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.HexColor("#f8fafc"), colors.white]),
                    ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
                    ("TOPPADDING",  (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(feat_tbl)
                story.append(Spacer(1, 14))

                story.append(HRFlowable(width="100%", thickness=0.5,
                                         color=colors.HexColor("#cbd5e1"), spaceBefore=4))
                story.append(Paragraph(
                    "Model: Ensemble Regressor · Optimiser: Gaussian Process (gp_minimize, 80 calls) "
                    "· Dataset: UCI Auto-MPG", body))

                doc.build(story)
                return buf.getvalue()

            except ImportError:
                # Minimal fallback: plain-text PDF via basic bytes
                lines = [
                    "%PDF-1.4",
                    "% AutoMPG Report (install reportlab for full PDF)",
                    f"% Generated: {ts}",
                    f"% Your MPG: {pred_mpg:.2f}",
                    f"% Optimal MPG: {best_mpg:.2f}",
                    f"% Gain: {gain:+.2f} ({pct:+.1f}%)",
                ]
                return "\n".join(lines).encode()

        pdf_bytes = build_pdf()
        st.download_button(
            label     = "⬇️  Download PDF Report",
            data      = pdf_bytes,
            file_name = f"autompg_report_{ts.replace(' ','_').replace(':','-')}.pdf",
            mime      = "application/pdf",
            use_container_width=True,
        )

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  Model: Ensemble Regressor &nbsp;·&nbsp;
  Optimiser: Gaussian Process (gp_minimize · 80 calls) &nbsp;·&nbsp;
  Dataset: UCI Auto-MPG &nbsp;·&nbsp;
  Federal Polytechnic Nekede, Owerri
</div>
""", unsafe_allow_html=True)