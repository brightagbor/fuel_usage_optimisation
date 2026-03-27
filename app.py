import streamlit as st
import numpy as np
import joblib
from skopt import gp_minimize

# Page Config 
st.set_page_config(
    page_title="AutoMPG · Fuel Intelligence",
    page_icon="⛽",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=IBM+Plex+Sans:wght@300;400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink:    #1a1a18;
    --mid:    #6b6b60;
    --faint:  #c8c8b8;
    --paper:  #f5f3ee;
    --cream:  #ede9e0;
    --green:  #2d6a4f;
    --lime:   #74c69d;
    --red:    #c1440e;
    --border: #ddd9cc;
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--paper) !important;
    color: var(--ink) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
footer { display: none !important; }

[data-testid="block-container"] {
    max-width: 720px !important;
    padding: 3rem 2rem 5rem !important;
    margin: 0 auto !important;
}

/* ── Masthead ── */
.masthead {
    border-top: 4px solid var(--ink);
    border-bottom: 1px solid var(--ink);
    padding: 1.5rem 0 1.2rem;
    margin-bottom: 2.5rem;
    text-align: center;
}
.masthead-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--mid);
    margin-bottom: 0.5rem;
}
.masthead-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 6vw, 4.5rem);
    font-weight: 900;
    line-height: 1;
    color: var(--ink);
    letter-spacing: -0.01em;
}
.masthead-title span { color: var(--green); }
.masthead-sub {
    font-size: 0.72rem;
    color: var(--mid);
    margin-top: 0.65rem;
    letter-spacing: 0.05em;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Section headings ── */
.sec-head {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--mid);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin: 0 0 1.25rem;
}

/* ── Input labels & widgets ── */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--mid) !important;
    font-weight: 500 !important;
}
[data-testid="stNumberInput"] input {
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--ink) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 2px rgba(45,106,79,0.12) !important;
    outline: none !important;
}
[data-testid="stSelectbox"] > div > div {
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--ink) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] [role="slider"] {
    background: var(--green) !important;
    border: 2px solid white !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2) !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    background: var(--ink) !important;
    color: white !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.6rem !important;
    border-radius: 3px !important;
}

/* ── Bounds panel ── */
.bounds-panel {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.75rem;
}

/* ── THE Button ── */
.stButton > button {
    width: 100%;
    padding: 1rem !important;
    background: var(--green) !important;
    border: none !important;
    border-radius: 4px !important;
    color: white !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.15s !important;
    box-shadow: 0 2px 8px rgba(45,106,79,0.25) !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    background: #1f4d38 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Results ── */
.results-wrap {
    margin-top: 2.5rem;
    border-top: 4px solid var(--ink);
    padding-top: 1.75rem;
    animation: fadeUp 0.4s ease both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.results-headline {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 1.5rem;
}

/* ── Two stat boxes ── */
.stat-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.stat-box {
    flex: 1;
    padding: 1.5rem;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: white;
    text-align: center;
}
.stat-box.highlight { border-color: var(--green); background: #f0f7f4; }
.stat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--mid);
    margin-bottom: 0.4rem;
}
.stat-number {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1;
    color: var(--ink);
}
.stat-number.green { color: var(--green); }
.stat-unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--mid);
    margin-top: 0.3rem;
}
.stat-badge {
    display: inline-block;
    background: var(--green);
    color: white;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.06em;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    margin-top: 0.6rem;
}
.stat-badge.neg { background: var(--red); }

/* ── Comparison table ── */
.cmp-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; margin-top: 0.5rem; }
.cmp-table thead tr { border-bottom: 2px solid var(--ink); }
.cmp-table th {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--mid);
    padding: 0.5rem 0.5rem 0.6rem;
    text-align: left;
    font-weight: 500;
}
.cmp-table td {
    padding: 0.6rem 0.5rem;
    border-bottom: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace;
    color: var(--ink);
    vertical-align: middle;
}
.cmp-table tr:last-child td { border-bottom: none; }
.cmp-table tbody tr:hover td { background: #f8f7f2; }
.feat-lbl { color: var(--mid); font-size: 0.7rem; }
.bar-track { background: var(--border); border-radius: 2px; height: 4px; margin-top: 4px; width: 100%; }
.bar-curr  { height: 4px; border-radius: 2px; background: var(--faint); }
.bar-opt   { height: 4px; border-radius: 2px; background: var(--green); }
.change-up   { color: var(--green); font-size: 0.7rem; font-weight: 500; }
.change-down { color: var(--red);   font-size: 0.7rem; font-weight: 500; }
.change-same { color: var(--mid);   font-size: 0.7rem; }

/* ── Progress ── */
.stProgress > div > div > div > div {
    background: var(--green) !important;
    border-radius: 2px !important;
}

.rule { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }

.footer {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    color: var(--faint);
    text-align: center;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# Load Model & Scaler
@st.cache_resource
def load_artifacts():
    import os, pickle
    from sklearn.pipeline import Pipeline

    model = joblib.load("results/ensemble1_best_model.pkl")

    # If the model is already a Pipeline with preprocessing, use it directly
    if hasattr(model, "named_steps") or hasattr(model, "steps"):
        return model, None

    # Try loading a companion scaler (common patterns)
    scaler = None
    scaler_paths = [
        "results/scaler.pkl",
    ]
    for path in scaler_paths:
        if os.path.exists(path):
            try:
                scaler = joblib.load(path)
                break
            except Exception:
                pass

    # Also try pickle
    if scaler is None:
        for path in [p.replace(".pkl","_scaler.pkl") for p in scaler_paths]:
            if os.path.exists(path):
                try:
                    with open(path,"rb") as f:
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

def safe_predict(X_raw):
    """Predict MPG, applying scaler if one was found."""
    from sklearn.pipeline import Pipeline
    X = np.array(X_raw, dtype=float).reshape(1, -1)
    # Pipeline handles its own preprocessing
    if hasattr(model, "steps") or hasattr(model, "named_steps"):
        return float(model.predict(X)[0])
    # External scaler
    if scaler is not None:
        X = scaler.transform(X)
    return float(model.predict(X)[0])

# Masthead # 
st.markdown("""
<div class="masthead">
    <div class="masthead-eyebrow">Bayesian Optimisation · Ensemble ML</div>
    <div class="masthead-title">Auto<span>MPG</span></div>
    <div class="masthead-sub">Fuel efficiency prediction & optimal design finder</div>
</div>
""", unsafe_allow_html=True)

if not model_ok:
    st.error(f"Cannot load model: `{load_err}`")
    st.stop()

# Debug: show model type in sidebar (remove after confirming fix) # 
with st.sidebar:
    st.caption(f"Model type: `{type(model).__name__}`")
    if scaler is not None:
        st.caption(f"Scaler: `{type(scaler).__name__}`")
    else:
        st.caption("No companion scaler found")
    # Sanity probe
    try:
        p_low  = safe_predict([4, 90.0,  65.0, 1800.0, 20.0, 82, 3])
        p_high = safe_predict([8, 400.0, 200.0, 5000.0, 10.0, 70, 1])
        st.caption(f"Probe low MPG: `{p_low:.2f}`")
        st.caption(f"Probe high MPG: `{p_high:.2f}`")
        if abs(p_low - p_high) < 0.5:
            st.warning("⚠️ Model outputs near-identical values for very different inputs — scaler may be missing from results/ folder.")
    except Exception as ex:
        st.caption(f"Probe error: {ex}")

ORIGIN_MAP = {1: "USA", 2: "Europe", 3: "Japan"}

# Car Specifications 
st.markdown('<p class="sec-head">Car Specifications</p>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    cylinders    = st.selectbox("Cylinders", [3, 4, 5, 6, 8])
    horsepower   = st.number_input("Horsepower (hp)", 40.0, 250.0, 100.0, step=1.0)
    acceleration = st.number_input("Acceleration (0–60s)", 8.0, 25.0, 15.0, step=0.5)
    origin       = st.selectbox("Origin", [1, 2, 3], format_func=lambda v: ORIGIN_MAP[v])
with c2:
    displacement = st.number_input("Displacement (cu·in)", 50.0, 500.0, 200.0, step=1.0)
    weight       = st.number_input("Weight (lbs)", 1500.0, 5500.0, 3000.0, step=10.0)
    model_year   = st.selectbox("Model Year", list(range(70, 83)), format_func=lambda y: f"19{y:02d}")

input_arr = np.array([[cylinders, displacement, horsepower, weight,
                        acceleration, model_year, origin]])

# Optimisation Bounds
st.markdown('<hr class="rule">', unsafe_allow_html=True)
st.markdown('<p class="sec-head">Optimisation Search Bounds</p>', unsafe_allow_html=True)

st.markdown('<div class="bounds-panel">', unsafe_allow_html=True)
b1, b2 = st.columns(2)
with b1:
    cyl_r  = st.slider("Cylinders",       3,    8,    (3, 8))
    disp_r = st.slider("Displacement",    50.0, 500.0, (68.0, 455.0))
    hp_r   = st.slider("Horsepower",      40.0, 250.0, (46.0, 230.0))
    wt_r   = st.slider("Weight",          1500.0, 5500.0, (1600.0, 5000.0))
with b2:
    acc_r  = st.slider("Acceleration",    8.0,  25.0, (8.0, 25.0))
    yr_r   = st.slider("Model Year",      70,   82,   (70, 82))
    org_r  = st.slider("Origin",          1,    3,    (1, 3))
st.markdown('</div>', unsafe_allow_html=True)

# Single Action Button 
go = st.button("⛽  Predict MPG & Find Optimal Configuration")

if go:
    # Step 1: Predict current config 
    pred_mpg = safe_predict(input_arr[0])

    # Step 2: Optimisation  
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
        raw = [int(round(cyl)), float(disp), float(hp),
               float(wt), float(acc),
               int(round(year)), max(1, min(3, int(round(org))))]
        return safe_predict(raw)

    def objective(x):
        return -predict_mpg_from_vec(x)

    prog_slot = st.empty()
    pbar      = prog_slot.progress(0, text="Starting optimisation…")
    counter   = [0]

    def cb(res):
        counter[0] += 1
        best_now = -min(res.func_vals)
        pbar.progress(
            min(counter[0] / N_CALLS, 1.0),
            text=f"Evaluation {counter[0]}/{N_CALLS}  ·  best so far: {best_now:.2f} MPG"
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

    prog_slot.empty()

    best_x    = list(result.x)
    best_x[0] = int(round(best_x[0]))
    best_x[5] = int(round(best_x[5]))
    best_x[6] = max(1, min(3, int(round(best_x[6]))))

    # Re-predict with rounded integers for exact match
    best_mpg = predict_mpg_from_vec(best_x)

    st.session_state["pred_mpg"]  = pred_mpg
    st.session_state["best_mpg"]  = best_mpg
    st.session_state["best_x"]    = best_x
    st.session_state["input_arr"] = input_arr.copy()
    st.session_state["bounds"]    = (cyl_r, disp_r, hp_r, wt_r, acc_r, yr_r, org_r)

# Show Results  # ───
if "pred_mpg" in st.session_state:
    pred_mpg = st.session_state["pred_mpg"]
    best_mpg = st.session_state["best_mpg"]
    best_x   = st.session_state["best_x"]
    orig     = st.session_state["input_arr"][0]
    cyl_r, disp_r, hp_r, wt_r, acc_r, yr_r, org_r = st.session_state["bounds"]

    gain = best_mpg - pred_mpg
    pct  = (gain / pred_mpg * 100) if pred_mpg else 0
    badge_cls = "stat-badge" if gain >= 0 else "stat-badge neg"
    badge_txt = (f"▲ +{gain:.2f} MPG  ({pct:+.1f}%)" if gain >= 0
                 else f"▼ {gain:.2f} MPG  ({pct:.1f}%)")

    st.markdown('<div class="results-wrap">', unsafe_allow_html=True)
    st.markdown('<p class="results-headline">Analysis Results</p>', unsafe_allow_html=True)

    # Stat cards
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-box">
        <div class="stat-label">Your Configuration</div>
        <div class="stat-number">{pred_mpg:.1f}</div>
        <div class="stat-unit">miles per gallon</div>
      </div>
      <div class="stat-box highlight">
        <div class="stat-label">Optimal Configuration</div>
        <div class="stat-number green">{best_mpg:.1f}</div>
        <div class="stat-unit">miles per gallon</div>
        <span class="{badge_cls}">{badge_txt}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature comparison table
    feat_meta = [
        ("Cylinders",    cyl_r[0],  cyl_r[1],  ".0f"),
        ("Displacement", disp_r[0], disp_r[1], ".1f"),
        ("Horsepower",   hp_r[0],   hp_r[1],   ".1f"),
        ("Weight",       wt_r[0],   wt_r[1],   ".0f"),
        ("Acceleration", acc_r[0],  acc_r[1],  ".1f"),
        ("Model Year",   yr_r[0],   yr_r[1],   ".0f"),
        ("Origin",       org_r[0],  org_r[1],  ".0f"),
    ]

    rows = ""
    for i, (name, lo, hi, fmt) in enumerate(feat_meta):
        o   = float(orig[i])
        op  = float(best_x[i])
        rng = max(float(hi) - float(lo), 1e-9)
        w_o  = max(0, min(100, int((o  - float(lo)) / rng * 100)))
        w_op = max(0, min(100, int((op - float(lo)) / rng * 100)))

        if i == 5:
            o_disp  = f"19{int(o):02d}"
            op_disp = f"19{int(op):02d}"
        elif i == 6:
            o_disp  = ORIGIN_MAP.get(int(o),  str(int(o)))
            op_disp = ORIGIN_MAP.get(int(op), str(int(op)))
        else:
            o_disp  = f"{o:{fmt}}"
            op_disp = f"{op:{fmt}}"

        delta = op - o
        if abs(delta) < 0.05:
            chg = '<span class="change-same">—</span>'
        elif delta > 0:
            chg = f'<span class="change-up">▲ +{delta:.1f}</span>'
        else:
            chg = f'<span class="change-down">▼ {delta:.1f}</span>'

        rows += f"""
        <tr>
          <td class="feat-lbl">{name}</td>
          <td>{o_disp}
            <div class="bar-track"><div class="bar-curr" style="width:{w_o}%"></div></div>
          </td>
          <td>{op_disp}
            <div class="bar-track"><div class="bar-opt" style="width:{w_op}%"></div></div>
          </td>
          <td>{chg}</td>
        </tr>"""

    st.markdown(f"""
    <p class="sec-head" style="margin-top:0">Feature Comparison</p>
    <table class="cmp-table">
      <thead>
        <tr>
          <th>Feature</th>
          <th>Your Value</th>
          <th>Optimal Value</th>
          <th>Change</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer # ──
st.markdown("""
<div class="footer">
    Model: Ensemble Regressor &nbsp;·&nbsp;
    Optimiser: Gaussian Process (gp_minimize · 40 calls) &nbsp;·&nbsp;
    Dataset: UCI Auto-MPG
</div>
""", unsafe_allow_html=True)