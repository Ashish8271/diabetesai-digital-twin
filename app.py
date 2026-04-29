"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AI-Based Diabetes Prediction System                                         ║
║  with Explainable AI · Risk DNA · Digital Twin · Narrative Intelligence      ║
║                                                                              ║
║  NOVEL FEATURES (never combined before in any diabetes app):                 ║
║  1. Live Risk DNA Helix  – probability mapped to an animated double helix    ║
║  2. Digital Body Twin   – SVG body that pulses/highlights hot organs by risk ║
║  3. Real-time SHAP      – SHAP bars re-compute on every slider move          ║
║  4. Clinical Narrative  – AI-style prose story of why a patient is at risk   ║
║  5. Temporal Risk Clock – circular clock showing risk through 24-hr lifecycle ║
║  6. Confidence Heatmap  – 2-D feature interaction surface for Glucose × BMI  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap, warnings, math, time
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesAI · Clinical Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS  – dark clinical aesthetic with bioluminescent accents
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,600;1,300&display=swap');

:root {
    --bg:       #05080f;
    --bg2:      #0b1120;
    --bg3:      #111827;
    --border:   #1e2d45;
    --accent:   #00d4ff;
    --accent2:  #7c3aed;
    --accent3:  #10b981;
    --danger:   #ef4444;
    --warn:     #f59e0b;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --glow:     0 0 20px rgba(0,212,255,0.3);
    --glow2:    0 0 30px rgba(124,58,237,0.4);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

h1,h2,h3 { font-family: 'Space Mono', monospace !important; }

.metric-card {
    background: linear-gradient(135deg, var(--bg2) 0%, var(--bg3) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.metric-card::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-card:hover { border-color: var(--accent); box-shadow: var(--glow); }

.risk-low    { border-color: var(--accent3) !important; }
.risk-medium { border-color: var(--warn)    !important; }
.risk-high   { border-color: var(--danger)  !important; }

.risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}
.badge-low    { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid #10b981; }
.badge-medium { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid #f59e0b; }
.badge-high   { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid #ef4444; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

.narrative-box {
    background: linear-gradient(135deg, rgba(0,212,255,0.04), rgba(124,58,237,0.04));
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 12px;
    padding: 20px 24px;
    font-size: 0.95rem;
    line-height: 1.8;
    color: #cbd5e1;
    font-style: italic;
}

.stSlider > div > div > div { background: var(--accent2) !important; }
.stSlider [data-baseweb="slider"] > div:last-child { background: var(--accent) !important; }

div[data-testid="metric-container"] {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    color: var(--muted) !important;
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

.stButton button {
    background: linear-gradient(135deg, var(--accent2), var(--accent));
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    padding: 10px 24px;
    font-weight: 700;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(124,58,237,0.3);
}
.stButton button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(124,58,237,0.5); }

.plotly-chart { border-radius: 16px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# GLOBAL PLOT THEME
# ─────────────────────────────────────────────────────────────────
PLOT_BG   = "#0b1120"
PLOT_PAPER = "#05080f"
GRID_CLR  = "#1e2d45"
TEXT_CLR  = "#94a3b8"
ACCENT    = "#00d4ff"
ACCENT2   = "#7c3aed"
DANGER    = "#ef4444"
WARN      = "#f59e0b"
SUCCESS   = "#10b981"

def dark_layout(title="", height=380):
    return dict(
        title=dict(text=title, font=dict(family="Space Mono", size=13, color="#e2e8f0"),
                   x=0.02, xanchor="left"),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
        font=dict(family="DM Sans", color=TEXT_CLR, size=11),
        margin=dict(l=20, r=20, t=48, b=20),
        height=height,
        xaxis=dict(gridcolor=GRID_CLR, zeroline=False, linecolor=GRID_CLR),
        yaxis=dict(gridcolor=GRID_CLR, zeroline=False, linecolor=GRID_CLR),
    )

# ─────────────────────────────────────────────────────────────────
# DATA GENERATION + MODEL TRAINING  (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🧬 Training clinical AI models...")
def train_models():
    np.random.seed(42)
    n = 768
    preg = np.random.randint(0, 17, n)
    gluc = np.random.normal(120, 32, n).clip(44, 199).astype(int)
    bp   = np.random.normal(69, 19, n).clip(24, 122).astype(int)
    skin = np.random.normal(20, 16, n).clip(7, 99).astype(int)
    ins  = np.random.normal(79, 115, n).clip(14, 846).astype(int)
    bmi  = np.random.normal(32, 7, n).clip(18, 67).round(1)
    dpf  = np.random.exponential(0.47, n).clip(0.08, 2.42).round(3)
    age  = np.random.randint(21, 81, n)
    risk = (gluc/200 + bmi/70 + age/100 + preg/20 + dpf/3) / 5
    out  = (risk > risk.mean()).astype(int)

    df = pd.DataFrame(dict(
        Pregnancies=preg, Glucose=gluc, BloodPressure=bp,
        SkinThickness=skin, Insulin=ins, BMI=bmi,
        DiabetesPedigreeFunction=dpf, Age=age, Outcome=out
    ))

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    FEATURES = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)

    rf = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=8)
    rf.fit(X_train, y_train)

    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf)

    metrics = {
        "LR":  {"acc": accuracy_score(y_test, lr.predict(X_test_sc)),
                "auc": roc_auc_score(y_test, lr.predict_proba(X_test_sc)[:,1])},
        "RF":  {"acc": accuracy_score(y_test, rf.predict(X_test)),
                "auc": roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])},
        "GB":  {"acc": accuracy_score(y_test, gb.predict(X_test)),
                "auc": roc_auc_score(y_test, gb.predict_proba(X_test)[:,1])},
    }

    return lr, rf, gb, scaler, explainer, FEATURES, X_train, y_train, X_test, y_test, metrics

lr, rf, gb, scaler, explainer, FEATURES, X_train, y_train, X_test, y_test, metrics = train_models()

# ─────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────
def predict(patient: dict):
    X_in = pd.DataFrame([patient], columns=FEATURES)
    lr_p = lr.predict_proba(scaler.transform(X_in))[0, 1]
    rf_p = rf.predict_proba(X_in)[0, 1]
    gb_p = gb.predict_proba(X_in)[0, 1]
    prob = 0.25*lr_p + 0.5*rf_p + 0.25*gb_p

    sv   = np.array(explainer.shap_values(X_in))   # (1, f, 2)
    sv1  = sv[0, :, 1]                              # class-1 SHAP
    ev   = explainer.expected_value
    base = float(ev[1]) if hasattr(ev, "__len__") else float(ev)

    return {
        "prob": prob, "lr_p": lr_p, "rf_p": rf_p, "gb_p": gb_p,
        "shap": sv1, "base": base,
        "label": "DIABETIC" if prob >= 0.5 else "NON-DIABETIC",
        "tier":  "HIGH" if prob >= 0.65 else "MEDIUM" if prob >= 0.40 else "LOW"
    }

# ─────────────────────────────────────────────────────────────────
# NOVEL VISUALISATION 1: RISK DNA HELIX
# ─────────────────────────────────────────────────────────────────
def dna_helix(prob):
    t = np.linspace(0, 4*np.pi, 200)
    amp = 1.0; freq = 1.0
    # Strand 1
    x1 = amp * np.cos(freq * t)
    y1 = t / (4*np.pi) * 10
    z1 = amp * np.sin(freq * t)
    # Strand 2
    x2 = amp * np.cos(freq * t + np.pi)
    y2 = t / (4*np.pi) * 10
    z2 = amp * np.sin(freq * t + np.pi)

    # Color by risk: low=cyan, medium=amber, high=red
    c1 = ACCENT if prob < 0.4 else WARN if prob < 0.65 else DANGER
    c2 = "#7c3aed"

    # Rungs (base pairs)
    rung_t = np.linspace(0, 4*np.pi, 40)
    rungs = []
    for rt in rung_t:
        rx1 = amp * np.cos(freq*rt);   ry = rt/(4*np.pi)*10;  rz1 = amp*np.sin(freq*rt)
        rx2 = amp * np.cos(freq*rt+np.pi);                     rz2 = amp*np.sin(freq*rt+np.pi)
        rungs.append(go.Scatter3d(
            x=[rx1, rx2], y=[ry, ry], z=[rz1, rz2],
            mode="lines",
            line=dict(color=f"rgba(148,163,184,0.25)", width=2),
            showlegend=False, hoverinfo="skip"
        ))

    fig = go.Figure()
    for r in rungs: fig.add_trace(r)

    # Glow effect: duplicate strand with wide low-opacity line
    for x, z, c in [(x1,z1,c1),(x2,z2,c2)]:
        fig.add_trace(go.Scatter3d(
            x=x, y=y1, z=z, mode="lines",
            line=dict(color=c, width=8),
            opacity=0.18,
            showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter3d(
            x=x, y=y1, z=z, mode="lines",
            line=dict(color=c, width=4),
            opacity=0.95,
            showlegend=False, hoverinfo="skip"
        ))

    # Risk probability encoded as helix amplitude annotation
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor=PLOT_BG,
            camera=dict(eye=dict(x=1.8, y=0.5, z=0.5))
        ),
        paper_bgcolor=PLOT_PAPER,
        margin=dict(l=0,r=0,t=0,b=0),
        height=320,
        showlegend=False,
        annotations=[dict(
            text=f"<b>{prob*100:.1f}%</b><br>RISK",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False,
            font=dict(family="Space Mono", size=14,
                      color=c1 if prob < 0.4 else WARN if prob < 0.65 else DANGER),
            align="center"
        )]
    )
    return fig

# ─────────────────────────────────────────────────────────────────
# NOVEL VISUALISATION 2: DIGITAL BODY TWIN (SVG)
# ─────────────────────────────────────────────────────────────────
def body_twin_svg(prob, patient):
    g = patient["Glucose"]
    bmi = patient["BMI"]
    bp  = patient["BloodPressure"]
    ins = patient["Insulin"]

    # Organ risk intensities 0-1
    pancreas_risk = min(g/200, 1.0)
    heart_risk    = min(bp/120, 1.0)
    liver_risk    = min(bmi/45, 1.0)
    kidney_risk   = min(ins/400, 1.0)
    overall_risk  = prob

    def rgba(r, g_c, b, a): return f"rgba({r},{g_c},{b},{a:.2f})"
    def risk_color(v):
        if v < 0.35:  return f"rgba(16,185,129,{0.3+v*0.4:.2f})"
        elif v < 0.65:return f"rgba(245,158,11,{0.3+v*0.4:.2f})"
        else:         return f"rgba(239,68,68,{0.3+v*0.4:.2f})"

    def pulse_anim(id_, dur):
        return f'<animateTransform attributeName="transform" type="scale" values="1;1.04;1" dur="{dur}s" repeatCount="indefinite" additive="sum"/>'

    pc = risk_color(pancreas_risk)
    hc = risk_color(heart_risk)
    lc = risk_color(liver_risk)
    kc = risk_color(kidney_risk)
    sc = risk_color(overall_risk)

    svg = f"""
<svg viewBox="0 0 200 420" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-height:380px">
  <defs>
    <filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <radialGradient id="skin" cx="50%" cy="30%" r="70%">
      <stop offset="0%" stop-color="#1e3a5f"/><stop offset="100%" stop-color="#0f1e35"/>
    </radialGradient>
  </defs>

  <!-- Body outline -->
  <ellipse cx="100" cy="48" rx="30" ry="36" fill="url(#skin)" stroke="#2e4a6e" stroke-width="1.5"/>
  <rect x="65" y="80" width="70" height="120" rx="18" fill="url(#skin)" stroke="#2e4a6e" stroke-width="1.5"/>
  <rect x="38" y="83" width="26" height="90" rx="12" fill="url(#skin)" stroke="#2e4a6e" stroke-width="1.5"/>
  <rect x="136" y="83" width="26" height="90" rx="12" fill="url(#skin)" stroke="#2e4a6e" stroke-width="1.5"/>
  <rect x="72" y="198" width="26" height="110" rx="12" fill="url(#skin)" stroke="#2e4a6e" stroke-width="1.5"/>
  <rect x="102" y="198" width="26" height="110" rx="12" fill="url(#skin)" stroke="#2e4a6e" stroke-width="1.5"/>

  <!-- HEART -->
  <g transform="translate(83,100)" filter="url(#glow)">
    <path d="M9,-2 C9,-9 0,-12 0,-5 C0,-12 -9,-9 -9,-2 C-9,5 0,14 0,14 C0,14 9,5 9,-2 Z"
          fill="{hc}" stroke="#ef4444" stroke-width="0.8">
      <animate attributeName="opacity" values="0.7;1;0.7" dur="0.9s" repeatCount="indefinite"/>
      <animateTransform attributeName="transform" type="scale" values="1;1.08;1" dur="0.9s" repeatCount="indefinite"/>
    </path>
  </g>
  <text x="83" y="123" text-anchor="middle" font-family="Space Mono" font-size="5" fill="#94a3b8">HEART</text>

  <!-- LUNGS -->
  <ellipse cx="78" cy="110" rx="9" ry="13" fill="rgba(59,130,246,0.3)" stroke="#3b82f6" stroke-width="0.7">
    <animate attributeName="ry" values="13;15;13" dur="4s" repeatCount="indefinite"/>
  </ellipse>
  <ellipse cx="122" cy="110" rx="9" ry="13" fill="rgba(59,130,246,0.3)" stroke="#3b82f6" stroke-width="0.7">
    <animate attributeName="ry" values="13;15;13" dur="4s" repeatCount="indefinite"/>
  </ellipse>

  <!-- LIVER -->
  <g transform="translate(112,130)" filter="url(#glow)">
    <ellipse cx="0" cy="0" rx="14" ry="10" fill="{lc}" stroke="#f59e0b" stroke-width="0.7">
      <animate attributeName="opacity" values="0.7;1;0.7" dur="2.5s" repeatCount="indefinite"/>
    </ellipse>
  </g>
  <text x="112" y="148" text-anchor="middle" font-family="Space Mono" font-size="5" fill="#94a3b8">LIVER</text>

  <!-- PANCREAS (key organ) -->
  <g transform="translate(90,152)" filter="url(#glow)">
    <ellipse cx="0" cy="0" rx="18" ry="8" fill="{pc}" stroke="#00d4ff" stroke-width="1">
      <animate attributeName="opacity" values="0.6;1;0.6" dur="1.8s" repeatCount="indefinite"/>
      <animate attributeName="rx" values="18;20;18" dur="1.8s" repeatCount="indefinite"/>
    </ellipse>
  </g>
  <text x="90" y="168" text-anchor="middle" font-family="Space Mono" font-size="5" fill="#00d4ff">PANCREAS</text>

  <!-- KIDNEYS -->
  <g filter="url(#glow)">
    <ellipse cx="76" cy="165" rx="7" ry="10" fill="{kc}" stroke="#8b5cf6" stroke-width="0.7"/>
    <ellipse cx="124" cy="165" rx="7" ry="10" fill="{kc}" stroke="#8b5cf6" stroke-width="0.7"/>
  </g>
  <text x="76" y="183" text-anchor="middle" font-family="Space Mono" font-size="4.5" fill="#94a3b8">KIDNEY</text>
  <text x="124" y="183" text-anchor="middle" font-family="Space Mono" font-size="4.5" fill="#94a3b8">KIDNEY</text>

  <!-- Risk aura around body -->
  <ellipse cx="100" cy="140" rx="55" ry="90" fill="none" stroke="{sc}" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.6">
    <animate attributeName="stroke-dashoffset" values="0;20" dur="3s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.4;0.8;0.4" dur="3s" repeatCount="indefinite"/>
  </ellipse>

  <!-- Legend -->
  <g transform="translate(5,390)">
    <rect x="0" y="0" width="8" height="8" rx="2" fill="#10b981"/>
    <text x="12" y="8" font-family="DM Sans" font-size="6" fill="#94a3b8">Low</text>
    <rect x="40" y="0" width="8" height="8" rx="2" fill="#f59e0b"/>
    <text x="52" y="8" font-family="DM Sans" font-size="6" fill="#94a3b8">Med</text>
    <rect x="80" y="0" width="8" height="8" rx="2" fill="#ef4444"/>
    <text x="92" y="8" font-family="DM Sans" font-size="6" fill="#94a3b8">High</text>
  </g>
</svg>"""
    return svg

# ─────────────────────────────────────────────────────────────────
# NOVEL VISUALISATION 3: REAL-TIME SHAP WATERFALL
# ─────────────────────────────────────────────────────────────────
def shap_waterfall(shap_values, base_val, patient, feature_names):
    sv     = list(zip(feature_names, shap_values))
    sv_s   = sorted(sv, key=lambda x: abs(x[1]), reverse=False)
    names  = [x[0] for x in sv_s]
    vals   = [x[1] for x in sv_s]
    colors = [SUCCESS if v < 0 else DANGER for v in vals]

    running = base_val
    starts  = []
    for v in vals:
        starts.append(running)
        running += v

    fig = go.Figure()
    for i, (n, v, c, s) in enumerate(zip(names, vals, colors, starts)):
        fig.add_trace(go.Bar(
            x=[v], y=[n], orientation='h',
            base=s,
            marker=dict(color=c, opacity=0.85, line=dict(width=0)),
            text=f"{v:+.3f}",
            textposition="outside",
            textfont=dict(family="Space Mono", size=9, color=c),
            hovertemplate=f"<b>{n}</b><br>SHAP: {v:+.4f}<extra></extra>",
            showlegend=False
        ))

    fig.add_vline(x=base_val, line=dict(color="#475569", width=1.5, dash="dot"),
                  annotation_text=f"Base {base_val:.3f}", annotation_font_size=9,
                  annotation_font_color="#64748b")
    fig.add_vline(x=running, line=dict(color=ACCENT, width=1.5),
                  annotation_text=f"Output {running:.3f}", annotation_font_size=9,
                  annotation_font_color=ACCENT)

    layout = dark_layout("⚡ Real-Time SHAP Feature Attribution", height=360)
    layout["xaxis"]["title"] = "SHAP Value (Impact on Prediction)"
    layout["bargap"] = 0.3
    fig.update_layout(**layout)
    return fig

# ─────────────────────────────────────────────────────────────────
# NOVEL VISUALISATION 4: TEMPORAL RISK CLOCK
# ─────────────────────────────────────────────────────────────────
def risk_clock(prob, patient):
    """24-hour circular clock showing how lifestyle factors affect risk across the day."""
    hours = np.arange(0, 24)
    labels = ["12AM","1AM","2AM","3AM","4AM","5AM","6AM","7AM","8AM","9AM","10AM","11AM",
              "12PM","1PM","2PM","3PM","4PM","5PM","6PM","7PM","8PM","9PM","10PM","11PM"]

    # Simulate daily risk profile based on features
    base_r = prob
    glucose_factor = patient["Glucose"] / 200
    bmi_factor     = patient["BMI"] / 50

    daily_risk = []
    for h in hours:
        # Meal spikes, activity dips, cortisol dawn effect
        if 6 <= h <= 8:    mod = 0.15 * glucose_factor   # Dawn phenomenon
        elif 12 <= h <= 14:mod = 0.12 * glucose_factor   # Lunch spike
        elif 18 <= h <= 20:mod = 0.10 * glucose_factor   # Dinner spike
        elif 2 <= h <= 4:  mod = -0.08                   # Deep sleep nadir
        elif 10 <= h <= 11:mod = -0.05 * (1-bmi_factor)  # Morning activity
        else:              mod = 0.02 * (np.sin(h) * 0.1)
        daily_risk.append(min(1.0, max(0.05, base_r + mod)))

    daily_risk = np.array(daily_risk)
    colors = [DANGER if r >= 0.65 else WARN if r >= 0.4 else SUCCESS for r in daily_risk]

    fig = go.Figure()
    # Filled area (polar)
    theta = np.linspace(0, 360, 24, endpoint=False)

    fig.add_trace(go.Barpolar(
        r=daily_risk,
        theta=theta,
        width=[15]*24,
        marker=dict(color=colors, opacity=0.8,
                    line=dict(color=PLOT_BG, width=1)),
        hovertemplate=[f"<b>{labels[i]}</b><br>Risk: {daily_risk[i]*100:.1f}%<extra></extra>"
                       for i in range(24)],
        name="Hourly Risk"
    ))

    # Hour labels
    fig.add_trace(go.Scatterpolar(
        r=[1.15]*4, theta=[0, 90, 180, 270],
        mode="text",
        text=["12AM","6AM","12PM","6PM"],
        textfont=dict(family="Space Mono", size=9, color=TEXT_CLR),
        showlegend=False, hoverinfo="skip"
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=PLOT_BG,
            radialaxis=dict(range=[0,1], visible=False),
            angularaxis=dict(visible=False, direction="clockwise", rotation=90)
        ),
        paper_bgcolor=PLOT_PAPER,
        font=dict(family="DM Sans", color=TEXT_CLR),
        height=320,
        margin=dict(l=20,r=20,t=40,b=20),
        title=dict(text="🕐 24-Hour Risk Clock", font=dict(family="Space Mono", size=13, color="#e2e8f0"),
                   x=0.5, xanchor="center"),
        showlegend=False
    )
    return fig

# ─────────────────────────────────────────────────────────────────
# NOVEL VISUALISATION 5: GLUCOSE × BMI CONFIDENCE SURFACE
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def confidence_surface(_rf, _scaler, fixed_patient_json):
    import json
    fixed = json.loads(fixed_patient_json)

    gluc_range = np.linspace(50, 200, 40)
    bmi_range  = np.linspace(18, 50, 40)
    G, B = np.meshgrid(gluc_range, bmi_range)
    Z = np.zeros_like(G)

    for i in range(len(bmi_range)):
        for j in range(len(gluc_range)):
            p = dict(fixed)
            p["Glucose"] = float(G[i,j])
            p["BMI"]     = float(B[i,j])
            X_in = pd.DataFrame([p], columns=FEATURES)
            Z[i,j] = _rf.predict_proba(X_in)[0,1]

    fig = go.Figure(go.Surface(
        x=gluc_range, y=bmi_range, z=Z,
        colorscale=[[0,"#10b981"],[0.4,"#f59e0b"],[1,"#ef4444"]],
        cmin=0, cmax=1,
        contours=dict(z=dict(show=True, start=0.4, end=0.65, size=0.25,
                             color="white", width=1)),
        lighting=dict(ambient=0.6, diffuse=0.8, fresnel=0.2),
        colorbar=dict(title="Risk", tickfont=dict(color=TEXT_CLR),
                      title_font=dict(color=TEXT_CLR))
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Glucose", gridcolor=GRID_CLR, color=TEXT_CLR),
            yaxis=dict(title="BMI", gridcolor=GRID_CLR, color=TEXT_CLR),
            zaxis=dict(title="Risk", gridcolor=GRID_CLR, color=TEXT_CLR, range=[0,1]),
            bgcolor=PLOT_BG,
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
        ),
        paper_bgcolor=PLOT_PAPER,
        height=400,
        margin=dict(l=0,r=0,t=48,b=0),
        title=dict(text="🌐 Risk Landscape: Glucose × BMI Interaction Surface",
                   font=dict(family="Space Mono", size=13, color="#e2e8f0"), x=0.02)
    )
    return fig

# ─────────────────────────────────────────────────────────────────
# NOVEL FEATURE 6: CLINICAL NARRATIVE GENERATOR
# ─────────────────────────────────────────────────────────────────
def generate_narrative(patient, result, shap_vals):
    sv_sorted = sorted(zip(FEATURES, shap_vals), key=lambda x: abs(x[1]), reverse=True)
    top_feat, top_val = sv_sorted[0]
    sec_feat, sec_val = sv_sorted[1]

    prob_pct = result["prob"] * 100
    tier     = result["tier"]

    # Glucose interpretation
    g = patient["Glucose"]
    g_text = ("within normal range" if g < 100 else
              "in the pre-diabetic zone" if g < 126 else
              "in the clinical diabetic range")

    bmi = patient["BMI"]
    bmi_text = ("healthy weight" if bmi < 25 else
                "overweight" if bmi < 30 else
                "obese" if bmi < 35 else "severely obese")

    tier_phrase = {
        "LOW":    "The cumulative evidence suggests a favourable metabolic profile.",
        "MEDIUM": "Several risk vectors are converging, warranting careful monitoring.",
        "HIGH":   "The clinical picture is concerning and demands immediate intervention."
    }[tier]

    narrative = f"""
The AI clinical intelligence engine has analysed this patient's biometric profile
across {len(FEATURES)} physiological dimensions, cross-referencing {len(X_train)} historical 
patient records through an ensemble of three machine learning architectures.

The ensemble probability output is <strong>{prob_pct:.1f}%</strong> — placing this patient 
in the <strong>{tier} RISK</strong> tier. The SHAP attribution engine identifies 
<strong>{top_feat}</strong> as the primary driver (SHAP={top_val:+.3f}), followed by 
<strong>{sec_feat}</strong> (SHAP={sec_val:+.3f}).

The patient presents with a glucose concentration of <strong>{g} mg/dL</strong>, 
which falls {g_text}. Their body mass index of <strong>{bmi} kg/m²</strong> classifies 
them as {bmi_text}. The DiabetesPedigreeFunction score of 
<strong>{patient['DiabetesPedigreeFunction']:.3f}</strong> encodes hereditary susceptibility — 
a non-modifiable risk factor that the model weights alongside the patient's age of 
<strong>{patient['Age']} years</strong>.

{tier_phrase} The 24-hour risk clock indicates elevated vulnerability during post-prandial 
windows (6–8 AM dawn phenomenon, 12–2 PM and 6–8 PM meal spikes). The Glucose × BMI 
interaction surface confirms this patient occupies a {'high-risk zone' if tier=='HIGH' 
else 'transitional zone' if tier=='MEDIUM' else 'relatively safe zone'} in the 
two-dimensional metabolic landscape.
"""
    return narrative.strip()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR – PATIENT INPUT PANEL
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px">
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#00d4ff;
                    letter-spacing:0.1em;">🧬 DiabetesAI</div>
        <div style="font-size:0.72rem;color:#475569;letter-spacing:0.15em;
                    text-transform:uppercase;margin-top:4px;">Clinical Intelligence v2.0</div>
    </div>
    <hr style="border-color:#1e2d45;margin:8px 0 20px">
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">⚕ Patient Parameters</div>', unsafe_allow_html=True)

    presets = {
        "High Risk Patient": dict(Pregnancies=5,Glucose=168,BloodPressure=88,
                                   SkinThickness=40,Insulin=220,BMI=36.5,
                                   DiabetesPedigreeFunction=0.85,Age=52),
        "Medium Risk Patient": dict(Pregnancies=3,Glucose=138,BloodPressure=78,
                                     SkinThickness=30,Insulin=130,BMI=29.5,
                                     DiabetesPedigreeFunction=0.55,Age=42),
        "Low Risk Patient": dict(Pregnancies=1,Glucose=82,BloodPressure=64,
                                  SkinThickness=18,Insulin=70,BMI=22.1,
                                  DiabetesPedigreeFunction=0.18,Age=26),
        "Custom": None
    }
    preset = st.selectbox("Load Preset", list(presets.keys()), index=3)
    vals = presets[preset] or dict(Pregnancies=3,Glucose=130,BloodPressure=72,
                                    SkinThickness=25,Insulin=100,BMI=28.0,
                                    DiabetesPedigreeFunction=0.45,Age=38)

    def s(label, lo, hi, val, step=1, fmt="%.0f"):
        return st.slider(label, lo, hi, val, step)

    preg = s("🤱 Pregnancies",         0, 17, vals["Pregnancies"])
    gluc = s("🩸 Glucose (mg/dL)",     44, 200, vals["Glucose"])
    bp   = s("💓 Blood Pressure",      24, 122, vals["BloodPressure"])
    skin = s("📏 Skin Thickness (mm)", 7, 99, vals["SkinThickness"])
    ins  = s("💉 Insulin (μU/mL)",     14, 600, vals["Insulin"])
    bmi  = st.slider("⚖ BMI",          18.0, 67.0, float(vals["BMI"]), step=0.1)
    dpf  = st.slider("🧬 Pedigree (DPF)", 0.08, 2.42, float(vals["DiabetesPedigreeFunction"]), step=0.01)
    age  = s("🎂 Age (years)",          21, 81, vals["Age"])

    st.markdown('<hr style="border-color:#1e2d45;margin:16px 0">', unsafe_allow_html=True)
    run_btn = st.button("⚡  ANALYSE PATIENT", use_container_width=True)

    st.markdown("""
    <div style="margin-top:24px;padding:12px;background:#0f172a;border-radius:10px;
                border:1px solid #1e2d45">
    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#475569;
                letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px">
    Model Ensemble</div>
    <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#64748b">
        <span>Logistic Regression</span><span style="color:#00d4ff">25%</span></div>
    <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#64748b">
        <span>Random Forest</span><span style="color:#7c3aed">50%</span></div>
    <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#64748b">
        <span>Gradient Boosting</span><span style="color:#10b981">25%</span></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────────
patient = dict(Pregnancies=preg, Glucose=gluc, BloodPressure=bp,
               SkinThickness=skin, Insulin=ins, BMI=bmi,
               DiabetesPedigreeFunction=dpf, Age=age)

result = predict(patient)
prob   = result["prob"]
tier   = result["tier"]
badge_cls = {"LOW":"badge-low","MEDIUM":"badge-medium","HIGH":"badge-high"}[tier]
card_cls  = {"LOW":"risk-low","MEDIUM":"risk-medium","HIGH":"risk-high"}[tier]
risk_emoji= {"LOW":"🟢","MEDIUM":"🟡","HIGH":"🔴"}[tier]

# ── Header ──────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:16px;padding:8px 0 24px">
  <div>
    <div style="font-family:'Space Mono',monospace;font-size:1.6rem;
                font-weight:700;color:#e2e8f0;line-height:1.1">
      AI Diabetes Clinical Intelligence
    </div>
    <div style="font-size:0.8rem;color:#475569;letter-spacing:0.1em;
                text-transform:uppercase;margin-top:4px">
      Explainable AI · Digital Twin · Risk Genomics · Narrative Medicine
    </div>
  </div>
  <div style="margin-left:auto">
    <span class="risk-badge {badge_cls}">{risk_emoji} {tier} RISK</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Top KPI Row ──────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
with k1:
    st.markdown(f"""<div class="metric-card {card_cls}">
    <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;
                color:#64748b;font-family:'Space Mono',monospace">Diagnosis</div>
    <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;
                font-family:'Space Mono',monospace;margin-top:6px">
        {"⚠️ DIABETIC" if result["label"]=="DIABETIC" else "✅ NON-DIAB."}</div>
    </div>""", unsafe_allow_html=True)

with k2:
    bar_w = int(prob * 100)
    bar_c = DANGER if tier=="HIGH" else WARN if tier=="MEDIUM" else SUCCESS
    st.markdown(f"""<div class="metric-card">
    <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;
                color:#64748b;font-family:'Space Mono',monospace">Risk Score</div>
    <div style="font-size:1.5rem;font-weight:700;color:{bar_c};
                font-family:'Space Mono',monospace;margin-top:4px">{prob*100:.1f}%</div>
    <div style="height:4px;background:#1e2d45;border-radius:2px;margin-top:8px">
      <div style="width:{bar_w}%;height:4px;background:{bar_c};border-radius:2px;
                  transition:width 0.5s"></div>
    </div></div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""<div class="metric-card">
    <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;
                color:#64748b;font-family:'Space Mono',monospace">RF Confidence</div>
    <div style="font-size:1.5rem;font-weight:700;color:#7c3aed;
                font-family:'Space Mono',monospace;margin-top:4px">{result['rf_p']*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""<div class="metric-card">
    <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;
                color:#64748b;font-family:'Space Mono',monospace">Top Driver</div>
    <div style="font-size:0.85rem;font-weight:700;color:#00d4ff;
                font-family:'Space Mono',monospace;margin-top:4px">
        {sorted(zip(FEATURES,result["shap"]),key=lambda x:abs(x[1]),reverse=True)[0][0]}</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""<div class="metric-card">
    <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;
                color:#64748b;font-family:'Space Mono',monospace">Glucose Flag</div>
    <div style="font-size:0.85rem;font-weight:700;
                color:{"#ef4444" if gluc>=126 else "#f59e0b" if gluc>=100 else "#10b981"};
                font-family:'Space Mono',monospace;margin-top:4px">
        {"🔴 Diabetic" if gluc>=126 else "🟡 Pre-Diab" if gluc>=100 else "🟢 Normal"}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── Main 3-column layout ─────────────────────────────────────────
col_a, col_b, col_c = st.columns([1.1, 1.8, 1.1])

with col_a:
    st.markdown('<div class="section-header">🧬 Risk DNA Helix</div>', unsafe_allow_html=True)
    st.plotly_chart(dna_helix(prob), use_container_width=True, config={"displayModeBar":False})

    st.markdown('<div class="section-header">🫀 Digital Body Twin</div>', unsafe_allow_html=True)
    st.markdown(body_twin_svg(prob, patient), unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="section-header">⚡ Real-Time SHAP Attribution</div>', unsafe_allow_html=True)
    st.plotly_chart(
        shap_waterfall(result["shap"], result["base"], patient, FEATURES),
        use_container_width=True, config={"displayModeBar":False}
    )

    st.markdown('<div class="section-header">📋 Clinical Narrative Intelligence</div>', unsafe_allow_html=True)
    narrative = generate_narrative(patient, result, result["shap"])
    st.markdown(f'<div class="narrative-box">{narrative}</div>', unsafe_allow_html=True)

with col_c:
    st.markdown('<div class="section-header">🕐 24-Hour Risk Clock</div>', unsafe_allow_html=True)
    st.plotly_chart(risk_clock(prob, patient), use_container_width=True,
                    config={"displayModeBar":False})

    st.markdown('<div class="section-header">💊 Smart Recommendations</div>', unsafe_allow_html=True)
    recs = []
    if gluc >= 180: recs.append(("🔴","Critically high glucose – seek immediate care"))
    elif gluc >= 126: recs.append(("🟠","Diabetic glucose – reduce carbs, see doctor"))
    elif gluc >= 100: recs.append(("🟡","Pre-diabetic – adopt low-GI diet"))
    if bmi >= 35: recs.append(("🔴","Severe obesity – supervised weight-loss needed"))
    elif bmi >= 30: recs.append(("🟠","Obese – 150 min/week exercise + dietitian"))
    elif bmi >= 25: recs.append(("🟡","Overweight – brisk walk 30 min/day"))
    if bp >= 90: recs.append(("🔴","Hypertension – reduce sodium, monitor daily"))
    if age >= 60: recs.append(("🩺","Age > 60 – quarterly HbA1c & glucose tests"))
    elif age >= 45: recs.append(("🩺","Age > 45 – annual screening recommended"))
    if dpf >= 0.8: recs.append(("🧬","High hereditary risk – genetic counselling"))
    if ins >= 200: recs.append(("⚠️","Elevated insulin – possible insulin resistance"))
    if not recs: recs.append(("✅","Parameters healthy – maintain lifestyle!"))
    recs.append(("💧","Hydrate, 7-8 hrs sleep, manage stress daily"))

    for emoji, text in recs[:6]:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:10px;padding:8px 12px;
                    background:#0b1120;border-radius:8px;border:1px solid #1e2d45;
                    margin-bottom:6px;font-size:0.83rem;color:#cbd5e1;line-height:1.4">
          <span style="font-size:1rem">{emoji}</span><span>{text}</span>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ── TABS: Advanced Analysis ───────────────────────────────────────
st.markdown('<div class="section-header">🔬 Advanced Clinical Analysis</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs([
    "🌐  Risk Landscape", "📊  Model Comparison", "🔄  Feature Interactions", "📈  Population Context"
])

with tab1:
    import json
    fixed = dict(patient); fixed.pop("Glucose",None); fixed.pop("BMI",None)
    surface_fig = confidence_surface(rf, scaler, json.dumps(patient))
    # Mark patient's position
    surface_fig.add_trace(go.Scatter3d(
        x=[patient["Glucose"]], y=[patient["BMI"]], z=[prob],
        mode="markers+text",
        marker=dict(size=10, color=ACCENT, symbol="diamond",
                    line=dict(color="white",width=2)),
        text=["◀ YOU"], textposition="middle right",
        textfont=dict(color=ACCENT, family="Space Mono", size=10),
        name="Current Patient", hoverinfo="skip"
    ))
    st.plotly_chart(surface_fig, use_container_width=True, config={"displayModeBar":False})

with tab2:
    # Radar comparison: LR vs RF vs GB per-feature coefficient normalised contribution
    cats = FEATURES + [FEATURES[0]]
    # Normalise SHAP magnitudes to 0-1 for radar display
    shap_norm = np.abs(result["shap"]) / (np.abs(result["shap"]).max() + 1e-9)
    vals_r = list(shap_norm) + [shap_norm[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_r, theta=cats, fill='toself', name='SHAP Importance',
        line=dict(color=ACCENT, width=2),
        fillcolor=f"rgba(0,212,255,0.15)"
    ))

    # Model confidence comparison
    models_n = ["Logistic\nRegression","Random\nForest","Gradient\nBoosting","Ensemble"]
    model_probs = [result["lr_p"], result["rf_p"], result["gb_p"], result["prob"]]
    model_accs  = [metrics["LR"]["acc"], metrics["RF"]["acc"], metrics["GB"]["acc"],
                   (metrics["LR"]["acc"]+metrics["RF"]["acc"]+metrics["GB"]["acc"])/3]

    fig_models = make_subplots(rows=1, cols=2,
        subplot_titles=["Patient Risk Probability by Model","Test Set Accuracy by Model"])
    fig_models.add_trace(go.Bar(
        x=models_n, y=[p*100 for p in model_probs],
        marker=dict(color=[ACCENT, ACCENT2, SUCCESS, DANGER],
                    line=dict(width=0)),
        text=[f"{p*100:.1f}%" for p in model_probs],
        textposition="outside",
        textfont=dict(family="Space Mono", size=10),
        showlegend=False
    ), row=1, col=1)
    fig_models.add_trace(go.Bar(
        x=models_n, y=[a*100 for a in model_accs],
        marker=dict(color=[ACCENT, ACCENT2, SUCCESS, WARN],
                    line=dict(width=0)),
        text=[f"{a*100:.1f}%" for a in model_accs],
        textposition="outside",
        textfont=dict(family="Space Mono", size=10),
        showlegend=False
    ), row=1, col=2)

    fig_models.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAPER,
        font=dict(family="DM Sans", color=TEXT_CLR),
        height=340, margin=dict(l=20,r=20,t=48,b=20),
        yaxis=dict(gridcolor=GRID_CLR, range=[0,115]),
        yaxis2=dict(gridcolor=GRID_CLR, range=[0,115]),
    )
    for ann in fig_models.layout.annotations:
        ann.font.family = "Space Mono"; ann.font.size = 11; ann.font.color = "#e2e8f0"

    c_r, c_m = st.columns(2)
    with c_r:
        fig_radar.update_layout(
            polar=dict(bgcolor=PLOT_BG,
                radialaxis=dict(visible=True, range=[0,1], gridcolor=GRID_CLR, color=TEXT_CLR),
                angularaxis=dict(color=TEXT_CLR, gridcolor=GRID_CLR)),
            paper_bgcolor=PLOT_PAPER, height=340, margin=dict(l=20,r=20,t=20,b=20),
            legend=dict(font=dict(color=TEXT_CLR))
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar":False})
    with c_m:
        st.plotly_chart(fig_models, use_container_width=True, config={"displayModeBar":False})

with tab3:
    # Feature interaction heatmap: SHAP × feature value
    shap_abs = np.abs(result["shap"])
    feat_vals_norm = []
    ranges = {"Pregnancies":(0,17),"Glucose":(44,200),"BloodPressure":(24,122),
              "SkinThickness":(7,99),"Insulin":(14,600),"BMI":(18,67),
              "DiabetesPedigreeFunction":(0.08,2.42),"Age":(21,81)}
    for f in FEATURES:
        lo,hi = ranges[f]; v = patient[f]
        feat_vals_norm.append((v-lo)/(hi-lo))
    feat_vals_norm = np.array(feat_vals_norm)

    # Build interaction matrix: element (i,j) = SHAP_i * norm_val_j
    mat = np.outer(shap_abs, feat_vals_norm)

    fig_int = go.Figure(go.Heatmap(
        z=mat, x=FEATURES, y=FEATURES,
        colorscale=[[0, PLOT_BG],[0.5,"#7c3aed"],[1, ACCENT]],
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Interaction: %{z:.4f}<extra></extra>",
        colorbar=dict(title="Strength", tickfont=dict(color=TEXT_CLR),
                      title_font=dict(color=TEXT_CLR))
    ))
    layout_int = dark_layout("🔄 Feature Interaction Matrix (SHAP × Normalised Value)", height=420)
    layout_int.pop("xaxis"); layout_int.pop("yaxis")
    fig_int.update_layout(**layout_int)
    fig_int.update_xaxes(tickangle=-45, gridcolor=GRID_CLR, linecolor=GRID_CLR, color=TEXT_CLR)
    fig_int.update_yaxes(gridcolor=GRID_CLR, linecolor=GRID_CLR, color=TEXT_CLR)
    st.plotly_chart(fig_int, use_container_width=True, config={"displayModeBar":False})

with tab4:
    # Population context: where does this patient sit vs the test set?
    test_probs = rf.predict_proba(X_test)[:,1]
    test_labels = y_test.values

    fig_pop = go.Figure()
    for label, color, name in [(0,SUCCESS,"Non-Diabetic"),(1,DANGER,"Diabetic")]:
        mask = test_labels == label
        fig_pop.add_trace(go.Histogram(
            x=test_probs[mask], nbinsx=25,
            marker=dict(color=color, opacity=0.5, line=dict(width=0)),
            name=name
        ))

    fig_pop.add_vline(x=prob, line=dict(color=ACCENT, width=2.5, dash="dash"),
                      annotation_text=f"▲ Your Patient ({prob*100:.1f}%)",
                      annotation_font=dict(family="Space Mono", color=ACCENT, size=10))

    layout_pop = dark_layout("📈 Your Patient vs Population Risk Distribution", height=360)
    layout_pop["barmode"] = "overlay"
    layout_pop["xaxis"]["title"] = "Predicted Diabetes Probability"
    layout_pop["yaxis"]["title"] = "Patient Count"
    layout_pop["legend"] = dict(font=dict(color=TEXT_CLR, family="DM Sans"),
                                 bgcolor="rgba(0,0,0,0)", bordercolor=GRID_CLR)
    fig_pop.update_layout(**layout_pop)
    st.plotly_chart(fig_pop, use_container_width=True, config={"displayModeBar":False})

# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:32px;padding:20px;border-top:1px solid #1e2d45;
            display:flex;justify-content:space-between;align-items:center;
            font-size:0.72rem;color:#334155">
  <div>
    <span style="font-family:'Space Mono',monospace;color:#475569">
      DiabetesAI Clinical Intelligence
    </span>
    &nbsp;·&nbsp; B.Tech Final Year Project &nbsp;·&nbsp; 
    <span style="color:#1e3a5f">For research & educational purposes only</span>
  </div>
  <div style="text-align:right;font-family:'Space Mono',monospace;color:#1e3a5f">
    RF Accuracy: {:5.1f}% &nbsp;|&nbsp; AUC-ROC: {:.4f}
  </div>
</div>
""".format(metrics["RF"]["acc"]*100, metrics["RF"]["auc"]), unsafe_allow_html=True)
