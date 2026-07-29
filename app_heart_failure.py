import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import math

st.set_page_config(
    page_title="Heart Failure Survival Predictor",
    page_icon="🫀",
    layout="centered"
)

# ---------------------------------------------------------
# Theme / typography / signature elements (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ---------- Design tokens ---------- */
:root {
    --canvas: #EEF2EF;
    --surface: #FFFFFF;
    --ink: #12262B;
    --ink-muted: #4B5D5A;
    --ink-soft: #6B7C79;
    --line: #DAE3DF;
    --teal: #0F6B5C;
    --teal-deep: #08201C;
    --teal-tint: #E6F1EC;
    --good: #1E9E74;
    --amber: #D9A441;
    --alert: #C2452B;
    --shadow-sm: 0 1px 2px rgba(18, 38, 43, 0.05), 0 1px 1px rgba(18, 38, 43, 0.04);
    --shadow-md: 0 8px 24px rgba(18, 38, 43, 0.08), 0 2px 6px rgba(18, 38, 43, 0.05);
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(1200px 480px at 50% -10%, rgba(15, 107, 92, 0.10), rgba(15, 107, 92, 0) 60%),
        var(--canvas);
}

.block-container {
    padding-top: 2.6rem;
    padding-bottom: 3rem;
    max-width: 760px;
}

/* ---------- Header ---------- */
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--teal);
    font-weight: 600;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.eyebrow::before {
    content: "";
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 0 3px var(--teal-tint);
}

.hf-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.35rem;
    color: var(--ink);
    line-height: 1.12;
    letter-spacing: -0.01em;
    margin-bottom: 0.5rem;
}

.hf-subtitle {
    color: var(--ink-muted);
    font-size: 1.03rem;
    font-weight: 500;
    line-height: 1.5;
    max-width: 48ch;
    margin-bottom: 1.4rem;
}

/* Section label above each card */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink);
    font-weight: 700;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin: 1.8rem 0 0.6rem 0;
}
.section-label .tag {
    color: var(--ink-soft);
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: none;
    font-size: 0.76rem;
}

/* ---------- Form cards (st.container(border=True)) ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    box-shadow: var(--shadow-sm);
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: 16px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
    gap: 0.35rem;
}

/* Streamlit widget labels */
[data-testid="stWidgetLabel"] p {
    color: var(--ink) !important;
    font-weight: 600 !important;
    font-size: 0.93rem !important;
}
[data-testid="stWidgetLabel"] {
    margin-bottom: 0.1rem;
}

/* Radio option text */
[data-testid="stRadio"] label p {
    color: var(--ink) !important;
    font-weight: 500 !important;
}
[data-testid="stRadio"] > div {
    gap: 0.4rem;
}

/* Slider / number input current value text */
[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"],
[data-testid="stThumbValue"] {
    color: var(--ink) !important;
    font-weight: 600 !important;
}
[data-testid="stSlider"] [role="slider"] {
    box-shadow: 0 0 0 5px var(--teal-tint) !important;
}
div[data-baseweb="input"] {
    border-radius: 8px !important;
}

/* Tooltip icon next to labels */
[data-testid="stTooltipIcon"] svg {
    color: var(--teal) !important;
}

/* ---------- ECG hero strip ---------- */
.ecg-wrap {
    width: 100%;
    height: 58px;
    margin: 0.1rem 0 1.6rem 0;
    overflow: hidden;
    border-bottom: 1px solid var(--line);
    padding-bottom: 0.9rem;
}
.ecg-path {
    stroke: var(--teal);
    stroke-width: 2.25;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-dasharray: 900;
    stroke-dashoffset: 900;
    animation: draw 3.6s linear infinite;
}
@keyframes draw {
    0%   { stroke-dashoffset: 900; }
    100% { stroke-dashoffset: 0; }
}

/* ---------- Buttons ---------- */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.01em;
    border-radius: 10px;
    height: 3rem;
    background: var(--teal) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--teal) !important;
    box-shadow: var(--shadow-sm);
    transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
}
.stButton > button:hover {
    background: #0C5648 !important;
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ---------- Monitor result panel ---------- */
.monitor-panel {
    background:
        radial-gradient(420px 220px at 85% -10%, rgba(166, 233, 210, 0.10), rgba(166, 233, 210, 0) 60%),
        var(--teal-deep);
    border-radius: 18px;
    padding: 1.9rem 2rem 1.7rem 2rem;
    margin-top: 0.7rem;
    margin-bottom: 0.9rem;
    box-shadow: var(--shadow-md);
}
.monitor-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #A6E9D2;
    font-weight: 600;
    margin-bottom: 1rem;
    text-align: center;
}
.gauge-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.gauge-readout {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.5rem;
    font-weight: 600;
    line-height: 1;
    margin-top: -0.6rem;
}
.gauge-caption {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: #9FBBB4;
    font-weight: 500;
    margin-top: 0.2rem;
    letter-spacing: 0.02em;
}
.monitor-verdict {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    margin-top: 0.9rem;
    text-align: center;
}
.monitor-note {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: #8FA8A2;
    font-weight: 500;
    text-align: center;
    margin-top: 0.3rem;
    max-width: 40ch;
}
.pulse-icon {
    display: inline-block;
    animation: beat 1.6s ease-in-out infinite;
    margin-right: 0.45rem;
}
@keyframes beat {
    0%, 100% { transform: scale(1); }
    15% { transform: scale(1.18); }
    30% { transform: scale(1); }
}

.hf-caption {
    color: var(--ink-soft);
    font-size: 0.84rem;
    font-weight: 500;
    margin-top: 1.1rem;
    padding-top: 0.9rem;
    border-top: 1px solid var(--line);
}

/* ---------- Glossary expander ---------- */
[data-testid="stExpander"] {
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    background-color: var(--surface) !important;
    margin-bottom: 1.3rem;
    box-shadow: var(--shadow-sm);
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.98rem;
    color: var(--ink) !important;
    padding: 0.2rem 0;
}
[data-testid="stExpander"] summary:hover {
    color: var(--teal) !important;
}

/* Glossary grid */
.glossary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(215px, 1fr));
    gap: 0.65rem;
    margin-top: 0.4rem;
}
.glossary-card {
    background: var(--canvas);
    border: 1px solid var(--line);
    border-left: 3px solid var(--teal);
    border-radius: 10px;
    padding: 0.7rem 0.85rem;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.glossary-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-sm);
}
.glossary-term {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.86rem;
    color: var(--ink);
    font-weight: 700;
    margin-bottom: 0.15rem;
}
.glossary-unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.04em;
    color: var(--teal);
    font-weight: 600;
    text-transform: uppercase;
    display: inline-block;
    margin-bottom: 0.3rem;
}
.glossary-def {
    color: var(--ink-muted);
    font-size: 0.83rem;
    font-weight: 450;
    line-height: 1.4;
}
.glossary-target {
    margin-top: 0.85rem;
    padding: 0.85rem 1rem;
    background: var(--teal-tint);
    border: 1px solid var(--teal);
    border-left: 3px solid var(--teal);
    border-radius: 10px;
}
.glossary-target .glossary-term { color: var(--teal); }

/* Mobile tightening */
@media (max-width: 640px) {
    .hf-title { font-size: 1.85rem; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .glossary-grid { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Feature glossary (definitions used for tooltips + reference panel)
# ---------------------------------------------------------
FEATURE_GLOSSARY = {
    "age": ("Age", "years", "Age of the patient."),
    "anaemia": ("Anaemia", "yes / no", "Decrease of red blood cells or hemoglobin."),
    "creatinine_phosphokinase": ("Creatinine phosphokinase (CPK)", "mcg/L", "Level of the CPK enzyme in the blood. Elevated after heart or muscle injury."),
    "diabetes": ("Diabetes", "yes / no", "Whether the patient has diabetes."),
    "ejection_fraction": ("Ejection fraction", "%", "Percentage of blood leaving the heart at each contraction."),
    "high_blood_pressure": ("High blood pressure", "yes / no", "Whether the patient has hypertension."),
    "platelets": ("Platelets", "kiloplatelets/mL", "Concentration of platelets in the blood."),
    "sex": ("Sex", "female / male", "Biological sex of the patient."),
    "serum_creatinine": ("Serum creatinine", "mg/dL", "Level of serum creatinine in the blood. A marker of kidney function."),
    "serum_sodium": ("Serum sodium", "mEq/L", "Level of serum sodium in the blood."),
    "smoking": ("Smoking", "yes / no", "Whether the patient smokes."),
    "time": ("Follow-up period", "days", "Number of days the patient was monitored after diagnosis."),
}

def glossary_help(key):
    """Return a short tooltip string for a widget's help= parameter."""
    _, unit, definition = FEATURE_GLOSSARY[key]
    return f"{definition} (unit: {unit})" if unit not in ("yes / no", "female / male") else definition

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
MODEL_PATH = "heart_failure_rf_model_randomsearch.pkl"
FEATURES_PATH = "feature_names.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    if os.path.exists(FEATURES_PATH):
        feature_names = joblib.load(FEATURES_PATH)
    else:
        feature_names = list(model.feature_names_in_)
    return model, feature_names

model, feature_names = load_model()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown('<div class="eyebrow">Clinical decision support</div>', unsafe_allow_html=True)
st.markdown('<div class="hf-title">Heart Failure Survival Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hf-subtitle">Estimates a patient\'s risk of death during follow-up, '
    'based on twelve clinical and lifestyle indicators.</div>',
    unsafe_allow_html=True
)

# Animated ECG strip (signature element)
st.markdown("""
<div class="ecg-wrap">
<svg viewBox="0 0 900 54" width="100%" height="54" preserveAspectRatio="none">
  <path class="ecg-path" d="M0,27 L150,27 L165,27 L175,10 L185,44 L195,4 L205,40 L215,27 L260,27
  L410,27 L425,27 L435,10 L445,44 L455,4 L465,40 L475,27 L520,27
  L670,27 L685,27 L695,10 L705,44 L715,4 L725,40 L735,27 L900,27"/>
</svg>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(
        "Model file not found. Place **heart_failure_rf_model_randomsearch.pkl** "
        "in the same folder as app.py, then refresh."
    )
    st.stop()

# ---------------------------------------------------------
# Feature glossary section
# ---------------------------------------------------------
with st.expander("📖  What do these fields mean?", expanded=False):
    cards = "".join(
        f"""<div class="glossary-card">
                <div class="glossary-term">{term}</div>
                <div class="glossary-unit">{unit}</div>
                <div class="glossary-def">{definition}</div>
            </div>"""
        for term, unit, definition in FEATURE_GLOSSARY.values()
    )
    st.markdown(f'<div class="glossary-grid">{cards}</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="glossary-target">
               <div class="glossary-term">Target · death event</div>
               <div class="glossary-def">Whether the patient died during the follow-up period.
               This is what the model predicts — the readout below shows the estimated
               probability of this outcome.</div>
           </div>""",
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.markdown(
    '<div class="section-label"><span>Vitals</span><span class="tag">6 fields</span></div>',
    unsafe_allow_html=True
)
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=18, max_value=100, value=60, help=glossary_help("age"))
        ejection_fraction = st.slider("Ejection fraction (%)", min_value=10, max_value=80, value=38, help=glossary_help("ejection_fraction"))
        serum_sodium = st.slider("Serum sodium (mEq/L)", min_value=110, max_value=150, value=137, help=glossary_help("serum_sodium"))
    with col2:
        serum_creatinine = st.number_input("Serum creatinine (mg/dL)", min_value=0.1, max_value=10.0, value=1.1, step=0.1, help=glossary_help("serum_creatinine"))
        creatinine_phosphokinase = st.number_input("Creatinine phosphokinase (mcg/L)", min_value=0, max_value=8000, value=250, help=glossary_help("creatinine_phosphokinase"))
        platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=25000.0, max_value=900000.0, value=263000.0, step=1000.0, format="%.0f", help=glossary_help("platelets"))

st.markdown(
    '<div class="section-label"><span>History &amp; monitoring</span><span class="tag">6 fields</span></div>',
    unsafe_allow_html=True
)
with st.container(border=True):
    col3, col4 = st.columns(2)
    with col3:
        anaemia = st.radio("Anaemia", ["No", "Yes"], horizontal=True, help=glossary_help("anaemia"))
        diabetes = st.radio("Diabetes", ["No", "Yes"], horizontal=True, help=glossary_help("diabetes"))
        high_blood_pressure = st.radio("High blood pressure", ["No", "Yes"], horizontal=True, help=glossary_help("high_blood_pressure"))
    with col4:
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True, help=glossary_help("sex"))
        smoking = st.radio("Smoking", ["No", "Yes"], horizontal=True, help=glossary_help("smoking"))
        time = st.slider("Follow-up period (days)", min_value=0, max_value=300, value=115, help=glossary_help("time"))

st.write("")
predict_clicked = st.button("Run prediction", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Result
# ---------------------------------------------------------
if predict_clicked:
    input_dict = {
        "age": age,
        "anaemia": 1 if anaemia == "Yes" else 0,
        "creatinine_phosphokinase": creatinine_phosphokinase,
        "diabetes": 1 if diabetes == "Yes" else 0,
        "ejection_fraction": ejection_fraction,
        "high_blood_pressure": 1 if high_blood_pressure == "Yes" else 0,
        "platelets": platelets,
        "serum_creatinine": serum_creatinine,
        "serum_sodium": serum_sodium,
        "sex": 1 if sex == "Male" else 0,
        "smoking": 1 if smoking == "Yes" else 0,
        "time": time,
    }
    input_df = pd.DataFrame([input_dict])[feature_names]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        accent = "#E2542D"
        verdict = "Elevated risk of a death event"
    else:
        accent = "#43D9A3"
        verdict = "Lower risk of a death event"

    # Needle geometry for the semicircular risk gauge
    cx, cy, r_track, r_needle = 120, 120, 95, 76
    theta = math.radians(180 * (1 - probability))
    needle_x = cx + r_needle * math.cos(theta)
    needle_y = cy - r_needle * math.sin(theta)

    st.markdown(f"""
    <div class="monitor-panel">
        <div class="monitor-eyebrow">Model output · probability of a death event</div>
        <div class="gauge-wrap">
            <svg width="240" height="132" viewBox="0 0 240 140">
                <defs>
                    <linearGradient id="gaugeGrad" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stop-color="#1E9E74"/>
                        <stop offset="50%" stop-color="#D9A441"/>
                        <stop offset="100%" stop-color="#C2452B"/>
                    </linearGradient>
                </defs>
                <path d="M {cx - r_track},{cy} A {r_track},{r_track} 0 0 1 {cx + r_track},{cy}"
                      fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="18" stroke-linecap="round"/>
                <path d="M {cx - r_track},{cy} A {r_track},{r_track} 0 0 1 {cx + r_track},{cy}"
                      fill="none" stroke="url(#gaugeGrad)" stroke-width="13" stroke-linecap="round" opacity="0.92"/>
                <line x1="{cx}" y1="{cy}" x2="{needle_x:.1f}" y2="{needle_y:.1f}"
                      stroke="#F4FBF8" stroke-width="4" stroke-linecap="round"/>
                <circle cx="{cx}" cy="{cy}" r="7.5" fill="#F4FBF8"/>
            </svg>
            <div class="gauge-readout" style="color:{accent};">{probability:.1%}</div>
            <div class="gauge-caption">predicted probability</div>
        </div>
        <div class="monitor-verdict" style="color:#E7F3EF;"><span class="pulse-icon">♥</span>{verdict}</div>
        <div class="monitor-note">Based on the {len(feature_names)} values entered above — not a diagnosis.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="hf-caption">For educational purposes only — not a substitute for '
    'professional medical advice or diagnosis.</div>',
    unsafe_allow_html=True
)