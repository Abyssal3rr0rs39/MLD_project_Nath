import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F1F4F3;
}

/* Eyebrow label */
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #0F6B5C;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

/* Header title */
.hf-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    color: #12262B;
    line-height: 1.15;
    margin-bottom: 0.35rem;
}

.hf-subtitle {
    color: #2E4044;
    font-size: 1rem;
    font-weight: 500;
    max-width: 46ch;
    margin-bottom: 1.1rem;
}

/* Section eyebrow used above each chart card */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #12262B;
    font-weight: 700;
    margin: 1.4rem 0 0.4rem 0;
    border-bottom: 1.5px solid #A9BAB7;
    padding-bottom: 0.35rem;
}

/* Streamlit widget labels (slider/radio/number input captions) */
[data-testid="stWidgetLabel"] p {
    color: #12262B !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* Radio option text */
[data-testid="stRadio"] label p {
    color: #12262B !important;
    font-weight: 500 !important;
}

/* Slider / number input current value text */
[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"],
[data-testid="stThumbValue"] {
    color: #12262B !important;
    font-weight: 600 !important;
}

/* ECG hero strip */
.ecg-wrap {
    width: 100%;
    height: 54px;
    margin: 0.2rem 0 1.3rem 0;
    overflow: hidden;
}
.ecg-path {
    stroke: #0F6B5C;
    stroke-width: 2;
    fill: none;
    stroke-dasharray: 900;
    stroke-dashoffset: 900;
    animation: draw 3.6s linear infinite;
}
@keyframes draw {
    0%   { stroke-dashoffset: 900; }
    100% { stroke-dashoffset: 0; }
}

/* Buttons */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    letter-spacing: 0.01em;
    border-radius: 8px;
    height: 2.9rem;
}

/* Monitor result panel */
.monitor-panel {
    background: #08201C;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-top: 0.6rem;
    margin-bottom: 0.8rem;
}
.monitor-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #A6E9D2;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.monitor-readout {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.6rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 0.15rem;
}
.monitor-verdict {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    margin-top: 0.3rem;
}
.monitor-bar-track {
    width: 100%;
    height: 8px;
    background: rgba(230, 243, 239, 0.15);
    border-radius: 999px;
    overflow: hidden;
    margin: 0.55rem 0 0.7rem 0;
}
.monitor-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.4s ease;
}
.pulse-icon {
    display: inline-block;
    animation: beat 1.6s ease-in-out infinite;
    margin-right: 0.5rem;
}
@keyframes beat {
    0%, 100% { transform: scale(1); }
    15% { transform: scale(1.18); }
    30% { transform: scale(1); }
}

.hf-caption {
    color: #45575B;
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 0.6rem;
}

/* Glossary expander */
[data-testid="stExpander"] {
    border: 1px solid #A9BAB7 !important;
    border-radius: 10px !important;
    background-color: #FFFFFF !important;
    margin-bottom: 1.1rem;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    color: #12262B !important;
}

/* Glossary grid */
.glossary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 0.7rem;
    margin-top: 0.3rem;
}
.glossary-card {
    background: #F1F4F3;
    border: 1px solid #D3DCDA;
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
}
.glossary-term {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #0F6B5C;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.glossary-unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #5B6E6A;
    font-weight: 500;
}
.glossary-def {
    color: #2E4044;
    font-size: 0.86rem;
    font-weight: 500;
    line-height: 1.35;
}
.glossary-target {
    margin-top: 0.9rem;
    padding: 0.75rem 0.9rem;
    background: #EAF3F0;
    border: 1px dashed #0F6B5C;
    border-radius: 10px;
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
st.markdown('<div class="section-label">Vitals</div>', unsafe_allow_html=True)
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

st.markdown('<div class="section-label">History &amp; monitoring</div>', unsafe_allow_html=True)
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
        accent = "#43F2A3"
        verdict = "Lower risk of a death event"

    st.markdown(f"""
    <div class="monitor-panel">
        <div class="monitor-eyebrow">Model output · predicted probability of death event</div>
        <div class="monitor-readout" style="color:{accent};">
            <span class="pulse-icon">♥</span>{probability:.1%}
        </div>
        <div class="monitor-bar-track">
            <div class="monitor-bar-fill" style="width:{probability * 100:.1f}%; background:{accent};"></div>
        </div>
        <div class="monitor-verdict" style="color:#E7F3EF;">{verdict}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="hf-caption">For educational purposes only — not a substitute for '
    'professional medical advice or diagnosis.</div>',
    unsafe_allow_html=True
)