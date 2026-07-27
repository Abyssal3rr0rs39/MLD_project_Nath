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

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top, rgba(15, 107, 92, 0.15), transparent 35%),
                linear-gradient(180deg, #E7F4F1 0%, #F9FBFB 100%);
    color: #12262B;
    padding: 1.2rem 0 2rem;
}

.page-shell {
    max-width: 1080px;
    margin: 0 auto;
}

.card {
    background: rgba(255, 255, 255, 0.96);
    border-radius: 26px;
    border: 1px solid rgba(15, 107, 92, 0.12);
    box-shadow: 0 24px 48px rgba(11, 34, 36, 0.08);
    padding: 1.8rem;
    margin-bottom: 1.6rem;
}

.hero-card {
    background: linear-gradient(135deg, #0F6B5C, #145E74);
    color: #ffffff;
    overflow: hidden;
}

.hero-card .eyebrow,
.hero-card .hf-title,
.hero-card .hf-subtitle {
    color: #ffffff;
}

.hero-card .hf-title {
    font-size: 2.6rem;
    letter-spacing: -0.03em;
}

.hero-card .hf-subtitle {
    color: rgba(255, 255, 255, 0.85);
    max-width: 56ch;
}

.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    margin-bottom: 1.1rem;
}

.hero-chip {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 999px;
    padding: 0.45rem 0.9rem;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #E7F4F1;
}

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #55686C;
    font-weight: 700;
    margin-bottom: 1rem;
}

.input-grid {
    display: grid;
    gap: 1rem;
}

.monitor-panel {
    background: linear-gradient(180deg, rgba(13, 47, 47, 0.98), rgba(7, 22, 23, 0.96));
    border-radius: 22px;
    padding: 1.8rem 2rem;
    margin-top: 0.8rem;
}

.monitor-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7EE0B5;
    margin-bottom: 0.65rem;
}

.monitor-readout {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.2rem;
}

.monitor-verdict {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    margin-top: 0.4rem;
    color: rgba(255, 255, 255, 0.88);
}

.pulse-icon {
    display: inline-block;
    animation: beat 1.6s ease-in-out infinite;
    margin-right: 0.5rem;
}

@keyframes beat {
    0%, 100% { transform: scale(1); }
    15% { transform: scale(1.16); }
    30% { transform: scale(1); }
}

.stButton > button {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    letter-spacing: 0.02em;
    border-radius: 12px;
    height: 3.1rem;
    background-color: #0F6B5C;
    color: #ffffff;
    border: none;
    box-shadow: 0 16px 30px rgba(15, 107, 92, 0.18);
}

.stButton > button:hover {
    background-color: #14816f;
}

.info-card {
    background: #F8FBFB;
    border: 1px solid rgba(8, 32, 28, 0.08);
}

.hf-caption {
    color: #55686C;
    font-size: 0.85rem;
    margin-top: 0.6rem;
}

@media (max-width: 768px) {
    .hero-card .hf-title {
        font-size: 2rem;
    }
}
</style>
""", unsafe_allow_html=True)

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
st.markdown('<div class="page-shell">', unsafe_allow_html=True)
st.markdown('<div class="card hero-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-tag"><span class="hero-chip">Clinical decision support</span></div>' +
    '<div class="eyebrow">Random Forest prediction for heart failure</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="hf-title">Heart Failure Survival Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hf-subtitle">Estimates risk of death from heart failure using a '
    'Random Forest model tuned with RandomizedSearchCV on clinical records.</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="ecg-wrap">
<svg viewBox="0 0 900 54" width="100%" height="54" preserveAspectRatio="none">
  <path class="ecg-path" d="M0,27 L150,27 L165,27 L175,10 L185,44 L195,4 L205,40 L215,27 L260,27
  L410,27 L425,27 L435,10 L445,44 L455,4 L465,40 L475,27 L520,27
  L670,27 L685,27 L695,10 L705,44 L715,4 L725,40 L735,27 L900,27"/>
</svg>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if model is None:
    st.error(
        "Model file not found. Place **heart_failure_rf_model_randomsearch.pkl** "
        "in the same folder as this app script, then refresh."
    )
    st.stop()

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.markdown('<div class="card info-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Vitals</div>', unsafe_allow_html=True)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=18, max_value=100, value=60)
        ejection_fraction = st.slider("Ejection fraction (%)", min_value=10, max_value=80, value=38)
        serum_sodium = st.slider("Serum sodium (mEq/L)", min_value=110, max_value=150, value=137)
    with col2:
        serum_creatinine = st.number_input("Serum creatinine (mg/dL)", min_value=0.1, max_value=10.0, value=1.1, step=0.1)
        creatinine_phosphokinase = st.number_input("Creatinine phosphokinase (mcg/L)", min_value=0, max_value=8000, value=250)
        platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=25000.0, max_value=900000.0, value=263000.0, step=1000.0, format="%.0f")

    st.markdown('<div class="section-label">History &amp; monitoring</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        anaemia = st.radio("Anaemia", ["No", "Yes"], horizontal=True)
        diabetes = st.radio("Diabetes", ["No", "Yes"], horizontal=True)
        high_blood_pressure = st.radio("High blood pressure", ["No", "Yes"], horizontal=True)
    with col4:
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        smoking = st.radio("Smoking", ["No", "Yes"], horizontal=True)
        time = st.slider("Follow-up period (days)", min_value=0, max_value=300, value=115)

st.markdown('</div>', unsafe_allow_html=True)

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

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="monitor-panel">
        <div class="monitor-eyebrow">Model output</div>
        <div class="monitor-readout" style="color:{accent};">
            <span class="pulse-icon">♥</span>{probability:.1%}
        </div>
        <div class="monitor-verdict">{verdict}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("What features mattered most (model-level, not per-patient)"):
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        st.bar_chart(importance.set_index("Feature"), color="#0F6B5C")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="hf-caption">For educational purposes only — not a substitute for '
    'professional medical advice or diagnosis.</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)
