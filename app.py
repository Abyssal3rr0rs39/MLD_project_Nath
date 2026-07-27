import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Heart Failure Survival Predictor",
    page_icon="❤️",
    layout="centered"
)

# ---------------------------------------------------------
# Load model + feature names
# ---------------------------------------------------------
MODEL_PATH = "heart_failure_rf_model_randomsearch.pkl"
FEATURES_PATH = "feature_names.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)

    # Prefer a standalone feature_names.pkl if present, otherwise fall back
    # to the feature names stored inside the trained model itself.
    if os.path.exists(FEATURES_PATH):
        feature_names = joblib.load(FEATURES_PATH)
    else:
        feature_names = list(model.feature_names_in_)

    return model, feature_names

model, feature_names = load_model()

st.title("❤️ Heart Failure Survival Predictor")
st.write(
    "This app estimates the risk of death from heart failure based on a "
    "patient's clinical record, using a Random Forest model tuned with "
    "RandomizedSearchCV."
)

if model is None:
    st.error(
        "Model files not found. Please place **heart_failure_rf_model_randomsearch.pkl** "
        "and **feature_names.pkl** in the same folder as this app.py, then refresh the page."
    )
    st.stop()

st.divider()
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=100, value=60)
    creatinine_phosphokinase = st.number_input(
        "Creatinine Phosphokinase (mcg/L)", min_value=0, max_value=8000, value=250
    )
    ejection_fraction = st.slider("Ejection Fraction (%)", min_value=10, max_value=80, value=38)
    platelets = st.number_input(
        "Platelets (kiloplatelets/mL)", min_value=25000.0, max_value=900000.0,
        value=263000.0, step=1000.0, format="%.0f"
    )
    serum_creatinine = st.number_input(
        "Serum Creatinine (mg/dL)", min_value=0.1, max_value=10.0, value=1.1, step=0.1
    )
    serum_sodium = st.slider("Serum Sodium (mEq/L)", min_value=110, max_value=150, value=137)
    time = st.slider("Follow-up Period (days)", min_value=0, max_value=300, value=115)

with col2:
    anaemia = st.radio("Anaemia", ["No", "Yes"], horizontal=True)
    diabetes = st.radio("Diabetes", ["No", "Yes"], horizontal=True)
    high_blood_pressure = st.radio("High Blood Pressure", ["No", "Yes"], horizontal=True)
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
    smoking = st.radio("Smoking", ["No", "Yes"], horizontal=True)

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
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

    # Build dataframe in the exact column order the model was trained on
    input_df = pd.DataFrame([input_dict])[feature_names]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ High risk of death event  —  probability: **{probability:.1%}**")
    else:
        st.success(f"✅ Low risk of death event  —  probability: **{probability:.1%}**")

    st.progress(min(max(probability, 0.0), 1.0))

    with st.expander("What features mattered most (model-level, not per-patient)"):
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        st.bar_chart(importance.set_index("Feature"))

    st.caption(
        "This tool is for educational purposes only and is not a substitute "
        "for professional medical advice or diagnosis."
    )
