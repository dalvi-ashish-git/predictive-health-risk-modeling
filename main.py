"""Streamlit interface for the heart disease screening model."""

import streamlit as st

from heart_disease_model import get_model, predict_patient


st.set_page_config(page_title="Heart Risk Screening", page_icon="+", layout="wide")
st.title("UCI Heart Disease Screening")
st.caption("Academic decision-support demonstration. This is not a diagnosis.")

bundle = get_model()
options = bundle.categorical_options

with st.sidebar:
    st.header("Model performance")
    st.metric("Accuracy", f"{bundle.metrics['accuracy']:.1%}")
    st.metric("ROC-AUC", f"{bundle.metrics['roc_auc']:.3f}")
    st.caption("Metrics use a stratified held-out test set.")

st.subheader("Patient inputs")
with st.form("screening_form"):
    demographic, clinical, exercise = st.columns(3)
    with demographic:
        st.markdown("#### Demographics")
        age = st.number_input("Age", min_value=18.0, max_value=120.0, value=50.0, step=1.0)
        sex = st.selectbox("Sex", options["sex"])
        cp = st.selectbox("Chest pain type", options["cp"])
    with clinical:
        st.markdown("#### Clinical measurements")
        trestbps = st.number_input("Resting blood pressure (mm Hg)", 60.0, 260.0, 130.0, step=1.0)
        chol = st.number_input("Serum cholesterol (mg/dl)", 80.0, 700.0, 210.0, step=1.0)
        fbs = int(st.toggle("Fasting blood sugar > 120 mg/dl", value=False))
        restecg = st.selectbox("Resting ECG", options["restecg"])
    with exercise:
        st.markdown("#### Exercise findings")
        thalach = st.number_input("Maximum heart rate", 40.0, 250.0, 150.0, step=1.0)
        exang = int(st.toggle("Exercise-induced angina", value=False))
        oldpeak = st.number_input("ST depression", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Exercise ST slope", options["slope"])
        ca = st.number_input("Major vessels (0-3)", 0.0, 3.0, 0.0, step=1.0)
        thal = st.selectbox("Thalassemia status", options["thal"])
    submitted = st.form_submit_button("Run clinical screening", type="primary")

if submitted:
    patient = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    result = predict_patient(patient)
    st.divider()
    st.subheader("Clinical screening result")
    result_col, details_col = st.columns([1, 2])
    with result_col:
        if result["prediction"]:
            st.error("HIGH RISK SCREENING RESULT")
        else:
            st.success("LOW RISK SCREENING RESULT")
        st.metric("Estimated cardiovascular risk", f"{result['probability']:.1%}")
    with details_col:
        st.markdown(
            f"""
            <div style="border: 1px solid #d9dee7; border-radius: 8px; padding: 1rem 1.25rem; background-color: #f8fafc;">
                <div style="font-size: 0.8rem; color: #5b6472; text-transform: uppercase;">Calculated health age</div>
                <div style="font-size: 2rem; font-weight: 700; color: #172033; margin: 0.2rem 0 0.75rem;">{result['health_age']} years</div>
                <div style="color: #4b5563;">Use this output for academic analysis only; it does not replace clinical assessment.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
