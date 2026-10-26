 """Streamlit interface for the Customer Churn Prediction System."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
DATA_PATH = ROOT / "data" / "customer_churn.csv"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Prediction & Retention Assistant")
st.caption("Machine Learning • Customer Analytics • Applied AI")

if not MODEL_PATH.exists():
    st.warning("Model file not found. Run `python train_model.py` in the project folder, then refresh this page.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.sidebar:
    st.header("Customer Profile")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    tech_support = st.selectbox("Tech support", ["Yes", "No"])
    security = st.selectbox("Online security", ["Yes", "No"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    monthly = st.number_input("Monthly charges", min_value=0.0, max_value=300.0, value=75.0, step=1.0)
    calls = st.slider("Support calls in recent period", 0, 10, 2)
    dependents = st.selectbox("Has dependents?", ["Yes", "No"])
    predict = st.button("Predict churn risk", type="primary", use_container_width=True)

st.subheader("Customer Risk Assessment")
left, right = st.columns([1, 1])
with left:
    st.write("Enter a customer profile in the sidebar and select **Predict churn risk**.")
    st.info("This is a demonstration project trained on synthetic data. Do not use its predictions to make real business decisions without validating it on real, representative data.")

if predict:
    profile = pd.DataFrame([{
        "tenure_months": tenure,
        "contract_type": contract,
        "internet_service": internet,
        "tech_support": tech_support,
        "online_security": security,
        "payment_method": payment,
        "paperless_billing": paperless,
        "monthly_charges": monthly,
        "support_calls": calls,
        "has_dependents": dependents
    }])
    probability = float(model.predict_proba(profile)[0, 1])
    risk = "High" if probability >= 0.65 else ("Medium" if probability >= 0.35 else "Low")
    with right:
        st.metric("Estimated churn probability", f"{probability:.1%}")
        st.metric("Risk category", risk)
        st.progress(min(max(probability, 0.0), 1.0))
        st.subheader("Suggested retention actions")
        actions = []
        if contract == "Month-to-month":
            actions.append("Offer a suitable longer-term plan with clear pricing.")
        if tenure < 12:
            actions.append("Provide an onboarding check-in and explain key benefits.")
        if tech_support == "No" or calls >= 3:
            actions.append("Offer proactive technical support and follow up on unresolved issues.")
        if security == "No":
            actions.append("Explain available security features and their value.")
        if monthly >= 85:
            actions.append("Review plan fit and explain lower-cost options where appropriate.")
        if payment == "Electronic check":
            actions.append("Offer convenient alternative payment methods if the customer prefers.")
        if not actions:
            actions.append("Offer a satisfaction check-in and ask what would improve the service.")
        for action in actions:
            st.write(f"• {action}")
        st.caption("Suggestions are rule-based examples, not guarantees that a customer will stay.")

st.divider()
st.subheader("Dataset Overview")
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    c1, c2, c3 = st.columns(3)
    c1.metric("Demo customers", f"{len(df):,}")
    c2.metric("Churned customers", f"{(df['churn'] == 'Yes').sum():,}")
    c3.metric("Churn rate", f"{(df['churn'] == 'Yes').mean():.1%}")
    st.bar_chart(df["churn"].value_counts().rename_axis("Customer outcome").to_frame("Count"))
    with st.expander("Preview sample data"):
        st.dataframe(df.head(20), use_container_width=True)
