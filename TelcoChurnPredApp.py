import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from joblib import load


BASE_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_pipeline(filename: str):
    path = MODELS_DIR / filename
    if not path.is_file():
        path = BASE_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required model file was not found: {path}")
    return load(path)


def get_classifier(pipeline):
    for step_name in ("classifier", "model"):
        if step_name in pipeline.named_steps:
            return pipeline.named_steps[step_name]
    raise KeyError("Pipeline does not contain a 'classifier' or 'model' step.")


def get_preprocessor(pipeline):
    if "preprocessor" not in pipeline.named_steps:
        raise KeyError("Pipeline does not contain a 'preprocessor' step.")
    return pipeline.named_steps["preprocessor"]


def positive_class_shap_values(shap_values):
    values = getattr(shap_values, "values", shap_values)

    values = np.asarray(values)
    if values.ndim == 3:
        positive_index = 1 if values.shape[2] >= 2 else values.shape[2] - 1
        if hasattr(shap_values, "__getitem__"):
            return shap_values[..., positive_index]
        return values[..., positive_index]
    return values


def make_explanation(model, transformed_data):
    classifier = get_classifier(model)
    if hasattr(classifier, "coef_"):
        return shap.LinearExplainer(classifier, transformed_data)(transformed_data)
    return shap.Explainer(classifier)(transformed_data)


@st.cache_data(show_spinner=False)
def get_feature_importance(feature_names, feature_importance):
    return (
        pd.DataFrame(
            {"feature": feature_names, "importance": feature_importance}
        )
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )


models = {
    "Logistic Regression": load_pipeline("logistic_model.pkl"),
    "Random Forest": load_pipeline("rf_model.pkl"),
    "XGBoost": load_pipeline("xgb_model.pkl"),
}

tab1, tab2 = st.tabs(["Prediction", "Model Insights"])

with tab1:
    st.title("Customer Churn Prediction App")
    st.write("Enter customer details to predict churn probability.")

    model_choice = st.selectbox(
        "Choose Model",
        list(models),
    )

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
    multiplelines = st.selectbox(
        "Multiple Lines", ["Yes", "No", "No phone service"]
    )
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    onlinesecurity = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    onlinebackup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    deviceprotection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    techsupport = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    streamingtv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    streamingmovies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer",
            "Credit card",
        ],
    )
    monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

    customer_data = pd.DataFrame(
        {
            "gender": [gender],
            "SeniorCitizen": [senior],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phoneservice],
            "MultipleLines": [multiplelines],
            "InternetService": [internet],
            "OnlineSecurity": [onlinesecurity],
            "OnlineBackup": [onlinebackup],
            "DeviceProtection": [deviceprotection],
            "TechSupport": [techsupport],
            "StreamingTV": [streamingtv],
            "StreamingMovies": [streamingmovies],
            "Contract": [contract],
            "PaperlessBilling": [paperless],
            "PaymentMethod": [payment],
            "MonthlyCharges": [monthly],
            "TotalCharges": [total],
        }
    )

    if st.button("Predict Churn", type="primary"):
        selected_model = models[model_choice]
        probability = float(selected_model.predict_proba(customer_data)[0][1])

        st.metric("Churn Probability", f"{probability * 100:.1f}%")
        st.progress(probability)

        if probability > 0.6:
            st.error("High Risk Customer")
        elif probability > 0.3:
            st.warning("Medium Risk Customer")
        else:
            st.success("Low Risk Customer")

        st.write(f"Model Used: **{model_choice}**")
        st.subheader("Prediction Explanation (SHAP)")

        preprocessor = get_preprocessor(selected_model)
        transformed = preprocessor.transform(customer_data)
        feature_names = preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed, columns=feature_names)
        explanation = make_explanation(selected_model, transformed_df)

        fig = plt.figure(figsize=(9, 5))
        try:
            positive_explanation = positive_class_shap_values(explanation)
            if len(getattr(positive_explanation, "shape", ())) == 2:
                single_explanation = positive_explanation[0]
            else:
                single_explanation = positive_explanation
            shap.plots.waterfall(single_explanation, show=False)
        except (IndexError, AttributeError, TypeError):
            st.warning("SHAP could not format a waterfall plot for this model.")
        else:
            st.pyplot(fig)
        finally:
            plt.close(fig)

with tab2:
    st.subheader("Random Forest Feature Importance")
    rf_model = models["Random Forest"]
    rf_preprocessor = get_preprocessor(rf_model)
    rf_classifier = get_classifier(rf_model)
    feature_names = rf_preprocessor.get_feature_names_out()
    feature_importance = rf_classifier.feature_importances_

    feat_imp = get_feature_importance(
        tuple(feature_names),
        tuple(feature_importance),
    )

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = plt.cm.Blues(np.linspace(0.45, 0.85, len(feat_imp)))
    ax.barh(
        feat_imp["feature"],
        feat_imp["importance"],
        color=colors,
        edgecolor="none",
    )
    ax.invert_yaxis()
    ax.set_title("Top Drivers of Customer Churn")
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Transformed feature")
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("SHAP Summary Plot")
    sample_data = customer_data.copy()
    sample_data = pd.concat([sample_data] * 50, ignore_index=True)

    rng = pd.Series(range(50)).sample(50, replace=True, random_state=42)
    sample_data["tenure"] = rng.values
    sample_data["MonthlyCharges"] = (
        20.0 + 100.0 * pd.Series(range(50), dtype=float) / 49.0
    )
    sample_data["TotalCharges"] = (
        100.0 + 4900.0 * pd.Series(range(50), dtype=float) / 49.0
    )

    transformed_sample = rf_preprocessor.transform(sample_data)
    transformed_sample_df = pd.DataFrame(
        transformed_sample,
        columns=rf_preprocessor.get_feature_names_out(),
    )
    shap_values = make_explanation(rf_model, transformed_sample_df)
    positive_shap = positive_class_shap_values(shap_values)

    fig = plt.figure()
    shap.plots.beeswarm(
        positive_shap,
        max_display=15,
        show=False,
    )
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("SHAP Feature Importance")
    fig = plt.figure()
    shap.plots.bar(positive_shap, max_display=15, show=False)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Business Insights")
    st.markdown(
        """
### Key Drivers of Customer Churn

- Shorter tenure is associated with a higher risk of churn.
- Month-to-month contracts generally carry more risk than annual contracts.
- Higher monthly charges can increase churn risk.
- Fiber-optic customers may have elevated churn risk.
- Customers without support or security services may be more likely to leave.

### Business Recommendations

- Encourage long-term contracts through discounts or loyalty rewards.
- Bundle security and technical-support services.
- Target high-charge and low-tenure customers with retention offers.
- Monitor customer risk regularly and prioritize outreach by model probability.
"""
    )
