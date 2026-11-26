import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)


def get_model_path(filename: str) -> pathlib.Path:
    path = MODELS_DIR / filename
    if not path.is_file():
        path = BASE_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required model file was not found: {path}")
    return path


@st.cache_resource
def load_pipeline(filename: str):
    return load(get_model_path(filename))


def get_file_signature(path: pathlib.Path):
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size)


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
    return shap_values


def select_top_shap_features(shap_values, feature_count=20):
    values = np.asarray(getattr(shap_values, "values", shap_values))
    if values.ndim != 2:
        raise ValueError("Expected SHAP values with samples and features.")

    top_indices = np.argsort(np.abs(values).mean(axis=0))[-feature_count:][::-1]
    return shap_values[:, top_indices]


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


@st.cache_data(show_spinner="Calculating model performance...")
def calculate_model_performance(dataset_signature, model_signatures, _models):
    data = pd.read_csv(dataset_signature[0])
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna(subset=["TotalCharges"])

    target = (
        data["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )
    valid = target.notna()
    features = data.drop(columns=["Churn", "CustomerID"]).loc[valid]
    target = target.loc[valid].astype(int)
    _, X_test, _, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    results = []
    for model_name, *_ in model_signatures:
        model = _models[model_name]
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1 Score": f1_score(y_test, predictions, zero_division=0),
                "ROC AUC": roc_auc_score(y_test, probabilities),
            }
        )

    return pd.DataFrame(results).set_index("Model")


model_files = {
    "Logistic Regression": "logistic_model.pkl",
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl",
}
models = {
    model_name: load_pipeline(filename)
    for model_name, filename in model_files.items()
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
    st.subheader("Model Performance Comparison")
    st.caption(
        "Scores use the same stratified 20% holdout set used to train the models."
    )
    dataset_signature = get_file_signature(
        BASE_DIR / "Telecom-Customer-Churn_Dataset.csv"
    )
    model_signatures = tuple(
        (model_name, *get_file_signature(get_model_path(filename)))
        for model_name, filename in model_files.items()
    )
    performance = calculate_model_performance(
        dataset_signature,
        model_signatures,
        models,
    )
    st.dataframe(
        performance.style.format("{:.3f}"),
        width="stretch",
    )

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
    top_shap = select_top_shap_features(positive_shap)

    fig = plt.figure()
    shap.plots.beeswarm(
        top_shap,
        max_display=20,
        show=False,
    )
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("SHAP Feature Importance")
    fig = plt.figure()
    shap.plots.bar(top_shap, max_display=20, show=False)
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
