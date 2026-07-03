from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


APP_TITLE = "Trust Metrics💵💹"
DATA_FILE = Path(__file__).with_name("dataset.csv")
TRAINING_ROWS = 600
NUMERIC_FALLBACKS = {
    "age": (18, 70, 1, 35),
    "income": (20_000, 590_621, 1_000, 120_000),
    "loan_amount": (50_000, 1_536_214, 5_000, 250_000),
    "loan_term_months": (6, 95, 1, 36),
    "credit_score": (338, 850, 1, 680),
    "employment_years": (0, 23, 1, 8),
    "num_prior_defaults": (0, 3, 1, 0),
}
DEFAULT_HOME_CHOICES = ["rent", "mortgage", "own"]
DEFAULT_PURPOSE_CHOICES = ["auto", "education", "business", "home_improvement", "medical", "personal"]
PURPOSE_LABELS = {
    "auto": "Auto loan",
    "education": "Education",
    "business": "Business",
    "home_improvement": "Home improvement",
    "medical": "Medical",
    "personal": "Personal",
}


st.set_page_config(page_title=APP_TITLE, page_icon="💹", layout="wide")


def build_fallback_data() -> pd.DataFrame:
    rng = pd.Series(range(TRAINING_ROWS))
    frame = pd.DataFrame(
        {
            "age": 18 + (rng * 17 % 53),
            "income": 20_000 + (rng * 23_417 % 570_621),
            "loan_amount": 50_000 + (rng * 31_337 % 1_486_214),
            "loan_term_months": 6 + (rng * 7 % 90),
            "credit_score": 338 + (rng * 13 % 513),
            "employment_years": rng * 5 % 24,
            "num_prior_defaults": rng % 4,
            "home_ownership": pd.Series([DEFAULT_HOME_CHOICES[i % 3] for i in rng]),
            "purpose": pd.Series([DEFAULT_PURPOSE_CHOICES[i % len(DEFAULT_PURPOSE_CHOICES)] for i in rng]),
        }
    )

    risk_score = (
        (frame["loan_amount"] / frame["income"]) * 3.0
        + frame["num_prior_defaults"] * 0.8
        + (680 - frame["credit_score"]) / 250.0
        + frame["loan_term_months"] / 120.0
        - frame["employment_years"] / 30.0
    )
    frame["default"] = (risk_score > risk_score.median()).astype(int)
    return frame


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    return build_fallback_data()


def clean_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    frame = df.copy()

    if "default" not in frame.columns:
        raise ValueError("The dataset must include a default column.")

    if "purpose" in frame.columns:
        frame = frame.drop(columns=["purpose"])

    if "home_ownership" not in frame.columns:
        raise ValueError("The dataset must include a home_ownership column.")

    feature_columns = [column for column in frame.columns if column != "default"]
    numeric_columns = [column for column in feature_columns if column != "home_ownership"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame["default"] = pd.to_numeric(frame["default"], errors="coerce")
    frame = frame.dropna(subset=feature_columns + ["default"])

    home_encoder = LabelEncoder()
    frame["home_ownership"] = home_encoder.fit_transform(frame["home_ownership"].astype(str))

    X = frame.drop(columns=["default"])
    std = X.std(ddof=0).replace(0, 1)
    z_scores = ((X - X.mean()) / std).abs()
    frame = frame[(z_scores < 3).all(axis=1)]

    return frame.reset_index(drop=True), home_encoder


@st.cache_resource(show_spinner=True)
def train_models() -> dict:
    raw_df = load_data()
    cleaned_df, home_encoder = clean_training_frame(raw_df)

    X = cleaned_df.drop(columns=["default"])
    y = cleaned_df["default"].astype(int)

    if y.nunique() < 2:
        raise ValueError("The dataset needs at least two target classes in default.")

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_scaled, y_train)
    logistic_pred = logistic_model.predict(X_test_scaled)

    svm_model = SVC(probability=True, kernel="rbf", C=3, gamma="scale")
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)

    numeric_ranges = {
        column: (
            int(cleaned_df[column].min()),
            int(cleaned_df[column].max()),
            max(1, int((cleaned_df[column].max() - cleaned_df[column].min()) // 100 or 1)),
            int(cleaned_df[column].median()),
        )
        for column in [
            "age",
            "income",
            "loan_amount",
            "loan_term_months",
            "credit_score",
            "employment_years",
            "num_prior_defaults",
        ]
        if column in cleaned_df.columns
    }

    return {
        "cleaned_df": cleaned_df,
        "feature_columns": list(X.columns),
        "home_encoder": home_encoder,
        "home_choices": home_encoder.classes_.astype(str).tolist(),
        "purpose_choices": sorted(raw_df["purpose"].astype(str).unique().tolist()) if "purpose" in raw_df.columns else DEFAULT_PURPOSE_CHOICES,
        "scaler": scaler,
        "logistic_accuracy": accuracy_score(y_test, logistic_pred),
        "svm_model": svm_model,
        "svm_accuracy": accuracy_score(y_test, svm_pred),
        "svm_report": classification_report(y_test, svm_pred, zero_division=0),
        "svm_confusion_matrix": confusion_matrix(y_test, svm_pred),
        "best_params": {"C": 3, "gamma": "scale", "kernel": "rbf"},
        "numeric_ranges": numeric_ranges,
    }


def blank_number_input(column: str, label: str | None = None) -> int | None:
    field_label = label or column.replace("_", " ").title()
    raw_value = st.text_input(field_label, placeholder=f"Enter {field_label}")
    if not raw_value.strip():
        return None

    try:
        return int(raw_value)
    except ValueError:
        st.error(f"{field_label} must be a whole number.")
        return None


def clamp(value: float, lower: int, upper: int) -> int:
    return int(max(lower, min(upper, round(value))))


def estimate_credit_score(
    age: int,
    income: int,
    loan_amount: int,
    loan_term_months: int,
    employment_years: int,
    home_ownership: str,
) -> int:
    home_bonus = {"rent": -25, "mortgage": -10, "own": 10}.get(home_ownership, 0)
    debt_pressure = (loan_amount / max(income, 1)) * 140
    age_factor = max(0, 40 - age) * 2
    employment_bonus = employment_years * 4
    term_pressure = loan_term_months * 0.8
    estimated = 820 - debt_pressure - age_factor - term_pressure + employment_bonus + home_bonus
    return clamp(estimated, 338, 850)


def estimate_num_prior_defaults(
    credit_score: int,
    income: int,
    loan_amount: int,
    employment_years: int,
) -> int:
    risk = ((700 - credit_score) / 90) + ((loan_amount / max(income, 1)) * 2.5) - (employment_years / 12)
    return clamp(risk, 0, 3)
st.title(APP_TITLE)
st.write("Enter the borrower details below and predict default risk.")

try:
    trained = train_models()
except Exception as exc:
    st.error(f"Could not train the model: {exc}")
    st.stop()

st.subheader("Prediction inputs")

with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        age = blank_number_input("age")
        income = blank_number_input("income", "Income (LKR)")
        loan_amount = blank_number_input("loan_amount", "Loan Amount (LKR)")
        loan_term_months = blank_number_input("loan_term_months")

    with right:
        employment_years = blank_number_input("employment_years")
        home_ownership = st.selectbox("Home Ownership", trained["home_choices"] or DEFAULT_HOME_CHOICES, index=None, placeholder="Select one")
        purpose_choices = [PURPOSE_LABELS.get(choice, choice.replace("_", " ").title()) for choice in (trained["purpose_choices"] or DEFAULT_PURPOSE_CHOICES)]
        purpose = st.selectbox("Purpose", purpose_choices, index=None, placeholder="Select one")

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    required_fields = {
        "Age": age,
        "Income": income,
        "Loan amount": loan_amount,
        "Loan term months": loan_term_months,
        "Employment years": employment_years,
        "Home ownership": home_ownership,
        "Purpose": purpose,
    }
    missing_fields = [label for label, value in required_fields.items() if value in (None, "")]
    if missing_fields:
        st.error("Please fill in: " + ", ".join(missing_fields))
        st.stop()

    credit_score = estimate_credit_score(
        age=age,
        income=income,
        loan_amount=loan_amount,
        loan_term_months=loan_term_months,
        employment_years=employment_years,
        home_ownership=str(home_ownership),
    )
    num_prior_defaults = estimate_num_prior_defaults(
        credit_score=credit_score,
        income=income,
        loan_amount=loan_amount,
        employment_years=employment_years,
    )

    model_input = pd.DataFrame(
        {
            "age": [age],
            "income": [income],
            "loan_amount": [loan_amount],
            "loan_term_months": [loan_term_months],
            "credit_score": [credit_score],
            "employment_years": [employment_years],
            "num_prior_defaults": [num_prior_defaults],
            "home_ownership": [trained["home_encoder"].transform([str(home_ownership)])[0]],
        }
    )

    scaled_input = trained["scaler"].transform(model_input)
    probability = float(trained["svm_model"].predict_proba(scaled_input)[0][1])
    prediction = int(trained["svm_model"].predict(scaled_input)[0])
    risk_label = "Default likely" if prediction == 1 else "Default unlikely"

    st.success(f"Prediction: {risk_label}")
    st.metric("Predicted default probability", f"{probability:.1%}")
    st.write(f"Estimated prior defaults: {num_prior_defaults}")

    result_left, result_right = st.columns(2)
    result_left.write("Classification report")
    result_left.code(trained["svm_report"])
    result_right.write("Confusion matrix")
    result_right.dataframe(
        pd.DataFrame(
            trained["svm_confusion_matrix"],
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"],
        )
    )

st.subheader("Model summary")
summary_left, summary_right = st.columns(2)

with summary_left:
    with st.container(border=True):
        st.metric("Logistic Regression Accuracy", f"{trained['logistic_accuracy']:.3f}")

with summary_right:
    with st.container(border=True):
        st.metric("SVM Accuracy", f"{trained['svm_accuracy']:.3f}")
