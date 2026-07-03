# Trust Metrics Credit Risk Predictor

A credit risk prediction app built with Streamlit.

## Features

- Simple form-based UI for borrower details
- Automatic internal estimation of `credit_score` and `num_prior_defaults`
- Loads `dataset.csv` if it is present in the project folder
- Falls back to a built-in synthetic dataset if `dataset.csv` is missing
- Shows model accuracy cards for Logistic Regression and SVM

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd TrustMetrics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

If you want to use your own data, place the file here:

```text
dataset.csv
```

If `dataset.csv` is not present, the app will train on a fallback synthetic dataset.

### 4. Run the app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal to use the app.