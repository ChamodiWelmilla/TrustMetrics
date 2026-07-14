# 🛡️ TrustMetrics Risk Analyzer

A credit risk prediction and analytics dashboard built with Python and Streamlit. This application evaluates applicant financial profiles and estimates loan default probabilities using an advanced, balanced Support Vector Machine (SVM) pipeline.

## ✨ Features

- **Modern UI**: A sleek, dark-mode dashboard providing a highly polished user experience.
- **Dynamic Credit Scoring**: Automatically estimates a FICO-style credit score based on income, employment history, prior unpaid debts, and loan-to-income ratios—removing the need for applicants to know their exact score.
- **Robust Machine Learning Pipeline**: 
  - Trained on the `credit_risk_synthetic.csv` dataset.
  - Automatically handles class imbalances using balanced class weights (`class_weight='balanced'`) to prevent bias.
  - Strict data leakage prevention using `scikit-learn` Pipelines (scaling is done appropriately during cross-validation).
- **Interactive Insights**: Provides real-time insights and warnings explaining *why* an applicant was flagged (e.g., low income-to-loan ratio, previous unpaid debts).

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

### 3. Run the Dashboard

The machine learning model (`credit_risk_model.pkl`) is already pre-trained and included in the repository. You can launch the Streamlit server directly:

```bash
streamlit run app.py
```

Open the `Local URL` (typically `http://localhost:8501`) shown in your terminal to interact with the Risk Analyzer!

*(Optional)* If you wish to retrain the model from scratch on new data, you can run `python train_model.py` before launching the app.

## 🧰 Project Structure
- `app.py`: The main Streamlit web application.
- `train_model.py`: Script to preprocess data and train the ML model.
- `TrustMetrics.ipynb`: Jupyter Notebook documenting the exploratory data analysis, evaluation fixes, and hyperparameter tuning.
- `credit_risk_synthetic.csv`: The core synthetic dataset used for training.