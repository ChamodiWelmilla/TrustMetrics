import streamlit as st
import pandas as pd
import joblib
import time

st.set_page_config(
    page_title="TrustMetrics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #58a6ff !important;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #238636 0%, #2ea043 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease-in-out;
        width: 100%;
        box-shadow: 0 4px 15px rgba(46, 160, 67, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 160, 67, 0.6);
        background: linear-gradient(90deg, #2ea043 0%, #3fb950 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load('credit_risk_model.pkl')
    le = joblib.load('home_ownership_encoder.pkl')
    return model, le

try:
    model, le = load_assets()
except Exception as e:
    st.error(f"Model assets not found. Error: {str(e)}")
    st.stop()

# Header Section
st.title("🛡️ TrustMetrics")
st.markdown("Enter applicant data below to perform real-time credit default risk assessment using advanced SVM model.")
st.markdown("---")

# Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("👤 Applicant Profile")
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    income = st.number_input("Annual Income ($)", min_value=10000, max_value=1000000, value=60000, step=1000)
    employment_years = st.number_input("Years of Employment", min_value=0, max_value=50, value=5, step=1)
    
    home_ownership_options = list(le.classes_)
    home_ownership = st.selectbox("Home Ownership", home_ownership_options)

with col2:
    st.subheader("💳 Credit Information")
    loan_amount = st.number_input("Loan Amount Requested ($)", min_value=1000, max_value=2000000, value=25000, step=1000)
    loan_term_months = st.number_input("Loan Term (Months)", min_value=6, max_value=360, value=36, step=6)
    num_prior_defaults = st.number_input("Previous Unpaid Debts (Defaults)", min_value=0, max_value=20, value=0, step=1)

st.markdown("---")

# Calculate estimated credit score dynamically
base_score = 650
score_emp = min(employment_years * 8, 100)
score_def = min(num_prior_defaults * 75, 300)
score_inc = min((income / 10000) * 3, 50)
score_dti = min((loan_amount / income) * 20, 100) if income > 0 else 100

credit_score = int(base_score + score_emp - score_def + score_inc - score_dti)
credit_score = max(300, min(850, credit_score))

st.subheader("📈 Estimated Credit Profile")
st.info(f"Based on the provided applicant data (Income, Employment History, Prior Defaults, and requested Loan), the **Estimated Credit Score is: {credit_score}**.")

# Prediction action
if st.button("🚀 Analyze Risk Profile"):
    with st.spinner("Analyzing data through SVM pipeline..."):
        time.sleep(1) # Micro-animation effect
        
        # Prepare input
        home_ownership_encoded = le.transform([home_ownership])[0]
        
        input_data = pd.DataFrame({
            'age': [age],
            'income': [income],
            'loan_amount': [loan_amount],
            'loan_term_months': [loan_term_months],
            'credit_score': [credit_score],
            'employment_years': [employment_years],
            'num_prior_defaults': [num_prior_defaults],
            'home_ownership': [home_ownership_encoded]
        })
        
        # Predict
        prob = model.predict_proba(input_data)[0][1] # Probability of default
        pred = model.predict(input_data)[0]
        
        # Display Results
        st.subheader("📊 Assessment Results")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if pred == 1:
                st.error("🚨 **HIGH RISK**: Application flagged for potential default.")
            else:
                st.success("✅ **LOW RISK**: Application looks solid.")
                
            st.metric("Risk of Non-Payment (Default Likelihood)", f"{prob * 100:.2f}%")
            
        with res_col2:
            st.info("💡 **Insights**")
            if prob > 0.5:
                st.write("- Consider requesting additional collateral or a co-signer due to the elevated risk score.")
                if num_prior_defaults > 0:
                    st.write("- **Warning**: Prior defaults significantly increase risk.")
                if credit_score < 650:
                    st.write("- **Warning**: Low credit score is a major negative factor.")
            else:
                st.write("- Applicant exhibits strong trust metrics.")
                if income > loan_amount / 3:
                    st.write("- Good income-to-loan ratio.")
                if credit_score >= 720:
                    st.write("- Excellent credit history.")
