import pandas as pd

df = pd.read_csv('credit_risk_synthetic.csv')

# Drop string columns for correlation
df_numeric = df.select_dtypes(include=['number'])

print("\n--- Correlations with Default ---")
print(df_numeric.corr()['default'].sort_values())

print("\n--- Average Values by Default Status ---")
print(df.groupby('default').mean(numeric_only=True)[['age', 'income', 'loan_amount', 'credit_score']])
