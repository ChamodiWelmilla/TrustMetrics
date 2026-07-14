import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

print("Loading data...")
df = pd.read_csv('credit_risk_synthetic.csv')

# Save mapping for purpose and home_ownership if we want to use them in UI
# Wait, the notebook dropped 'purpose'. Let's drop it here too.
df = df.drop(columns=['purpose'])

le = LabelEncoder()
df['home_ownership'] = le.fit_transform(df['home_ownership'])
home_ownership_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

X = df.drop('default', axis=1)
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Outlier removal on train
z_scores = np.abs((X_train - X_train.mean()) / X_train.std())
train_mask = (z_scores < 3).all(axis=1)

X_train = X_train[train_mask]
y_train = y_train[train_mask]

print("Training model...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, C=100, gamma='scale', kernel='linear', class_weight='balanced'))
])

pipeline.fit(X_train, y_train)

print("Saving model and encoders...")
joblib.dump(pipeline, 'credit_risk_model.pkl')
joblib.dump(le, 'home_ownership_encoder.pkl')

print("Done! Model saved as credit_risk_model.pkl")
