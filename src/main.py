# 📦 Imports
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("All libraries successfully imported.")

# Load Data
print("Loading datasets...")
train = pd.read_csv('../data/dataset.csv')
test = pd.read_csv('../data/test.csv')
sample_sub = pd.read_csv('../data/sample_submission.csv')
print(f"🔍 Data loaded. Train shape: {train.shape} | Test shape: {test.shape}")

# Target Variable
print("🎯 Extracting target variable from training data...")
y = train['sale_price']
X = train.drop(columns=['sale_price'])

# 🛠Preprocessing
print("🛠Encoding categorical features...")
X['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([X, test], axis=0)

# Encode all object-type columns using Label Encoding
for col in combined.select_dtypes(include='object').columns:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))
print("Categorical features successfully encoded.")

# Separate the combined dataset back into training and test sets
X = combined[combined['is_train'] == 1].drop(columns='is_train')
X_test = combined[combined['is_train'] == 0].drop(columns='is_train')

# Train/Validation Split
print("Splitting data into training and validation subsets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape} | Validation set shape: {X_val.shape}")

# LightGBM Quantile Model Training Function
def train_lgb_quantile(X_tr, y_tr, alpha, label):
    """
    Train a LightGBM quantile regression model.

    Parameters:
        X_tr (DataFrame): Training features.
        y_tr (Series): Training target.
        alpha (float): Quantile level (e.g., 0.05 or 0.95).
        label (str): Label to identify the model (used for logging).

    Returns:
        model (LGBMRegressor): Trained LightGBM model.
    """
    print(f"Training LightGBM model for '{label}' (quantile alpha = {alpha})...")
    params = {
        'objective': 'quantile',
        'alpha': alpha,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'n_estimators': 1000,
        'verbosity': -1
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr)
    print(f"Model training complete for '{label}'.")
    return model

# Train models for lower and upper quantiles
model_lower = train_lgb_quantile(X_train, y_train, alpha=0.05, label="Lower Bound")
model_upper = train_lgb_quantile(X_train, y_train, alpha=0.95, label="Upper Bound")

# Generate Predictions
print("Making predictions on the test dataset...")
pred_lower = model_lower.predict(X_test)
pred_upper = model_upper.predict(X_test)
print("Prediction generation complete.")

# Prepare Submission File
print("Creating submission file...")
submission = pd.DataFrame({
    'id': test['id'],
    'pi_lower': pred_lower,
    'pi_upper': pred_upper
})

submission.to_csv('../output/submission.csv', index=False)
print("submission.csv saved!")