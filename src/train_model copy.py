import os
import joblib
from xgboost import XGBRegressor
import pandas as pd

print("Splitting data into training and testing sets chronologically...")

# Ensure date is in datetime format
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Use the last 20% of the data as the test set
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df.drop(columns=['net_quantity', 'date'])
y_train = train_df['net_quantity']
X_test = test_df.drop(columns=['net_quantity', 'date'])
y_test = test_df['net_quantity']

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

print("Data splitting completed successfully.")


print("Starting Feature Preprocessing (One-Hot Encoding without Scaling)")

# 1. Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(exclude=['object']).columns

# 2. Instantiate and Fit OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# handle_unknown='ignore' ensures test set rows with new categories get all zeros
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(X_train[categorical_cols])

# 3. Transform categorical columns
X_train_cat_encoded = ohe.transform(X_train[categorical_cols])
X_test_cat_encoded = ohe.transform(X_test[categorical_cols])

# 4. Create DataFrames for the encoded features
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)

X_train_cat_df = pd.DataFrame(X_train_cat_encoded, columns=encoded_feature_names, index=X_train.index)
X_test_cat_df = pd.DataFrame(X_test_cat_encoded, columns=encoded_feature_names, index=X_test.index)

# 5. Combine Numerical (original) and Categorical (encoded)
# We take the numerical columns directly from X_train/X_test without scaling
X_train_processed = pd.concat([X_train[numerical_cols], X_train_cat_df], axis=1)
X_test_processed = pd.concat([X_test[numerical_cols], X_test_cat_df], axis=1)

print(f"\nFinal Shape of X_train_processed: {X_train_processed.shape}")
print(f"Final Shape of X_test_processed: {X_test_processed.shape}")

print("\nOne-hot encoding completed successfully.")

print("Training and evaluating XGBoost Regressor...")

# 1. Import necessary libraries
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Instantiate an XGBRegressor model
xgb_model = XGBRegressor(random_state=42)

# 3. Train the XGBoost model
xgb_model.fit(X_train_processed, y_train)

# 4. Make predictions on X_test_processed
y_pred_xgb = xgb_model.predict(X_test_processed)

# 5. Calculate MAE
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

# 6. Calculate MSE
mse_xgb = mean_squared_error(y_test, y_pred_xgb)

# 7. Calculate RMSE
rmse_xgb = np.sqrt(mse_xgb)

# 8. Calculate MAPE
# Avoid division by zero for y_test values that are 0
y_test_for_mape = y_test.copy()
y_test_for_mape[y_test_for_mape == 0] = np.nan # Replace 0 with NaN
mape_xgb = np.nanmean(np.abs((y_test_for_mape - y_pred_xgb) / y_test_for_mape)) * 100

# 9. Calculate R2 Score
r2_xgb = r2_score(y_test, y_pred_xgb)

# 10. Print all calculated metrics
print(f"\nXGBoost Regression Metrics:")
print(f"  MAE: {mae_xgb:.4f}")
print(f"  MSE: {mse_xgb:.4f}")
print(f"  RMSE: {rmse_xgb:.4f}")
print(f"  MAPE: {mape_xgb:.4f}%")
print(f"  R2 Score: {r2_xgb:.4f}")

Best parameters for XGBoost: {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 200}


# 11. Create a scatter plot to visualize actual vs. predicted values
plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_test, y=y_pred_xgb, alpha=0.6)

# 12. Add a diagonal line representing perfect predictions
min_val = min(y_test.min(), y_pred_xgb.min())
max_val = max(y_test.max(), y_pred_xgb.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

# 13. Label axes and add title
plt.title('XGBoost: Actual vs. Predicted Values')
plt.xlabel('Actual Values (net_quantity)')
plt.ylabel('Predicted Values (net_quantity)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("XGBoost Regressor training and evaluation completed.")


def train_model(X_train, y_train):
    model = XGBRegressor(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=50,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model.pkl")
    print("Model saved successfully")

if __name__ == "__main__":
    # Dummy training data
    X_train = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1]
    })
    y_train = pd.Series([10, 20, 30, 40, 50])

    train_model(X_train, y_train)
