import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_model(data_path="data/cleaned/final_training_sales_data.csv"):

    print("Loading dataset...")
    df = pd.read_csv(data_path)

    # Ensure datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Chronological split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df.drop(columns=['net_quantity', 'date'])
    y_train = train_df['net_quantity']

    X_test = test_df.drop(columns=['net_quantity', 'date'])
    y_test = test_df['net_quantity']

    print("Preprocessing features...")

    categorical_cols = X_train.select_dtypes(include=['object']).columns
    numerical_cols = X_train.select_dtypes(exclude=['object']).columns

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(X_train[categorical_cols])

    X_train_cat = ohe.transform(X_train[categorical_cols])
    X_test_cat = ohe.transform(X_test[categorical_cols])

    encoded_feature_names = ohe.get_feature_names_out(categorical_cols)

    X_train_cat_df = pd.DataFrame(
        X_train_cat,
        columns=encoded_feature_names,
        index=X_train.index
    )

    X_test_cat_df = pd.DataFrame(
        X_test_cat,
        columns=encoded_feature_names,
        index=X_test.index
    )

    X_train_processed = pd.concat(
        [X_train[numerical_cols], X_train_cat_df],
        axis=1
    )

    X_test_processed = pd.concat(
        [X_test[numerical_cols], X_test_cat_df],
        axis=1
    )

    print("Training XGBoost model...")

    model = XGBRegressor(
        max_depth=3,
        learning_rate=0.2,
        n_estimators=200,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train_processed, y_train)

    # Evaluate
    y_pred = model.predict(X_test_processed)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    # Save artifacts
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/best_model.pkl")
    joblib.dump(ohe, "models/encoder.pkl")
    joblib.dump(X_train_processed.columns.tolist(), "models/feature_columns.pkl")

    print("\nModel and preprocessing artifacts saved successfully.")


if __name__ == "__main__":
    train_model()