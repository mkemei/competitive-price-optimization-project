import pandas as pd
import numpy as np
import os
import joblib
import holidays
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from utils import  PreprocessingPipeline
# ==========================================
# 1. PREPROCESSING PIPELINE CLASS
# ==========================================
# This class must be defined here so it can be pickled into pipeline.pkl
# ==========================================
# 2. DATA PREPARATION
# ==========================================
def prepare_weekly_data(df_original):
    df = df_original.copy()
   
    # 1. Ensure correct types
    df['date'] = pd.to_datetime(df['date'])
    

    # 2. Standardize to Weekly (Mondays)
    df['week_start'] = df['date'].dt.to_period('W').dt.start_time

    # 3. Aggregation Dictionary
    agg_dict = {
        'net_quantity': 'sum',
        'unit_price': 'mean',
        'historical_average_competitor_price': 'mean'
    }

    # Group by both Product and the new Week column
    df_weekly = df.groupby(['product', 'product_type', 'week_start']).agg(agg_dict).reset_index()
    df_weekly = df_weekly.rename(columns={'week_start': 'date'}).sort_values(['product', 'date'])

    # 4. Feature Engineering
    df_weekly['year'] = df_weekly['date'].dt.year
    df_weekly['week_of_year'] = df_weekly['date'].dt.isocalendar().week.astype(int)

    ke_holidays = holidays.Kenya(years=[2020, 2021, 2022, 2023, 2024])
    def get_holiday_features(dt):
        h_name = ke_holidays.get(dt)
        return h_name
    df_weekly['holiday_name'] = df_weekly['date'].apply(get_holiday_features)
    df_weekly['is_holiday'] = df_weekly['holiday_name'].notnull().astype(int)

    # 4. Create the "Pre-Holiday" flag (Day before)

    df_weekly['is_pre_holiday'] = df_weekly['date'].apply(lambda x: 1 if (x + pd.Timedelta(days=1)) in ke_holidays else 0)
  
    df_weekly = df_weekly.drop(columns=['holiday_name'])
    
    # Lagged features (Look-back)
    grouped = df_weekly.groupby('product')
    for i in [1]:
        df_weekly[f'net_quantity_lag_{i}'] = grouped['net_quantity'].shift(i)
        df_weekly[f'unit_price_lag_{i}'] = grouped['unit_price'].shift(i)

    # Competitor Price Gap (How much cheaper/more expensive were we than the average competitor?)
    df_weekly['competitor_gap'] = df_weekly['unit_price'] - df_weekly['historical_average_competitor_price']

    # Rolling 4-week mean to capture recent momentum
    for i in [4]:
       df_weekly[f'rolling_mean_net_quantity_{i}w'] = grouped['net_quantity'].transform(lambda x: x.shift(1).rolling(window=i, min_periods=1).mean())
       df_weekly[f'rolling_median_net_quantity_{i}w'] = grouped['net_quantity'].transform(lambda x: x.shift(1).rolling(window=i, min_periods=1).median())
	  
    # 2. Price vs Own History     
    df_weekly['price_vs_avg_ratio'] = df_weekly['unit_price'] / (grouped['unit_price'].transform('mean') + 1e-6)    

    # We fill NaNs using the median of that SPECIFIC product.  
    cols_to_fill = [col for col in df_weekly.columns if 'lag' in col or 'rolling' in col or 'gap' in col]
    
    for col in cols_to_fill:
        df_weekly[col] = df_weekly.groupby('product')[col].transform(lambda x: x.fillna(x.median() if not x.isna().all() else 0))

    # Final fallback for products with only 1 row of data
    df_weekly = df_weekly.fillna(0)
    
    return df_weekly

# ==========================================
# 3. EXECUTION SCRIPT
# ==========================================
def run_training_pipeline():
    print("Starting 2026 Weekly Model Training...")
    
    # Load Data
    data_path = "data/cleaned/sales.csv"
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found.")
        return

    df_raw = pd.read_csv(data_path)
    df_weekly = prepare_weekly_data(df_raw)
    
    # Chronological Split
    df_weekly = df_weekly.sort_values('date')
    split_idx = int(len(df_weekly) * 0.8)
    train_df = df_weekly.iloc[:split_idx]
    test_df = df_weekly.iloc[split_idx:]

    # Define X and y (Log target)
    X_train_raw = train_df.drop(columns=['net_quantity', 'date'])
    y_train = np.log1p(train_df['net_quantity'])
    
    X_test_raw = test_df.drop(columns=['net_quantity', 'date'])
    y_test = np.log1p(test_df['net_quantity'])

    # Identify columns for Pipeline
    categorical_cols = X_train_raw.select_dtypes(include=['object']).columns
    numerical_cols = X_train_raw.select_dtypes(exclude=['object']).columns

    # Fit OneHotEncoder
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(X_train_raw[categorical_cols])

    # Transform data for training
    X_train_cat = pd.DataFrame(ohe.transform(X_train_raw[categorical_cols]), 
                               columns=ohe.get_feature_names_out(categorical_cols), index=X_train_raw.index)
    X_train_final = pd.concat([X_train_raw[numerical_cols], X_train_cat], axis=1)

    # Initialize Pipeline Object
    pipeline = PreprocessingPipeline(
        ohe_fitted_instance=ohe,
        numerical_cols_train=numerical_cols.tolist(),
        categorical_cols_train=categorical_cols.tolist(),
        final_column_order=X_train_final.columns.tolist()
    )

    # Random Forest Tuning
    print("Tuning Random Forest...")
    rf_param_dist = {
        'n_estimators': [100, 300, 500, 800],
        'max_depth': [10, 20, 30, 40, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
  
    tscv = TimeSeriesSplit(n_splits=5)


    search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=rf_param_dist,
    n_iter=50, 
    cv=tscv, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, 
    verbose=1,
    random_state=42
)
  
    search.fit(X_train_final, y_train)
    best_model = search.best_estimator_

    # Save Artifacts
    print("Saving model bundle to /models...")
    os.makedirs("models", exist_ok=True)

    pipeline = PreprocessingPipeline(ohe, numerical_cols.tolist(), categorical_cols.tolist(), X_train_final.columns.tolist())
    
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(pipeline, "models/pipeline.pkl")
    
    print(f"Training Complete. Best Params: {search.best_params_}")
    return True

if __name__ == "__main__":
    run_training_pipeline()