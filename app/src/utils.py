# app/src/utils.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, ohe_fitted_instance, numerical_cols_train, categorical_cols_train, final_column_order):
        self.ohe = ohe_fitted_instance
        self.numerical_cols_train = numerical_cols_train
        self.categorical_cols_train = categorical_cols_train
        self.final_column_order = final_column_order
        self.encoded_feature_names = self.ohe.get_feature_names_out(self.categorical_cols_train)

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        if isinstance(X, (dict, pd.Series)):
            X = pd.DataFrame([X])
        
        X_copy = X.copy()
        X_copy = X_copy.drop(columns=[col for col in ['date', 'net_quantity'] if col in X_copy.columns], errors='ignore')

        # Ensure columns exist before copying
        existing_cat = [c for c in self.categorical_cols_train if c in X_copy.columns]
        existing_num = [c for c in self.numerical_cols_train if c in X_copy.columns]

        X_cat = X_copy[existing_cat].copy()
        X_num = X_copy[existing_num].copy()
        
        X_cat_encoded = self.ohe.transform(X_cat)
        X_cat_df = pd.DataFrame(X_cat_encoded, columns=self.encoded_feature_names, index=X_copy.index)
        
        processed_df = pd.concat([X_num, X_cat_df], axis=1)
        final_df = processed_df.reindex(columns=self.final_column_order, fill_value=0)
        return final_df