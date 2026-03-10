# optimizer/price_optimizer.py
import numpy as np
import pandas as pd

class PriceOptimizer:
    """
    Retail price optimization engine.
    """

    def __init__(self, model, encoder, feature_columns):
        """
        model: trained ML model
        encoder: OneHotEncoder object for categorical features
        feature_columns: list of all final columns after encoding
        """
        self.model = model
        self.encoder = encoder
        self.feature_columns = feature_columns

        # Configuration
        self.MAX_DISCOUNT = 0.10
        self.MAX_PREMIUM = 0.08
        self.COGS_RATIO = 0.70
        self.MAX_LIFT_CAP = 0.15
        self.PRICE_COL = "unit_price"

    def preprocess(self, df):
        """Apply encoder and align columns for model prediction"""
        df = df.drop(columns=["date"], errors="ignore")

        # Ensure all categorical columns seen during training exist
        for col in self.encoder.feature_names_in_:
            if col not in df.columns:
                df[col] = "missing"  # Must be string, not 0
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        numerical_cols = df.select_dtypes(exclude=['object']).columns

        if len(categorical_cols) > 0:
            cat_encoded = self.encoder.transform(df[categorical_cols])
            cat_df = pd.DataFrame(
                cat_encoded, 
                columns=self.encoder.get_feature_names_out(categorical_cols), 
                index=df.index
            )
        else:
            cat_df = pd.DataFrame(index=df.index)

        processed_df = pd.concat([df[numerical_cols], cat_df], axis=1)
        # Reindex to ensure all columns exist
        processed_df = processed_df.reindex(columns=self.feature_columns, fill_value=0)
        return processed_df

    def optimize_product(self, row, product_name=None):
        """Optimize a single product"""
        if product_name is None:
            product_name = row.get("product", "unknown")  # fallback
        
        base_price = row[self.PRICE_COL]
        price_grid = np.linspace(
            base_price * (1 - self.MAX_DISCOUNT),
            base_price * (1 + self.MAX_PREMIUM),
            15
        )

        # Build scenarios
        scenarios = []
        for price in price_grid:
            scenario = row.copy()
            scenario[self.PRICE_COL] = price
            scenarios.append(scenario)

        scenario_df = pd.DataFrame(scenarios)
        X = self.preprocess(scenario_df)
        demand = np.maximum(0.5, self.model.predict(X))  # Minimum realistic demand

        cost = base_price * self.COGS_RATIO
        profits = demand * (price_grid - cost)
        best_idx = np.argmax(profits)

        return {
            "product": product_name,
            "current_price": base_price,
            "optimal_price": price_grid[best_idx],
            "expected_demand": demand[best_idx],
            "expected_profit": profits[best_idx],
        }

    def optimize_dataset(self, df):
        """Optimize all latest products"""
        # Take latest row per product
        if 'product' not in df.columns:
            df['product'] = "unknown"
        latest_data = df.loc[df.groupby("product")["date"].idxmax()]
        results = []
        for _, row in latest_data.iterrows():
            product_name = row.get("product", "unknown")
            result = self.optimize_product(row, product_name=product_name)
            results.append(result)
        return pd.DataFrame(results)