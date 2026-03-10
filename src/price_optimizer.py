# optimizer.py
import numpy as np
import pandas as pd


class PriceOptimizer:
    """
    Retail price optimization engine.

    Can optimize a single product (row) or a dataset (latest per product).
    """

    def __init__(self, model, preprocess):
        """
        model: trained ML model (e.g., XGBoost)
        preprocess: preprocessing pipeline for model features
        """
        self.model = model
        self.preprocess = preprocess

        # Configurable parameters
        self.MAX_DISCOUNT = 0.10
        self.MAX_PREMIUM = 0.08
        self.COGS_RATIO = 0.70
        self.MAX_LIFT_CAP = 0.15
        self.PRICE_COL = "unit_price"

    def optimize_product(self, row):
        """
        Optimize a single product (row from dataframe).
        Returns a dict with optimal price, expected demand, and expected profit.
        """

        base_price = row[self.PRICE_COL]

        # 15-point price grid around current price
        price_grid = np.linspace(
            base_price * (1 - self.MAX_DISCOUNT),
            base_price * (1 + self.MAX_PREMIUM),
            15
        )

        # Generate scenarios
        scenarios = []
        for price in price_grid:
            scenario = row.copy()
            scenario[self.PRICE_COL] = price
            scenarios.append(scenario)

        scenario_df = pd.DataFrame(scenarios)

        # Preprocess and predict demand
        processed = self.preprocess.transform(scenario_df)
        demand = np.maximum(0.5, self.model.predict(processed))

        # Calculate profits
        cost = base_price * self.COGS_RATIO
        profits = demand * (price_grid - cost)

        # Pick best price
        best_idx = np.argmax(profits)

        return {
            "optimal_price": price_grid[best_idx],
            "expected_demand": demand[best_idx],
            "expected_profit": profits[best_idx],
        }

    def optimize_dataset(self, df):
        """
        Optimize all products in a dataframe (latest date per product).
        Returns a dataframe of recommendations.
        """

        latest_data = df.loc[df.groupby("product")["date"].idxmax()]
        results = []

        for _, row in latest_data.iterrows():
            result = self.optimize_product(row)
            results.append({
                "product": row["product"],
                "current_price": row[self.PRICE_COL],
                **result
            })

        return pd.DataFrame(results)


# Example usage:
if __name__ == "__main__":
    import joblib
    import pickle

    # Load trained model & preprocessing
    model = joblib.load("models/best_model.pkl")
    preprocess = joblib.load("models/feature_columns.pkl")  # replace with actual preprocessing

    # Instantiate optimizer
    optimizer = PriceOptimizer(model, preprocess)

    # Save as pickle
    with open("models/price_optimizer.pkl", "wb") as f:
        pickle.dump(optimizer, f)

    print("✅ PriceOptimizer pickled successfully.")