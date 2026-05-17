import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import holidays

class PriceOptimizerDefault:
    def __init__(self, model, pipeline, historical_df, train_price_stats, config=None):
        self.model = model
        self.pipeline = pipeline
        self.history = historical_df.copy()
        # Clean product names for consistent lookup
        self.history['product_lookup'] = self.history['product'].str.lower().str.strip()
        
        # Create a mapping of product to its type based on history
        self.product_to_type = self.history.groupby('product_lookup')['product_type'].first().to_dict()
        
        self.train_price_stats = train_price_stats
        # Updated to include 2026 as per your current timeframe
        self.ke_holidays = holidays.Kenya(years=list(range(2020, 2027)))
        self.config = config or {'max_discount': 0.20, 'max_premium': 0.30, 'demand_floor': 1}

    def build_scenario_features(self, product, date, target_price, competitor_price=None):
        product_clean = product.lower().strip()
        
        # 1. Retrieve the Product Type (Crucial for the OneHotEncoder in your pipeline)
        product_type = self.product_to_type.get(product_clean)
        if product_type is None:
            raise ValueError(f"Product '{product}' (or its type) not found in history.")

        # 2. Filter and aggregate history to weekly to match training
        hist = self.history[self.history['product_lookup'] == product_clean].copy()
        hist['week_start'] = pd.to_datetime(hist['date']).dt.to_period('W').dt.start_time
        weekly_hist = hist.groupby('week_start').agg({'net_quantity': 'sum', 'unit_price': 'mean'}).sort_index()
        
        current_date = pd.to_datetime(date)
        
        # 3. Build features exactly as defined in training (including product_type)
        row = {
            'product': product_clean,
            'product_type': product_type, # Added to match training columns
            'unit_price': target_price,
            'historical_average_competitor_price': competitor_price if competitor_price is not None else target_price,
            'year': current_date.year,
            'week_of_year': int(current_date.isocalendar()[1]),
            'is_holiday': 1 if current_date in self.ke_holidays else 0,
            'is_pre_holiday': 1 if (current_date + pd.Timedelta(days=1)) in self.ke_holidays else 0,
            'net_quantity_lag_1': weekly_hist['net_quantity'].iloc[-1] if not weekly_hist.empty else 0,
            'unit_price_lag_1': weekly_hist['unit_price'].iloc[-1] if not weekly_hist.empty else target_price,
            'competitor_gap': target_price - (competitor_price if competitor_price is not None else target_price),
            'rolling_mean_net_quantity_4w': weekly_hist['net_quantity'].tail(4).mean() if not weekly_hist.empty else 0,
            'rolling_median_net_quantity_4w': weekly_hist['net_quantity'].tail(4).median() if not weekly_hist.empty else 0,
            'price_vs_avg_ratio': target_price / (self.train_price_stats.get(product_clean, target_price) + 1e-6)
        }
        return pd.DataFrame([row])

    def predict_demand(self, product_name, price, date=pd.Timestamp.now(), competitor_price=None):
        df_row = self.build_scenario_features(product_name, date, price, competitor_price)
        # The pipeline will now see 'product_type' and encode it correctly
        processed_X = self.pipeline.transform(df_row)
        pred_log = self.model.predict(processed_X)[0]
        return max(self.config['demand_floor'], np.expm1(pred_log))

    def optimize_product(self, product_name, current_price, competitor_price=None):
        lower, upper = current_price * (1 - self.config['max_discount']), current_price * (1 + self.config['max_premium'])
        
        def objective(p):
            return -(self.predict_demand(product_name, float(p), competitor_price=competitor_price) * p)

        res = minimize_scalar(objective, bounds=(lower, upper), method='bounded')
        opt_price = float(res.x)
        
        demand_current = self.predict_demand(product_name, current_price, competitor_price=competitor_price)
        demand_opt = self.predict_demand(product_name, opt_price, competitor_price=competitor_price)
        
        return {
            'product': product_name,
            'current_price': round(current_price, 2),
            'optimal_price': round(opt_price, 2),
            'current_demand': round(demand_current, 2),
            'expected_demand_opt': round(demand_opt, 2),
            'revenue_lift_pct': round(((demand_opt * opt_price) / (demand_current * current_price + 1e-6) - 1) * 100, 2)
        }