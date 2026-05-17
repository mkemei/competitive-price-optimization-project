import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import holidays
import matplotlib.pyplot as plt

class PriceOptimizerDefault:
    def __init__(self, model, pipeline, historical_df, train_price_stats, config=None):
        self.model = model
        self.pipeline = pipeline
        self.history = historical_df.copy()
        self.history['product_lookup'] = self.history['product'].astype(str).str.lower().str.strip()
        self.product_to_type = self.history.groupby('product_lookup')['product_type'].first().to_dict()
        self.train_price_stats = {str(k).lower().strip(): float(v) for k, v in train_price_stats.items()}
        self.ke_holidays = holidays.Kenya(years=list(range(2020, 2028)))

        default_config = {
            'max_discount': 0.20,
            'max_premium': 0.30,
            'demand_floor': 1,
            'min_current_price_pct': 0.70,
            'min_recommended_price_pct': 0.80,
            'max_recommended_price_pct': 1.30,
            'elasticity': 0.0,         # Set to > 0 ONLY if your ML model is price-blind
            'comp_elasticity': 0.0     # Set to > 0 ONLY if your ML model is competitor-blind
        }

        self.config = {**default_config, **(config or {})}

    def build_scenario_features(self, product, date, target_price, competitor_price=None):
        product_clean = str(product).lower().strip()
        product_type = self.product_to_type.get(product_clean)

        if product_type is None:
            raise ValueError(f"Product '{product}' or its product_type was not found in history.")

        hist = self.history[self.history['product_lookup'] == product_clean].copy()

        if hist.empty:
            raise ValueError(f"No historical data found for product '{product}'.")

        hist['date'] = pd.to_datetime(hist['date'], errors='coerce')
        hist = hist.dropna(subset=['date'])
        hist['week_start'] = hist['date'].dt.to_period('W').dt.start_time

        weekly_hist = hist.groupby('week_start').agg({'net_quantity': 'sum', 'unit_price': 'mean'}).sort_index()

        current_date = pd.to_datetime(date)
        historical_avg_price = self.train_price_stats.get(product_clean, target_price)
        effective_competitor_price = competitor_price if competitor_price is not None else target_price

        row = {
            'product': product_clean,
            'product_type': product_type,
            'unit_price': float(target_price),
            'historical_average_competitor_price': float(effective_competitor_price),
            'year': current_date.year,
            'week_of_year': int(current_date.isocalendar()[1]),
            'is_holiday': 1 if current_date.normalize() in self.ke_holidays else 0,
            'is_pre_holiday': 1 if (current_date.normalize() + pd.Timedelta(days=1)) in self.ke_holidays else 0,
            'net_quantity_lag_1': weekly_hist['net_quantity'].iloc[-1] if not weekly_hist.empty else 0,
            'unit_price_lag_1': weekly_hist['unit_price'].iloc[-1] if not weekly_hist.empty else float(target_price),
            'competitor_gap': float(target_price) - float(effective_competitor_price),
            'rolling_mean_net_quantity_4w': weekly_hist['net_quantity'].tail(4).mean() if not weekly_hist.empty else 0,
            'rolling_median_net_quantity_4w': weekly_hist['net_quantity'].tail(4).median() if not weekly_hist.empty else 0,
            'price_vs_avg_ratio': float(target_price) / (historical_avg_price + 1e-6),
           # 'discount_percentage': 0
        }

        return pd.DataFrame([row])

    def predict_demand(self, product_name, price, date=None, competitor_price=None):
        if date is None:
            date = pd.Timestamp.now()

        product_clean = str(product_name).lower().strip()
        price = float(price)

        if price <= 0:
            raise ValueError("Price must be greater than zero.")

        df_row = self.build_scenario_features(product_clean, date, price, competitor_price)
        processed_X = self.pipeline.transform(df_row)
        pred_log = self.model.predict(processed_X)[0]
        base_units = np.expm1(pred_log)

        base_price = self.train_price_stats.get(product_clean, price)
        elasticity = abs(self.config.get('elasticity', 0.0))
        comp_elasticity = self.config.get('comp_elasticity', 0.0)

        adjusted_units = base_units
        
        if elasticity > 0:
            adjusted_units *= ((base_price / price) ** elasticity)

        if comp_elasticity > 0 and competitor_price is not None and float(competitor_price) > 0:
            adjusted_units *= ((float(competitor_price) / price) ** comp_elasticity)

        return max(self.config.get('demand_floor', 1), adjusted_units)

    def optimize_product(self, product_name, current_price, competitor_price=None):
        product_clean = str(product_name).lower().strip()
        current_price = float(current_price)

        if current_price <= 0:
            raise ValueError("Current price must be greater than zero.")

        historical_avg = self.train_price_stats.get(product_clean)

        if historical_avg is None:
            historical_avg = current_price

        min_current_pct = self.config.get('min_current_price_pct', 0.70)
        min_recommended_pct = self.config.get('min_recommended_price_pct', 0.80)
        max_recommended_pct = self.config.get('max_recommended_price_pct', 1.30)

        minimum_allowed_current_price = min_current_pct * historical_avg

        if current_price < minimum_allowed_current_price:
            print(f"Warning: Current price KES {current_price:,.2f} for {product_clean} is below threshold boundaries.")

        #lower = max(current_price * (1 - self.config.get('max_discount', 0.20)), historical_avg * min_recommended_pct)
        lower = historical_avg * (1 - self.config.get('max_discount', 0.20))
        upper = min(current_price * (1 + self.config.get('max_premium', 0.30)), historical_avg * max_recommended_pct)

        if lower >= upper:
            lower = historical_avg * min_recommended_pct
            upper = historical_avg * max_recommended_pct

        def objective(p):
            demand = self.predict_demand(product_clean, float(p), competitor_price=competitor_price)
            revenue = demand * float(p)
            return -revenue

        res = minimize_scalar(objective, bounds=(lower, upper), method='bounded')
        opt_price = float(np.clip(res.x, lower, upper))

        demand_current = self.predict_demand(product_clean, current_price, competitor_price=competitor_price)
        demand_opt = self.predict_demand(product_clean, opt_price, competitor_price=competitor_price)

        current_revenue = demand_current * current_price
        optimized_revenue = demand_opt * opt_price
        revenue_lift_pct = ((optimized_revenue / (current_revenue + 1e-6)) - 1) * 100

        return {
            'product': product_name,
            'current_price': round(current_price, 2),
            'historical_avg_price': round(historical_avg, 2),
            'minimum_allowed_current_price': round(minimum_allowed_current_price, 2),
            'lower_bound': round(lower, 2),
            'upper_bound': round(upper, 2),
            'optimal_price': round(opt_price, 2),
            'current_demand': round(demand_current, 2),
            'expected_demand_opt': round(demand_opt, 2),
            'current_revenue': round(current_revenue, 2),
            'optimized_revenue': round(optimized_revenue, 2),
            'revenue_lift_pct': round(revenue_lift_pct, 2)
        }
