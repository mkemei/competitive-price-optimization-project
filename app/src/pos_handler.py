import pandas as pd
import os
from datetime import datetime

SALES_FILE = "data/cleaned/sales.csv"

def record_sale_to_csv(product_name, unit_price, net_quantity, discount_percentage):
    # 1. Fetch Latest 2026 Competitor Context
    comp_path = "data/competitor_prices/"
    comp_min, comp_avg = 0, 0
    
    if os.path.exists(comp_path):
        files = [f for f in os.listdir(comp_path) if f.endswith('.csv')]
        if files:
            # Get the most recent scrape file
            latest_file = sorted(files, reverse=True)[0]
            comp_df = pd.read_csv(os.path.join(comp_path, latest_file))
            
            # Match product
            match = comp_df[comp_df['matched_user_product_original'].str.lower() == product_name.lower()]
            if not match.empty:
                comp_min = match['min_price_across_sources'].values[0]
                comp_avg = match['avg_price_across_sources'].values[0]

    # 2. Create the new row for 2026
    new_row = {
        'product': product_name,
        'date': datetime.now().strftime("%Y-%m-%d"),
        'unit_price': unit_price,
        'net_quantity': net_quantity,
        'min_competitor_price': comp_min,
        'discount_percentage': discount_percentage
    }

    # 3. Append to CSV
    df_new = pd.DataFrame([new_row])
    # header=False if file exists so we don't write headers mid-file
    df_new.to_csv(SALES_FILE, mode='a', index=False, header=not os.path.exists(SALES_FILE))
    
    return True