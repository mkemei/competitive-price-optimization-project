import requests
import pandas as pd
import re
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup

# Ensure rapidfuzz is installed
try:
    from rapidfuzz import fuzz
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rapidfuzz"])
    from rapidfuzz import fuzz

# ==========================================================
# 1. CONFIGURATION
# ==========================================================
SHOPIFY_SITES = {
    "TotsShoppe": "https://totsshoppe.com",
    "Peekaboo": "https://peekaboo.ke"
}

JUMIA_CATEGORIES = {
    "Feeding": "https://www.jumia.co.ke/baby-feeding-foods/",
    "Diapering": "https://www.jumia.co.ke/baby-diapering/",
    "Bathing": "https://www.jumia.co.ke/baby-bathing-skin-care/"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

OUTPUT_PATH = "data/competitor_prices/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ==========================================================
# 2. SCRAPERS
# ==========================================================

def scrape_shopify(name, base_url):
    products = []
    print(f"📡 Scraping {name}...")
    page = 1

    while True:
        url = f"{base_url}/products.json?limit=250&page={page}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=15)
            if res.status_code != 200:
                break

            data = res.json().get("products", [])
            if not data:
                break

            for p in data:
                for v in p["variants"]:
                    try:
                        price = float(v["price"]) if v["price"] else None
                    except:
                        price = None

                    if price is not None:
                        products.append({
                            "source": name.lower().strip(),  # totsshoppe / peekaboo
                            "product_name": p["title"],
                            "brand": p.get("vendor", "N/A"),
                            "current_price": price
                        })

            page += 1

        except Exception as e:
            print(f"Error scraping {name}: {e}")
            break

    return products


def scrape_jumia_page(task_tuple):
    cat_name, base_url, page = task_tuple
    products = []

    url = f"{base_url}?page={page}"

    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(res.text, "html.parser")

        for art in soup.select('article.prd'):
            name_tag = art.select_one('.name')
            price_tag = art.select_one('.prc')

            if not name_tag or not price_tag:
                continue

            name = name_tag.text.strip()
            brand = art.get('data-brand', "N/A")
            price_raw = price_tag.text

            # Extract numeric value
            price_digits = ''.join(filter(str.isdigit, price_raw))
            if not price_digits:
                continue

            price = float(price_digits)

            products.append({
                "source": "jumia",
                "product_name": name,
                "brand": brand,
                "current_price": price
            })

    except Exception:
        pass

    return products

# ==========================================================
# 3. CLEANING
# ==========================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace('&', 'and')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # separate letters & numbers
    text = re.sub(r'(?<=\d)([a-z]+)', r' \1', text)
    text = re.sub(r'([a-z]+)(?=\d)', r'\1 ', text)

    return text

# ==========================================================
# 4. MAIN PIPELINE
# ==========================================================

def main(pos_csv_path):

    # -----------------------------
    # Load POS
    # -----------------------------
    try:
        pos_data = pd.read_csv(pos_csv_path)
        col = 'product' if 'product' in pos_data.columns else 'Product'
        user_products = pos_data[col].dropna().unique().tolist()
    except Exception as e:
        print(f"❌ Error loading POS file: {e}")
        return

    # -----------------------------
    # SCRAPING
    # -----------------------------
    all_data = []

    # Shopify
    for name, url in SHOPIFY_SITES.items():
        all_data.extend(scrape_shopify(name, url))

    # Jumia (parallel)
    print("📡 Scraping Jumia...")
    tasks = [(cat, url, p) for cat, url in JUMIA_CATEGORIES.items() for p in range(1, 6)]

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(scrape_jumia_page, tasks)
        for r in results:
            all_data.extend(r)

    df_raw = pd.DataFrame(all_data)

    if df_raw.empty:
        print("❌ No data scraped.")
        return

    # -----------------------------
    # MATCHING
    # -----------------------------
    print("🔗 Matching products...")

    df_raw["clean_name"] = df_raw["product_name"].apply(clean_text)
    user_map = {clean_text(p): p for p in user_products if isinstance(p, str)}

    matched = []

    for _, row in df_raw.iterrows():
        best_match = None
        best_score = 0

        for u_clean, original in user_map.items():
            score = fuzz.token_set_ratio(u_clean, row["clean_name"])

            if score >= 70 and score > best_score:
                best_score = score
                best_match = original

        if best_match:
            rec = row.to_dict()
            rec["matched_user_product_original"] = best_match
            matched.append(rec)

    if not matched:
        print("❌ No matches found.")
        return

    m_df = pd.DataFrame(matched)

    # -----------------------------
    # AGGREGATION (CORRECT LOGIC)
    # -----------------------------
    print("📊 Aggregating prices...")

    m_df['source'] = m_df['source'].str.lower().str.strip()

    # Source column mapping
    source_map = {
        'jumia': 'jumia_price',
        'totsshoppe': 'totshoppe_price',
        'peekaboo': 'peekaboo_price'
    }
    
    final_df = pd.DataFrame({'matched_user_product_original': m_df['matched_user_product_original'].unique()})
    # Source-specific min prices
    for src, col_name in source_map.items():
      src_df = (m_df[m_df['source'] == src].groupby('matched_user_product_original')['current_price'].min().reset_index().rename(columns={'current_price': col_name}))
      final_df = final_df.merge(src_df, on='matched_user_product_original', how='left')
    
    # Calculate Global Stats across the specific source columns
    price_cols = [c for c in source_map.values() if c in final_df.columns]
    
    if price_cols:
        final_df['min_price_across_sources'] = final_df[price_cols].min(axis=1)
        final_df['max_price_across_sources'] = final_df[price_cols].max(axis=1)
        final_df['avg_price_across_sources'] = final_df[price_cols].mean(axis=1).round(0)

    # -----------------------------
    # SAVE OUTPUT
    # -----------------------------
    output_file = os.path.join( OUTPUT_PATH, f"competitor_prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    final_df.to_csv(output_file, index=False)

    print("\n✅ SUCCESS")
    print(f"📁 File: {output_file}")
    print(f"📈 Products matched: {len(final_df)} / {len(user_products)}")

# ==========================================================
# 5. RUN
# ==========================================================

if __name__ == "__main__":
    main("data/pos_sales_data.csv")