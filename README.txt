```markdown
## Project Structure

```

price-optimization-system/
│
├── data/
│   ├── raw_sales.csv
│   ├── competitor_prices.csv
│   └── processed_data.csv
│
├── models/
│   └── trained_model.pkl
│
├── src/
│   ├── data_ingestion.py
│   ├── feature_eng.py
│   ├── train_model.py
│   ├── demand_prediction.py
│   ├── price_optimizer.py
│   └── utils.py
│
├── app/
│   └── dashboard.py
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── requirements.txt
└── README.md

```

### Key Modules

- **data_ingestion.py**  
  Handles loading and cleaning of raw sales data from the POS system.

- **feature_eng.py**  
  Performs **feature engineering** by generating additional variables such as:
  - lagged sales features
  - price-to-competitor ratios
  - rolling demand averages
  - price volatility indicators

  These engineered features improve the predictive performance of the machine learning model.

- **train_model.py**  
  Trains the machine learning model (XGBoost regression) using historical sales and engineered features.

- **demand_prediction.py**  
  Uses the trained model to forecast product demand for candidate price scenarios.

- **price_optimizer.py**  
  Evaluates different price points and determines the **optimal price that maximizes revenue or profit**.

- **dashboard.py**  
  Streamlit dashboard that allows users to:
  - select products
  - visualize analytics
  - view recommended prices

```
