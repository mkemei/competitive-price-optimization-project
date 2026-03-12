# DaPri: XGBoost Price Optimization Platform 🇰🇪

DaPri is an intelligent price optimization system designed specifically for Kenyan retail SMEs. It utilizes an **XGBoost Gradient Boosting** regressor to predict demand elasticity and prescribes optimal price points to maximize revenue while maintaining competitive market positioning.

## 🚀 Key Features
* **Predictive Demand Modeling:** Achievement of **0.93 R²** and **2.66% MAPE** using XGBoost.
* **Automated Competitor Tracking:** Integrates with Shopify-based retail APIs to monitor live prices.
* **Grid-Search Optimization:** Simulates thousands of price scenarios to find the revenue-maximizing point.
* **Admin Dashboard:** Includes Model Explainability (SHAP), performance monitoring (MAE/RMSE), and RBAC management.
* **Streamlit UI:** An intuitive interface for retail managers to manage inventory pricing.

## 🏗️ System Architecture

##Dummy ReadMe to be updated

```mermaid
%%{init: {"flowchart": {"curve": "step"}}}%%
flowchart TB
    subgraph DataCollection[Data Collection]
        Scraper[Competitor Scraper]
        POS[Internal POS CSVs]
    end

    subgraph DataLayer[Data Layer]
        Ingest[Data Ingestion]
        DS[(Datastore & .pkl Artifacts)]
    end

    subgraph IntelligenceLayer[Intelligence Layer]
        FE[Feature Engineering]
        XGB{XGBoost Engine}
        OPT[Optimization Logic]
    end

    UI[Streamlit Dashboard]

    Scraper & POS --> Ingest --> DS
    DS --> FE --> XGB --> OPT --> UI
