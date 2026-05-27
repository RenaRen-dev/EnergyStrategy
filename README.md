# Energy Commodity Data Pipeline (Snowflake ELT)

A high-performance, Snowflake-native ELT pipeline designed to process and normalize **162.6 Million rows** of energy commodity price data across **6,345 unique products** for Time-Series Foundation Models (e.g., Chronos).

## 🚀 High-Level Architecture

The pipeline follows a modern ELT (Extract-Load-Transform) pattern, leveraging Snowflake's compute for heavy-lift transformations while using Python for orchestration and external data ingestion (FRED).

### **1. Data Decoding (Parsing Phase)**
*   **Problem**: Raw data contains unstructured `DESCRIPTION` strings with 8+ different market dialects (DTN Rack, NYMEX Financial, ICE, Singapore Physical, etc.).
*   **Solution**: A vectorized Snowflake Python UDF (`parse_description_batch`) that uses complex regex patterns to decode descriptions into 6 structured dimensions:
    - `PRODUCT`, `GRADE`, `GEOGRAPHY`, `DELIVERY`, `TIMING`, `IS_SPOT`.
*   **Scale**: Decodes 163M rows in ~3 minutes using Snowflake's parallel processing.

### **2. Financial Normalization (FX Phase)**
*   **Data Source**: Federal Reserve Economic Data (FRED).
*   **Currencies**: Unified to **USD**.
*   **Historical Accuracy**: Uses daily historical exchange rates for `CAD` (DEXCAUS) and `EUR` (DEXUSEU).
*   **Euro Proxy**: Implements a programmatic synthetic Euro for pre-1999 data using German Mark (`EXGEUS`) and the fixed `1.95583` conversion rate.

### **3. Physical Normalization (UOM Phase)**
*   **Unit**: Unified to **US Gallons (GAL)**.
*   **Deterministic Logic**:
    - **Liters**: Unified via standard `3.78541` multiplier.
    - **Barrels**: Unified via standard `42.0` multiplier.
    - **Metric Tons (MT)**: Unified via product-specific gravity lookup (e.g., Gasoline = 8.5 BBL/MT, Diesel = 7.45 BBL/MT).

### **4. Statistical Normalization (ML-Ready Phase)**
*   **Hard Physical Cap**: Removes extreme outliers (domain knowledge: -$5 to $50/GAL) that poison statistical distributions.
*   **Winsorization**: Applies a 1% / 99% percentile cap per product to handle heavy-tailed financial distributions.
*   **Rolling Z-Scores**: Trailing 256-day standard scaling to produce stationary features in the [-3, +3] range, optimized for Time-Series Foundation Models.

---

## 🛠️ Technology Stack

- **Data Warehouse**: Snowflake (SQL + Python UDFs)
- **External Data**: St. Louis Fed (FRED API)
- **Orchestration**: Python 3.12 (Argparse, Snowflake-Connector)
- **Data Science**: Pandas, Matplotlib, Seaborn (for Auditing & FX)

---

## 📁 Project Structure

```
Energy-Strategy/
├── commodity/
│   ├── pipeline/
│   │   ├── main.py                # Single entry point orchestrator
│   │   ├── data_decoding.py       # Orchestrates the UDF parsing phase
│   │   └── normalization.py       # Orchestrates FX, Physical, and Statistical scaling
│   │
│   ├── utility/
│   │   ├── snowflake_client.py    # Snowflake connection (key-pair auth)
│   │   ├── fx_client.py           # FRED API & Snowflake write_pandas logic
│   │   ├── normalization_sql.py   # Physical scaling logic
│   │   ├── revin_sql.py           # Winsorization & Z-Score logic
│   │   └── parse_description_udf.py # Core regex parsing logic
│   │
│   ├── tests/                     # Unit & integration tests
│   ├── analysis/
│   │   └── EDA_analysis.ipynb     # ML-Ready Data Audit & Visualization
│   ├── backtest/                  # Backtesting engine
│   └── ml/                        # ML inference & Chronos baseline
│
├── refiner/                       # Refiner trading strategy module
├── requirements.txt               # Minimalist dependencies
└── README.md                      # Project documentation
```

---

## 🚦 Quick Start

### 1. **Configure Environment**
Ensure you have a `.env` file with your Snowflake credentials and an RSA private key in `api_key/`.

### 2. **Run the Pipeline**
The pipeline is fully modular. You can run the entire workflow or skip specific phases:

```bash
# Run everything (Decoding + Normalization)
python -m pipeline.main

# Skip the 163M row decoding if already done
python -m pipeline.main --skip-decoding

# Skip FX ingestion (use existing FX table)
python -m pipeline.main --skip-fx-ingest
```

### 3. **Verify Data Quality**
Open `analysis/EDA_analysis.ipynb` to run the ML-Ready Audit. This notebook provides:
- Unification confirmation (USD/GAL unique counts).
- Z-score distribution auditing (Mean ~0, Std ~1).
- Time-series stationarity visualization.

---

## 📈 Roadmap (Complete)
- [x] **Data Decoding**: 163M rows parsed via Snowflake UDF.
- [x] **FX Integration**: FRED historical rates for USD/EUR/CAD.
- [x] **Physical Unification**: All units to GAL.
- [x] **Statistical Scaling**: Winsorization & Rolling Z-Scores.

---

**Status**: ✅ Production Ready (ML-Ready Dataset Created)  
**Version**: 3.0  
**Last Updated**: 2026-05-01
