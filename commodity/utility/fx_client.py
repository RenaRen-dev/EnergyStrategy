import pandas as pd
from snowflake.connector.pandas_tools import write_pandas

TARGET_TABLE = "CMDTYA.PUBLIC.FX_RATES_DAILY"

def fetch_fred_series(series_id: str) -> pd.Series:
    """Fetch a FRED series as a pandas Series directly via CSV."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, index_col='observation_date', parse_dates=True, na_values='.')
    return df[series_id]

def ingest_historical_fx(sf_client, min_date, max_date) -> bool:
    """
    Download FRED FX rates, calculate synthetic Euro for pre-1999, 
    forward-fill, and upload to Snowflake.
    """
    print("[FX] Downloading DEXCAUS, DEXUSEU, and EXGEUS from FRED...")
    cad_series = fetch_fred_series('DEXCAUS')
    eur_series = fetch_fred_series('DEXUSEU')
    dem_series = fetch_fred_series('EXGEUS')  # Monthly EXGEUS since daily DEXGUS is discontinued
    
    print("[FX] Cleaning, blending synthetic Euro, and forward-filling...")
    all_days = pd.date_range(start=min_date, end=max_date, freq='D')
    
    cad_series.index = cad_series.index.tz_localize(None)
    eur_series.index = eur_series.index.tz_localize(None)
    dem_series.index = dem_series.index.tz_localize(None)
    
    df_yf = pd.DataFrame(index=all_days)
    
    cad_reindexed = cad_series.reindex(all_days).ffill().bfill()
    eur_reindexed = eur_series.reindex(all_days).ffill()
    dem_reindexed = dem_series.reindex(all_days).ffill().bfill()
    
    df_yf['CAD_TO_USD'] = 1.0 / cad_reindexed
    
    # Synthetic Euro = 1.95583 / EXGEUS
    synthetic_eur = 1.95583 / dem_reindexed
    df_yf['EUR_TO_USD'] = eur_reindexed.combine_first(synthetic_eur)
    
    df_yf = df_yf.dropna()
    
    df_melted = df_yf[['EUR_TO_USD', 'CAD_TO_USD']].copy()
    df_melted.index.name = 'DATE'
    df_melted = df_melted.reset_index()
    
    records = []
    for _, row in df_melted.iterrows():
        d = row['DATE'].date()
        records.append((d, 'EUR', row['EUR_TO_USD']))
        records.append((d, 'CAD', row['CAD_TO_USD']))
        
    df_final = pd.DataFrame(records, columns=['DATE', 'CURRENCY', 'EXCHANGE_RATE_TO_USD'])
    
    print(f"[FX] Uploading {len(df_final)} rows to {TARGET_TABLE}...")
    
    success, _, nrows, _ = write_pandas(
        sf_client._conn, 
        df_final, 
        TARGET_TABLE.split('.')[-1], 
        database=TARGET_TABLE.split('.')[0],
        schema=TARGET_TABLE.split('.')[1],
        auto_create_table=True,
        overwrite=True
    )
    
    if success:
        print(f"[FX] Successfully created {TARGET_TABLE} and inserted {nrows} rows.")
    return success
