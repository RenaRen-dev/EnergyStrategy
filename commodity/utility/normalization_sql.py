def get_physical_normalization_sql(source_table: str, target_table: str) -> str:
    return f"""
    CREATE OR REPLACE TABLE {target_table} AS
    WITH FX_RATES AS (
        SELECT DATE, CURRENCY, EXCHANGE_RATE_TO_USD 
        FROM CMDTYA.PUBLIC.FX_RATES_DAILY
    ),
    JOINED AS (
        SELECT 
            P.SYMBOL, P.DESCRIPTION, P.ASSESSDATE, 
            P.PRODUCT, P.GRADE, P.GEOGRAPHY, P.DELIVERY, P.TIMING, P.IS_SPOT,
            P.VALUE AS RAW_VALUE, P.UOM AS RAW_UOM, P.CURRENCY AS RAW_CURRENCY,
            
            -- Phase 1: Unify Currency to USD
            CASE 
                WHEN P.CURRENCY = 'USC' THEN P.VALUE / 100.0
                WHEN P.CURRENCY = 'CAC' THEN (P.VALUE / 100.0) * F_CAD.EXCHANGE_RATE_TO_USD
                WHEN P.CURRENCY = 'EUR' THEN P.VALUE * F_EUR.EXCHANGE_RATE_TO_USD
                WHEN P.CURRENCY = 'USD' THEN P.VALUE
                ELSE P.VALUE -- Fallback for unexpected currencies
            END AS USD_VALUE,
            
            -- Specific Gravity Lookup
            CASE 
                -- Lighter
                WHEN P.PRODUCT ILIKE '%Gasoline%' OR P.PRODUCT ILIKE '%Naphtha%' OR P.PRODUCT ILIKE '%RBOB%' THEN 8.5
                -- Medium
                WHEN P.PRODUCT ILIKE '%Diesel%' OR P.PRODUCT ILIKE '%Gas Oil%' OR P.PRODUCT ILIKE '%Kerosene%' OR P.PRODUCT ILIKE '%Jet%' OR P.PRODUCT ILIKE '%Stove Oil%' THEN 7.45
                -- Heavy
                WHEN P.PRODUCT ILIKE '%Heavy Fuel Oil%' OR P.PRODUCT ILIKE '%Bunker%' OR P.PRODUCT ILIKE '%Furnace Oil%' THEN 6.3
                ELSE 7.0 -- Generic fallback for unidentified oils
            END AS BBL_PER_MT
            
        FROM {source_table} P
        LEFT JOIN FX_RATES F_CAD ON P.ASSESSDATE = F_CAD.DATE AND F_CAD.CURRENCY = 'CAD'
        LEFT JOIN FX_RATES F_EUR ON P.ASSESSDATE = F_EUR.DATE AND F_EUR.CURRENCY = 'EUR'
    ),
    VOLUMETRIC AS (
        SELECT *,
            -- Phase 2 & 3: Unify to Gallons (GAL)
            CASE
                -- Phase 2: Gallons
                WHEN RAW_UOM = 'GAL' THEN USD_VALUE
                -- Phase 2: Liters to Gallons (Price per liter * liters in a gallon = Price per gallon)
                WHEN RAW_UOM = 'LTR' THEN USD_VALUE * 3.78541
                -- Phase 2: Barrels to Gallons (Price per BBL / 42 = Price per gallon)
                WHEN RAW_UOM = 'BBL' THEN USD_VALUE / 42.0
                -- Phase 3: Metric Tons to Gallons (USD/MT -> USD/BBL -> USD/GAL)
                WHEN RAW_UOM = 'MT' THEN (USD_VALUE / BBL_PER_MT) / 42.0
                ELSE USD_VALUE -- Unhandled UOMs
            END AS NORMALIZED_VALUE_USD_GAL
        FROM JOINED
    )
    SELECT 
        SYMBOL, DESCRIPTION, ASSESSDATE, 
        PRODUCT, GRADE, GEOGRAPHY, DELIVERY, TIMING, IS_SPOT,
        RAW_VALUE, RAW_UOM, RAW_CURRENCY,
        NORMALIZED_VALUE_USD_GAL,
        'GAL' AS NORMALIZED_UOM,
        'USD' AS NORMALIZED_CURRENCY
    FROM VOLUMETRIC;
    """
