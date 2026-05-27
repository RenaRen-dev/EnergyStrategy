def get_revin_sql(source_table: str, target_table: str) -> str:
    """
    Returns the CTAS SQL to perform native Snowflake RevIN processing:
    1. Deduplicate intra-day quotes.
    2. Generate a dense temporal grid (weekends included).
    3. Forward-fill prices across gaps.
    4. Compute 256-day rolling mean and standard deviation.
    5. Calculate bounded Z-Scores safely.
    """
    
    return f"""
    CREATE OR REPLACE TABLE {target_table} AS
    WITH 

    -- 1. Deduplicate intra-day quotes
    Deduplicated AS (
        SELECT
            SYMBOL,
            ASSESSDATE,
            MEDIAN(NORMALIZED_VALUE_USD_GAL) AS RAW_VALUE,
            ANY_VALUE(PRODUCT) AS PRODUCT,
            ANY_VALUE(GRADE) AS GRADE,
            ANY_VALUE(GEOGRAPHY) AS GEOGRAPHY,
            ANY_VALUE(DELIVERY) AS DELIVERY,
            ANY_VALUE(TIMING) AS TIMING
        FROM {source_table}
        GROUP BY SYMBOL, ASSESSDATE
    ),

    -- 2. Symbol Bounds (to constrain the grid per asset)
    SymbolBounds AS (
        SELECT 
            SYMBOL,
            MIN(ASSESSDATE) as MIN_DATE,
            MAX(ASSESSDATE) as MAX_DATE,
            ANY_VALUE(PRODUCT) AS PRODUCT,
            ANY_VALUE(GRADE) AS GRADE,
            ANY_VALUE(GEOGRAPHY) AS GEOGRAPHY,
            ANY_VALUE(DELIVERY) AS DELIVERY,
            ANY_VALUE(TIMING) AS TIMING
        FROM Deduplicated
        GROUP BY SYMBOL
    ),

    -- 3. Global Date Range to generate rows
    GlobalBounds AS (
        SELECT 
            MIN(MIN_DATE) as G_MIN,
            MAX(MAX_DATE) as G_MAX
        FROM SymbolBounds
    ),

    -- 4. Date Scaffold (guaranteed gap-free sequence)
    DateScaffold AS (
        SELECT 
            DATEADD(day, seq_val, (SELECT G_MIN FROM GlobalBounds)) AS GRID_DATE
        FROM (
            SELECT ROW_NUMBER() OVER (ORDER BY SEQ8()) - 1 AS seq_val
            FROM TABLE(GENERATOR(ROWCOUNT => 36500)) -- ~100 years max history
        )
        WHERE GRID_DATE <= (SELECT G_MAX FROM GlobalBounds)
    ),

    -- 5. Dense Grid (Cross join dates to symbols within their min/max lifetime)
    DenseGrid AS (
        SELECT 
            s.SYMBOL,
            d.GRID_DATE AS ASSESSDATE,
            s.PRODUCT,
            s.GRADE,
            s.GEOGRAPHY,
            s.DELIVERY,
            s.TIMING
        FROM SymbolBounds s
        JOIN DateScaffold d
          ON d.GRID_DATE >= s.MIN_DATE AND d.GRID_DATE <= s.MAX_DATE
    ),

    -- 6. Forward Fill
    ForwardFilled AS (
        SELECT 
            g.SYMBOL,
            g.ASSESSDATE,
            v.RAW_VALUE,
            g.PRODUCT,
            g.GRADE,
            g.GEOGRAPHY,
            g.DELIVERY,
            g.TIMING,
            LAST_VALUE(v.RAW_VALUE) IGNORE NULLS OVER (
                PARTITION BY g.SYMBOL 
                ORDER BY g.ASSESSDATE 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS FFILLED_VALUE
        FROM DenseGrid g
        LEFT JOIN Deduplicated v 
          ON g.SYMBOL = v.SYMBOL AND g.ASSESSDATE = v.ASSESSDATE
    ),

    -- 7. Rolling Stats (256-day Window)
    RollingStats AS (
        SELECT 
            SYMBOL,
            ASSESSDATE,
            FFILLED_VALUE AS NORMALIZED_VALUE_USD_GAL,
            PRODUCT, GRADE, GEOGRAPHY, DELIVERY, TIMING,
            AVG(FFILLED_VALUE) OVER (
                PARTITION BY SYMBOL 
                ORDER BY ASSESSDATE 
                ROWS BETWEEN 255 PRECEDING AND CURRENT ROW
            ) AS ROLLING_MEAN,
            STDDEV(FFILLED_VALUE) OVER (
                PARTITION BY SYMBOL 
                ORDER BY ASSESSDATE 
                ROWS BETWEEN 255 PRECEDING AND CURRENT ROW
            ) AS ROLLING_STD
        FROM ForwardFilled
    )

    -- 8. Final Select with Z-Score bounds
    SELECT
        SYMBOL,
        ASSESSDATE,
        NORMALIZED_VALUE_USD_GAL,
        ROLLING_MEAN,
        ROLLING_STD,
        CASE 
            WHEN ROLLING_STD IS NULL THEN 0.0
            ELSE 
                GREATEST(-3.0, LEAST(3.0, 
                    (NORMALIZED_VALUE_USD_GAL - ROLLING_MEAN) / (ROLLING_STD + 1e-8)
                ))
        END AS Z_SCORE,
        PRODUCT, GRADE, GEOGRAPHY, DELIVERY, TIMING
    FROM RollingStats
    ORDER BY SYMBOL, ASSESSDATE;
    """
