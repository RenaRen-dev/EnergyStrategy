# AI Energy Strategy: Full Project Report
## Target: DM003EL (Unleaded Gasoline Spot) | Horizon: 10 Business Days
**Author:** Xinyi Ren | **Date:** May 2026

---

# Part 1: Data Infrastructure

*Goal: Build a secure, automated pipeline to pull raw market data from Snowflake into a format AI models can consume.*

---

## 1.1 — Snowflake Connection (`utility/snowflake_client.py`)

**What:** A Python client that connects to our Snowflake data warehouse using **RSA key-pair authentication** (no passwords stored anywhere).

**How it works:**
1. Reads an RSA private key from `data/xren_private_key.p8`
2. Converts it to DER format (what Snowflake's connector expects)
3. All connection settings (account, user, role, warehouse) come from `.env` environment variables
4. Exposes `read_sql()` which uses Arrow-based `fetch_pandas_all()` for fast bulk downloads

**Why key-pair auth instead of passwords?**
- Passwords can be leaked in code, logs, or `.env` files shared accidentally
- RSA keys are cryptographic — even if someone sees the public key, they can't authenticate
- This is the security standard required for production data pipelines in financial firms

---

## 1.2 — FX Rate Ingestion (`utility/fx_client.py`)

**What:** Downloads historical exchange rates from the **Federal Reserve (FRED)** and uploads them to `CMDTYA.PUBLIC.FX_RATES_DAILY`.

**Currencies handled:**
- `DEXCAUS` → CAD/USD (Canadian Dollar)
- `DEXUSEU` → EUR/USD (Euro)
- `EXGEUS` → DEM/USD (German Mark — used to create a **synthetic Euro** for dates before 1999 when the Euro didn't exist yet)

**Synthetic Euro formula:** `EUR/USD = 1.95583 / DEM/USD` (the official Deutschmark-to-Euro conversion rate)

**Why we need this:** Our raw price data contains European contracts priced in EUR and Canadian contracts priced in CAC. Without daily FX rates, we can't convert them to USD. The synthetic Euro extends our conversion capability back to the 1990s, giving the AI more historical context.

---

## 1.3 — Raw Data Download (`utility/download_data.py`)

**What:** Downloads the full `PRICEDATA_PARSED` table from Snowflake in batches of 500,000 rows and saves it locally as Parquet (compressed columnar format).

**Why Parquet instead of CSV?**
- 10x smaller file size (columnar compression)
- 100x faster to load (no string parsing)
- Preserves data types (dates stay as dates, not strings)

---

# Part 2: Data Cleaning & Preparation

*Goal: Transform raw market data into a clean, normalized matrix the AI can understand.*

The cleaning pipeline has **8 sequential stages**. Each stage exists because skipping it would cause the AI to learn fake patterns.

---

## 2.1 — Description Decoding (`utility/parse_description_udf.py`)

**What:** A 963-line NLP parser that converts raw DESCRIPTION strings into 6 structured categorical features.

**The problem:** Raw descriptions are free-text strings from 8 different market data vendors, each with their own naming conventions:
```
"DTN Unl Reg Rochester MN UnbAvg"     → DTN Wholesale Rack format
"NYMEX RBOB Gasoline 01-Mo Floor"     → NYMEX Financial format
"ICE Gas Oil 03-Mo Comb"              → ICE Financial format
"Gasoline FOB NWE Cargo 01-Mo"        → International Physical format
"Enterprise Singapore Gasoline..."     → Singapore Trade Statistics
"CFTC Net Long Positions"             → Government Statistics
```

**What the parser outputs for each description:**

| Feature | Example | Why we need it |
|---------|---------|----------------|
| `PRODUCT` | Unleaded Gasoline | Group similar commodities together |
| `GRADE` | Regular (Unbranded) | Distinguish quality levels |
| `GEOGRAPHY` | Rochester MN | Know which regional market |
| `DELIVERY` | Rack Terminal | Physical vs. Financial contract |
| `TIMING` | Spot / 1-Month Forward | Spot price vs. futures curve |
| `IS_SPOT` | Yes / No | Filter for spot-only analysis |

**How it works (dispatcher pattern):**
1. Tokenize the description string by spaces
2. Check the **first token** to identify the vendor dialect (DTN, NYMEX, ICE, etc.)
3. Route to the appropriate sub-parser (8 parsers total)
4. Each sub-parser uses regex patterns and lookup dictionaries to extract the 6 features

**Why this matters:** Without decoding, symbol "DM003EL" is meaningless to a human. After decoding, it becomes "Unleaded Gasoline, Regular, New York Harbor, Barge, Spot" — now a trader knows exactly what contract the AI is forecasting.

**Deployment:** The parser runs as a **Snowflake Python UDF** (User-Defined Function) via `get_udf_sql()`. The entire parsing logic is embedded in the SQL `CREATE FUNCTION` statement so it executes server-side inside Snowflake — no data leaves the warehouse.

---

## 2.2 — Currency Normalization (`utility/normalization_sql.py`, Phase 1)

**What:** Convert all prices to USD.

```
USC (US Cents)       → USD:  value / 100
CAC (Canadian Cents) → USD:  (value / 100) × CAD/USD rate from FX_RATES_DAILY
EUR (Euros)          → USD:  value × EUR/USD rate from FX_RATES_DAILY
```

**Why:** A European gasoline cargo at "600 EUR/MT" and a US rack terminal at "2.50 USD/GAL" represent similar products at similar prices — but the raw numbers look completely different. Without currency unification, the AI would learn fake patterns based on denomination, not actual market movements.

---

## 2.3 — Volume Normalization (`utility/normalization_sql.py`, Phase 2)

**What:** Convert all units to **USD per Gallon**.

```
GAL → GAL:  no change
LTR → GAL:  USD/LTR × 3.78541  (liters per gallon)
BBL → GAL:  USD/BBL ÷ 42.0     (42 gallons per barrel)
MT  → GAL:  USD/MT ÷ BBL_PER_MT ÷ 42.0  (product-specific gravity)
```

**Product-specific gravity** (barrels per metric ton):
- Gasoline/Naphtha: **8.5** (lighter product)
- Diesel/Jet/Gas Oil: **7.45** (medium)
- Heavy Fuel Oil/Bunker: **6.3** (heavier)

**Why different gravity values?** A metric ton of gasoline contains more barrels than a metric ton of heavy fuel oil because gasoline is less dense. Using a single conversion factor would introduce systematic error in the AI's cross-product comparisons.

---

## 2.4 — Deduplication (`utility/revin_sql.py`, Step 1)

**What:** When multiple assessments exist for the same symbol on the same day, take the **median** value.

**Why median instead of mean?** The median ignores outlier assessments (e.g., a data entry error or a stale morning quote that doesn't reflect the afternoon market). This prevents noise from corrupting the rolling statistics computed later.

---

## 2.5 — Dense Temporal Grid + Forward Fill (`utility/revin_sql.py`, Steps 2–6)

**What:**
1. Build a "date scaffold" — every calendar day from the start to end of each symbol's life
2. Cross-join each symbol to its date range (creates a row for every day, even weekends)
3. Left-join actual prices onto this grid
4. Use `LAST_VALUE(IGNORE NULLS)` to carry forward the last known price

**Why:**
- **Weekends/holidays:** Markets don't publish prices on Saturday/Sunday. Forward-fill assumes the price didn't change (correct — it was last traded on Friday).
- **Data gaps:** Some products occasionally skip a day. Without fill, the AI sees a "NaN" and either crashes or interpolates incorrectly.
- **Alignment:** All symbols must have identical date indices to form the wide matrix the AI needs.

---

## 2.6 — RevIN: Rolling Z-Score Normalization (`utility/revin_sql.py`, Steps 7–8)

**What:** For each symbol, compute a **256-day rolling mean and standard deviation**, then normalize:

```
Z_Score = (Price − Rolling_Mean_256d) / (Rolling_Std_256d + 1e-8)
```

Bounded to **[-3.0, +3.0]**.

**Why RevIN (Reversible Instance Normalization) — 5 reasons:**

1. **Non-stationarity:** Gasoline was ~$1.50/gal in 2020 and ~$2.50/gal in 2024. Raw prices make the AI think "2.50 is always high" — but it's only high relative to 2020. Rolling Z-Score says: "2.50 is high relative to the last 256 days."

2. **Cross-product comparability:** Gasoline spot ranges $2–$3 while a NYMEX futures contract ranges $60–$80. Z-Score converts both to the same scale (roughly -3 to +3). A Z=+1.5 in gasoline = same "unusual move" significance as Z=+1.5 in futures.

3. **256-day window = ~1 trading year:** Captures a full seasonal cycle so the model doesn't confuse summer driving season (higher gasoline) with genuine anomalies.

4. **Reversibility — how to get back to real prices:**
   ```
   Forecast_Price = Rolling_Mean + (Z_Score_Forecast × Rolling_Std)
   ```
   Example: Mean=$2.35, Std=$0.15, Z=-1.42 → Price = $2.35 + (-1.42 × $0.15) = **$2.137/gal**

5. **Bounded [-3, +3]:** Extreme Z-Scores (Z=8 during a crash) would distort the model's attention mechanism. Clamping preserves 99.7% of the natural distribution while preventing outlier corruption.

---

## 2.7 — Liquidity Filter (`pipeline/data_preprocessing.py`, Step 2)

**What:** Drop any symbol with <**80%** data coverage.

```python
valid_mask = pivot_df.notna() & (pivot_df != 0)
active_symbols = valid_mask.mean()[valid_mask.mean() >= 0.80].index
```

**Result:** 46 raw symbols → **27 liquid symbols** retained.

**Why:** A symbol with 50% missing data means half its Z-Scores are forward-filled flat lines. The AI would learn "this product never moves" — when in reality it just wasn't being priced.

---

## 2.8 — Wide-Matrix Pivot + Candidate Selection (`pipeline/data_preprocessing.py`, Steps 2–3)

**What:** Pivot from long format (Symbol, Date, Z_Score) into a wide matrix (rows=dates, columns=symbols). Then select the top 50 most volatile symbols as candidates for model discovery.

```
            DM003EL   DM0043X   DM0033X   ...   DM003AS
2020-01-01   +0.42     +0.38     +0.45    ...    +0.33
...
2026-05-14   -1.59     -1.44     -1.62    ...    -1.38
```

**Shape:** ~2,312 rows × 27 columns

**Why volatility ranking for candidates?** Low-volatility symbols (e.g., government statistics that barely change) waste GPU memory. Ranking by variance ensures the model spends its attention budget on actively-trading contracts.

---

## 2.9 — Data Cleaning Pipeline: End-to-End Example

Follow **one real data point** — a European gasoline cargo — as it passes through every cleaning step:

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║  RAW INPUT:  "Gasoline FOB NWE Cargo"  |  Price: 600 EUR/MT  |  Date: Friday  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                                        │
                                        ▼
┌─────────────────────────────────┬──────────────────────────────────────────────┐
│  STEP 2.1: Description Decode   │  WHY: Raw text is unstructured. The AI      │
│                                 │  can't read "FOB NWE Cargo." We parse it    │
│  "Gasoline FOB NWE Cargo"       │  into structured fields the model can use.  │
│        ↓                        │                                              │
│  Product:   Unleaded Gasoline   │  RESULT: Now we know this is a European     │
│  Geography: Northwest Europe    │  physical gasoline contract, not a US       │
│  Delivery:  Cargo (Ship)        │  rack terminal or a futures contract.       │
│  Timing:    Spot                │                                              │
└─────────────────────────────────┴──────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────┬──────────────────────────────────────────────┐
│  STEP 2.2: Currency → USD       │  WHY: Can't compare EUR prices to USD       │
│                                 │  prices. The AI would learn fake patterns   │
│  600 EUR × 1.08 (FX rate)       │  from currency differences, not real        │
│        ↓                        │  market movements.                          │
│  = $648.00 USD/MT               │                                              │
│                                 │  MATH: Price_USD = Price_EUR × EUR/USD      │
└─────────────────────────────────┴──────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────┬──────────────────────────────────────────────┐
│  STEP 2.3: Volume → USD/Gallon  │  WHY: Can't compare $/Metric-Ton to        │
│                                 │  $/Gallon. A US rack price at $2.50/gal     │
│  $648.00 / (8.5 BBL/MT × 42)   │  and a European cargo at $648/MT are        │
│        ↓                        │  actually similar — but the raw numbers     │
│  = $1.81 USD/Gallon             │  look completely different.                 │
│                                 │                                              │
│                                 │  MATH: Price_GAL = USD / (Gravity × 42)    │
└─────────────────────────────────┴──────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────┬──────────────────────────────────────────────┐
│  STEP 2.5: Fill Weekends        │  WHY: Markets close on weekends. Without    │
│                                 │  filling, the AI sees "missing data" and    │
│  Friday:    $1.81               │  either crashes or guesses wrong.           │
│  Saturday:  $1.81  (filled)     │                                              │
│  Sunday:    $1.81  (filled)     │  METHOD: Carry forward Friday's price.      │
│  Monday:    $1.83  (new data)   │  The price didn't change — the market       │
│                                 │  was simply closed.                         │
└─────────────────────────────────┴──────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────┬──────────────────────────────────────────────┐
│  STEP 2.6: RevIN Z-Score        │  WHY: Gasoline was $1.50 in 2020 and $2.50  │
│                                 │  in 2024. The AI must compare "relative     │
│  Price:  $1.81                  │  movement" not "absolute price."            │
│  Mean₂₅₆: $1.65                │                                              │
│  Std₂₅₆:  $0.14                │  MATH: Z = (Price − Mean) / Std             │
│        ↓                        │        Z = (1.81 − 1.65) / 0.14            │
│  Z-Score = +1.14                │        Z = +1.14                            │
│                                 │                                              │
│  Meaning: Price is 1.14 std     │  BOUNDED: Clamped to [-3, +3] to prevent   │
│  deviations ABOVE its 256-day   │  outlier crashes from distorting the model. │
│  rolling average.               │                                              │
└─────────────────────────────────┴──────────────────────────────────────────────┘
                                        │
                                        ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║  ML-READY OUTPUT:  Symbol DM0043X  |  Date: Friday  |  Z-Score: +1.14        ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

# Part 3: Market Discovery — MOIRAI Inference (`ml/inference.py`)

*Goal: Use AI attention to discover which contracts actually influence the target — automatically, without human bias.*

---

## 3.1 — How MOIRAI Works: The Q, K, Attention Mechanism

**Model:** `Salesforce/moirai-1.1-R-small` (32M parameters)  
**Pre-trained on:** 27 billion time-series observations across energy, finance, weather, and retail  
**Role in our pipeline:** The **scout** — it scans all 27 contracts and discovers which ones are related. (Chronos-2 then uses those relationships to forecast.)

MOIRAI is a Transformer model — the same type of AI behind ChatGPT. But instead of reading text, it reads price histories. The key mechanism is **Self-Attention**, which lets the model figure out which contracts contain useful information for predicting other contracts.

### Case Study: Discovering the "Lead-Lag"

To see how the AI finds hidden signals, look at this 3-day window where Contract A "predicts" what Contract B will do 48 hours later.

| Day | Contract A (DM0043X) | Contract B (DM003EL) | Market Role |
| :--- | :--- | :--- | :--- |
| **Monday** | **$2.40 (Surge)** | $2.30 | **A moves first** |
| **Tuesday** | $2.55 | $2.31 | |
| **Wednesday** | $2.58 | **$2.48 (Surge)** | **B follows** |

**How the AI "See" this link:**
1. **The Question (Query $Q$):** On Wednesday, Contract B spikes and asks the system: *"I just surged; has this exact signature happened elsewhere recently?"*
2. **The Database (Key $K$):** The AI scans all other contracts. It "looks back" and finds that Contract A's surge on **Monday** is a mathematical match for B's surge on Wednesday.
3. **The Connection ($Q \times K$):** The AI creates a link because the "fingerprints" match. It identifies that A **drives** B with a 2-day delay.
4. **The Strategy:** We now monitor Contract A as an **Early Warning Signal**. When A moves, we know exactly what B is likely to do 48 hours later.

Now here's how this works mathematically, step by step:

---

### Step 1: Convert price patterns into vectors (Q and K)

Each contract's recent price pattern gets converted into two vectors:
- **Q (Query)** = "What kind of information am I looking for?"
- **K (Key)** = "What kind of information do I contain?"

Think of it like a dating app. Q is your "what I'm looking for" profile. K is your "who I am" profile. The model learns these vectors during pre-training.

```
Example (simplified to 3 numbers per vector):

Contract A (Gasoline Spot — recent pattern: trending down)
  Q_A = [0.8, -0.3, 0.5]   ← "I'm looking for contracts that predict mean-reversion"
  K_A = [0.2, 0.7, -0.1]   ← "I contain a downward trend signal"

Contract B (Futures M1 — recent pattern: also trending down, started 2 days earlier)
  Q_B = [0.1, 0.4, 0.2]    ← "I'm looking for spot price confirmation"
  K_B = [0.9, -0.2, 0.6]   ← "I contain an early downward signal"

Contract C (Rack Terminal — recent pattern: flat, no movement)
  Q_C = [0.3, 0.1, -0.4]   ← "I'm looking for direction clues"
  K_C = [0.0, 0.1, 0.0]    ← "I contain almost no useful signal"
```

### Step 2: Calculate similarity (Q × K)

To find out how much Contract A should "pay attention to" Contract B, we multiply A's Query by B's Key:

```
Attention Score = Q_A · K_B  (dot product)

A looking at B:  Q_A · K_B = (0.8×0.9) + (-0.3×-0.2) + (0.5×0.6) = 0.72 + 0.06 + 0.30 = 1.08  ← HIGH!
A looking at C:  Q_A · K_C = (0.8×0.0) + (-0.3×0.1)  + (0.5×0.0) = 0.00 - 0.03 + 0.00 = -0.03 ← LOW
A looking at A:  Q_A · K_A = (0.8×0.2) + (-0.3×0.7)  + (0.5×-0.1)= 0.16 - 0.21 - 0.05 = -0.10 ← LOW
```

**What this means:** A's Query ("I need mean-reversion clues") matches strongly with B's Key ("I have an early downward signal") — score 1.08. But C's Key ("I have nothing useful") gives a score near zero. **The model has discovered that B contains useful information for predicting A.**

### Step 3: Convert scores to attention weights (Softmax)

We normalize the raw scores into probabilities using softmax (so they sum to 1.0):

```
Raw scores for A:   A→A = -0.10,   A→B = 1.08,   A→C = -0.03

After softmax:      A→A = 0.12     A→B = 0.76     A→C = 0.12
                                    ↑
                          A pays 76% of its attention to B!
```

### Step 4: Do this for ALL pairs → 27×27 Attention Matrix

In our real pipeline, this runs for all 27 contracts simultaneously, producing a **27×27 grid**:

```
             Who is being attended to (K)
             DM003EL   DM003AS   DM0043X   DM0033X   ...
Who is     ┌──────────────────────────────────────────
attending  │
(Q)        │
DM003EL    │  0.04      0.22      0.08      0.06     ← DM003EL pays 22% attention to DM003AS
DM003AS    │  0.05      0.04      0.12      0.10     ← DM003AS only pays 5% to DM003EL
DM0043X    │  0.20      0.10      0.05      0.08
...        │
```

---

## 3.2 — Why Attention, Not Pearson Correlation?

The model is **asymmetric** ($A \rightarrow B \neq B \rightarrow A$), which is why we cannot use simple Pearson correlation. While correlation only shows that two contracts move together, attention reveals **who moved first**. If A pays heavy attention to B, but B ignores A, B is the **leading indicator** (Driver). This directional insight is essential for identifying early warning signals in trading.

---

## 3.3 — Selecting the Market Leader (Target)

The Market Leader is the "center of gravity" of the market. To find it, we sum how much all other contracts look at each contract (summing each column of the attention matrix):

```
Global Influence Score (total attention received):
  DM003EL:  12.45  ← Most looked-at contract → 🏆 MARKET LEADER (Target)
  DM003AS:  10.22
  DM0043X:   9.87
  ...
  DM0099Z:   2.13  ← Almost nobody looks at this one
```


---

## 3.4 — Pair Discovery & Directionality

**Output:** Market Leader = **DM003EL** (highest influence), plus **9 top pairs**.

For each pair, we compute bidirectional attention:
- `A→B` = how much A attends to B (how much A uses B's information)
- `B→A` = how much B attends to A
- Ratio = A→B / B→A
- If ratio > 1.1: B is a **DRIVER** (leading indicator — moves *before* the target)
- If ratio < 0.9: B is a **FOLLOWER** (lags the target — confirms but doesn't predict)
- If 0.9 ≤ ratio ≤ 1.1: **MIRROR** (simultaneous co-movement)

**Why this matters for trading:** A "driver" contains predictive information the Chronos-2 model can exploit. A "follower" only confirms what already happened. The full pair classification results are shown in Part 5.

---

# Part 4: Chronos-2 Forecasting (`ml/chronos_test.py`)

*Goal: Use the discovered pairs to produce a better forecast than using the target alone.*

---

## 4.1 — The Core Idea: Price Changes Become "Words"

Chronos-2 works like **ChatGPT, but for prices instead of text**.

Here's the analogy. ChatGPT reads a sentence like *"The weather today is sunny and warm, so tomorrow will be..."* and predicts the next word is probably *"sunny"*. Chronos-2 does the exact same thing, but with price movements:

```
Raw prices:    $2.50,  $2.48,  $2.55,  $2.60,  $2.58,  ???
                 ↓       ↓       ↓       ↓       ↓
"Words":       flat,   down,    up,     up,    down,   ???
                                                        ↓
Model predicts next "word":                             up
                                                        ↓
Back to price:                                        $2.62
```

**Step 1 — Tokenize:** Convert price history into a sequence of "tokens" (patterns of movement over groups of days)  
**Step 2 — Predict:** The language model predicts the next token — what movement pattern is most likely to come next  
**Step 3 — Decode:** Convert that predicted token back into an actual price forecast



---

## 4.2 — What the Model Outputs: Directional Probability

The model doesn't just say "price goes up." It gives a **probability distribution**:

```
Directional Probability:
  P(UP)   = 85%  →  Strong conviction — price likely rises
  P(DOWN) = 15%  →  Low probability of decline

Confidence Range (how far):
  Q10 = -1.69  (pessimistic — 10% chance it's worse than this)
  Q50 = -1.48  (median — most likely outcome)
  Q90 = -1.34  (optimistic — 10% chance it's better than this)
```

**How a trader uses this:**
- **P(UP) > 70%** → Strong signal, take a position
- **P(UP) = 50–60%** → Weak signal, maybe wait or reduce size
- **Q10–Q90 spread narrow** → Model is confident about the magnitude
- **Q10–Q90 spread wide** → High uncertainty — reduce position size even if direction is clear

---

## 4.3 — Pre-Trained & Zero-Shot: Why No Training Needed

**Model:** `amazon/chronos-2` (120M parameters)  
**Pre-trained on:** Billions of time-series observations including energy, finance, weather, and retail data  

**"Zero-shot"** means we use the model **as-is** — we never train it on our gasoline data. This works because:

- **Already an expert:** Chronos-2 has already "seen" billions of price patterns during pre-training — mean-reversion, momentum, seasonal cycles, regime shifts. When we show it gasoline data, it recognizes these patterns immediately.

---

# Part 5: Output Analysis & Results

*Goal: Prove the AI-discovered pairs provide real predictive value.*

---

## 5.1 — Experiment Design

| Experiment | Input | Purpose |
|-----------|-------|---------|
| **A) Baseline** | DM003EL only (univariate) | What can history alone predict? |
| **B) Full MV** | DM003EL + 9 pairs (10 variates) | Does adding pairs help? |
| **C) Ablation** | DM003EL + 1 pair at a time (×9 runs) | Which pairs help most? |

**Test Protocol:** Train on 2,302 days, test on last 10 days (held out).

---

## 5.2 — Metrics

- **MAE** = Mean Absolute Error across 10 forecast days (lower = better)
- **Hit Rate** = % of days where predicted direction matches actual direction

---

## 5.3 — MOIRAI Pair Discovery Output (Driver / Follower / Mirror)

| Pair | DM003EL→Pair | Pair→DM003EL | Relationship | Interpretation |
|------|:---:|:---:|:---:|---|
| DM003AS | High | Low | ← **DRIVER** | DM003AS leads DM003EL — potential early signal |
| DM0093Y | High | Low | ← **DRIVER** | DM0093Y provides forward-looking signal |
| DM0093X | High | Low | ← **DRIVER** | DM0093X leads the target |
| DM0043X | Medium | High | → **FOLLOWER** | DM0043X reacts to DM003EL moves |
| DM0033X | Medium | High | → **FOLLOWER** | DM0033X tracks DM003EL with a lag |
| DM0043Y | Medium | High | → **FOLLOWER** | Reacts after DM003EL |
| DM0043Z | Medium | Medium | ↔ **MIRROR** | Moves simultaneously — same underlying driver |
| DM0033Z | Medium | Medium | ↔ **MIRROR** | Co-movement, likely same product family |
| DM0033Y | Medium | Medium | ↔ **MIRROR** | Simultaneous movement |

**Key Takeaway:** The 3 DRIVER contracts (DM003AS, DM0093Y, DM0093X) are the most valuable — they move *before* DM003EL. This aligns with the ablation results below, where these same 3 produced the largest MAE improvements.

---

## 5.4 — Results: Baseline vs. Multivariate

| Configuration | MAE | Hit Rate | Improvement |
|:---|:---|:---|:---|
| A) Baseline (Target Only) | 0.0505 | 100% | — |
| **B) Full MV (+ 9 Pairs)** | **0.0292** | **100%** | **−42.1%** |

**The pairs reduced forecast error by 42.1%** while maintaining perfect directional accuracy.

---

## 5.5 — Per-Pair Ablation

| Rank | Pair | Solo MAE | Improvement vs Baseline | Improvement % |
|:---|:---|:---|:---|:---|
| 1 | DM003AS | 0.0153 | +0.0352 | **69.7%** |
| 2 | DM0093Y | 0.0221 | +0.0283 | 56.1% |
| 3 | DM0093X | 0.0240 | +0.0265 | 52.5% |
| 4 | DM0043X | 0.0243 | +0.0262 | 51.9% |
| 5 | DM0043Z | 0.0248 | +0.0257 | 50.9% |
| 6 | DM0043Y | 0.0248 | +0.0256 | 50.7% |
| 7 | DM0033X | 0.0284 | +0.0220 | 43.6% |
| 8 | DM0033Y | 0.0287 | +0.0218 | 43.2% |
| 9 | DM0033Z | 0.0288 | +0.0216 | 42.8% |

**All 9 pairs improved the forecast.** Not one degraded it. The MOIRAI discovery process finds genuinely useful signals.

---

## 5.6 — Feature Importance (% of Forecast Signal)

| # | Source | Importance | Role |
|---|--------|-----------|------|
| 1 | DM003EL (Own History) | 98.84% | Target autoregressive signal |
| 2 | DM003AS | 0.18% | Top covariate |
| 3 | DM0093Y | 0.14% | Covariate |
| 4–10 | DM0093X..DM0033Z | 0.11–0.13% each | Covariates |
| | **TOTAL** | **100.0%** | |

**Interpretation:** The target's own history is the dominant signal (expected for a foundation model). But the collective 1.16% from pairs is what turns a 0.0505 MAE into 0.0292. In quantitative trading, a consistent edge of even 1% compounds over hundreds of trades.

---

## 5.7 — Chronos-2 Day-by-Day Forecast Output (Actual Results)

Last known Z-Score (end of training window): **-1.5855**

| Day | Mean | Q10 | Q50 | Q90 | Actual | Dir | Hit |
|-----|------|-----|-----|-----|--------|-----|-----|
| D1 | -1.4828 | -1.6947 | -1.4828 | -1.3352 | -1.5721 | ▲ | ✅ |
| D2 | -1.4895 | -1.7575 | -1.4895 | -1.3301 | -1.5573 | ▲ | ✅ |
| D3 | -1.4969 | -1.8183 | -1.4969 | -1.3084 | -1.5428 | ▲ | ✅ |
| D4 | -1.5061 | -1.8490 | -1.5061 | -1.2665 | -1.5285 | ▲ | ✅ |
| D5 | -1.4891 | -1.9045 | -1.4891 | -1.2162 | -1.5145 | ▲ | ✅ |
| D6 | -1.4858 | -1.9600 | -1.4858 | -1.1738 | -1.5007 | ▲ | ✅ |
| D7 | -1.4797 | -2.0074 | -1.4797 | -1.1432 | -1.4871 | ▲ | ✅ |
| D8 | -1.4671 | -2.0140 | -1.4671 | -1.1099 | -1.4738 | ▲ | ✅ |
| D9 | -1.4712 | -2.0885 | -1.4712 | -1.0790 | -1.4606 | ▲ | ✅ |
| D10 | -1.4626 | -2.1351 | -1.4626 | -1.0305 | -1.4606 | ▲ | ✅ |

**Result: 10/10 directional hits (100% accuracy)**

**Reading the table:**
- All Z-Scores are negative (gasoline is below its 256-day rolling average)
- Direction ▲ means the model predicted the Z-Score would move UP (less negative) vs. the last training value of -1.5855
- The actual values confirmed this — gasoline was recovering toward its mean
- **Q10–Q90 spread widens** from ±0.18 on Day 1 to ±0.55 on Day 10 — the model is correctly more uncertain about distant forecasts

**How to convert to real prices:**
- If the last 256-day Rolling Mean = $2.35/gal and Rolling Std = $0.15/gal:
- Day 1 forecast: $2.35 + (-1.4828 × $0.15) = **$2.128/gal** (range: $2.096–$2.150)
- Day 10 forecast: $2.35 + (-1.4626 × $0.15) = **$2.131/gal** (range: $2.030–$2.195)

---

# Part 6: Key Questions & Road Map

---

## 6.1 — Snowflake: Resource Monitor Quota Exceeded
**Issue:** `Warehouse 'gg_wh1' cannot be resumed because resource monitor 'gg_rm' has exceeded its quota.`

**Explanation:** This is a budget safety mechanism in Snowflake. The resource monitor `gg_rm` was set up to prevent accidental overspending. It has detected that the credit usage for the current period has reached its limit and has automatically suspended the warehouse `gg_wh1`.

**Action Needed:** A Snowflake Admin must either increase the credit quota for `gg_rm` or reset the monitor for the new billing period to resume pipeline operations.

## 6.2 — Symbol Description Mapping
**Goal:** Replace cryptic symbol codes (like `DM003EL`) with human-readable descriptions in all final outputs.

**Implementation:** We will use the **Description Decoder** (from Part 2.1) to join the `PRICEDATA_PARSED` metadata with our forecast results. This ensures that a trader looking at the dashboard sees "Unleaded Gasoline Spot, NYH" instead of a raw database key, making the model's insights immediately actionable.

## 6.3 — Backtesting: Do we have a framework?
**Current Status:** We have an **Ablation Test** framework (in `ml/chronos_test.py`) that proves adding covariates reduces error. However, we do not yet have a full **Financial Backtester**.

**Next Step:** Build a "Walk-Forward" simulation. This will slide our 10-day forecast window across the last 252 trading days to calculate:
- **Hit Rate:** % of days the direction was predicted correctly.
- **Sharpe Ratio:** Risk-adjusted return of a strategy following the model's signals.
- **Max Drawdown:** The largest peak-to-trough decline in a simulated portfolio.

---

---

# Appendix: Repository Structure

```
Energy-Strategy/
├── utility/
│   ├── snowflake_client.py       ← RSA key-pair Snowflake connection
│   ├── fx_client.py              ← FRED FX rate ingestion (CAD, EUR, DEM)
│   ├── download_data.py          ← Bulk Parquet download from Snowflake
│   ├── normalization_sql.py      ← Currency + Volume normalization SQL
│   ├── revin_sql.py              ← RevIN: dedup, grid, forward-fill, Z-Score
│   └── parse_description_udf.py  ← 963-line NLP decoder (8 vendor dialects)
├── pipeline/
│   └── data_preprocessing.py     ← Fetch → Pivot → Liquidity filter → Candidates
├── ml/
│   ├── inference.py              ← MOIRAI global sweep + attention-based pair discovery
│   ├── chronos_baseline.py       ← Chronos-1 univariate baseline (AutoGluon)
│   └── chronos_test.py           ← Chronos-2 multivariate ablation test
├── data/
│   ├── pair_contribution_results.json  ← Latest experiment results
│   └── chronos2_contribution_test.png  ← 3-panel visualization
└── docs/
    └── Energy_Strategy_AI_Report.md    ← This report
```

---

*Report generated for Quant Management & Energy Trading Desk — May 2026*
