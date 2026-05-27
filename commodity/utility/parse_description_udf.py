"""
Parse the DESCRIPTION column of PRICEDATA into 6 categorical features.

Features: Product, Grade, Geography, Delivery, Timing, IS_SPOT

Handles 8 market dialects:
  1. DTN Wholesale Rack        (~71%)
  2. NYMEX Financial            (~12.5%)
  3. ICE Financial              (~2.2%)
  4. Enterprise Singapore       (~2.6%)
  5. Pipeline Physical          (~1.2%)
  6. International Physical     (~7%)
  7. Gov Statistics (CFTC/EIA)  (~1.5%)
  8. Fallback                   (~2%)

Usage as Snowpark Vectorized UDF:
    from utility.parse_description_udf import parse_description_batch
"""
from __future__ import annotations

import re
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

_CA_PROVINCES = {
    "AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE",
    "QC", "SK", "YT",
}

_STATE_CODES = _US_STATES | _CA_PROVINCES

_DTN_PRODUCT_MAP = {
    "Unl": "Unleaded Gasoline",
    "ULSD": "Diesel",
    "Kero": "Kerosene",
    "Jet": "Jet Fuel",
    "Furnace": "Furnace Oil",
    "No2/Diesel": "No. 2 Diesel",
    "Stove": "Stove Oil",
    "AvGas": "Aviation Gasoline",
    "No2": "No. 2 Oil",
    "Ethanol": "Ethanol",
    "Diesel": "Diesel",
}

_DTN_GRADE_MAP = {"Reg": "Regular", "Mid": "Midgrade", "Prem": "Premium"}

_DTN_SUFFIXES = {
    "BrAvg", "UnbAvg", "RkAvg", "BrMAvg", "UnbMAvg", "RkMAvg", "Br", "Unb",
}

_BRANDED_SUFFIXES = {"BrAvg", "Br", "BrMAvg"}
_UNBRANDED_SUFFIXES = {"UnbAvg", "Unb", "UnbMAvg"}
_RACK_SUFFIXES = {"RkAvg", "RkMAvg"}

# Regex for NYMEX / ICE forward-month patterns like "03-Mo" or "Mo01"
_MO_PATTERN = re.compile(r"(\d+)-Mo")
_MO_PATTERN2 = re.compile(r"Mo(\d+)")

# Regex for named month + year like "Sep 2020"
_MONTH_YEAR = re.compile(
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})"
)

# Geography keywords for international physical cargo
_DELIVERY_KEYWORDS = {
    "FOB", "CIF", "C+F", "Dlvd", "DAP", "DES", "CFR",
}

_GEO_KEYWORDS = {
    "Arab Gulf": "Arab Gulf",
    "NWE": "Northwest Europe",
    "MED": "Mediterranean",
    "Med": "Mediterranean",
    "USGC": "US Gulf Coast",
    "USAC": "US Atlantic Coast",
    "Spore": "Singapore",
    "Singapore": "Singapore",
    "Japan": "Japan",
    "Korea": "South Korea",
    "Australia": "Australia",
    "Fujairah": "Fujairah",
    "South China": "South China",
    "West Japan": "West Japan",
    "Europe": "Europe",
    "FARAG": "FARAG (NW Europe)",
    "Mt Belvieu": "Mt Belvieu TX",
}


# ---------------------------------------------------------------------------
# Default result template
# ---------------------------------------------------------------------------

def _default() -> dict:
    return {
        "Product": "Unspecified",
        "Grade": "Generic",
        "Geography": "Unspecified",
        "Delivery": "Unspecified",
        "Timing": "Spot",
        "IS_SPOT": "Yes",
    }


# ---------------------------------------------------------------------------
# DTN Rack Parser
# ---------------------------------------------------------------------------

def _parse_dtn(tokens: list[str]) -> dict:
    """Parse DTN Wholesale Rack descriptions."""
    result = _default()
    result["Delivery"] = "Rack Terminal"
    result["Timing"] = "Spot"
    result["IS_SPOT"] = "Yes"

    # tokens[0] = "DTN"
    if len(tokens) < 3:
        result["Product"] = " ".join(tokens[1:])
        return result

    product_tok = tokens[1]
    result["Product"] = _DTN_PRODUCT_MAP.get(product_tok, product_tok)
    idx = 2

    # Grade for Unleaded
    if product_tok == "Unl" and idx < len(tokens) and tokens[idx] in _DTN_GRADE_MAP:
        result["Grade"] = _DTN_GRADE_MAP[tokens[idx]]
        idx += 1
    elif product_tok == "ULSD":
        result["Grade"] = "Ultra-Low Sulfur"
    elif product_tok == "No2/Diesel" and idx < len(tokens) and tokens[idx] == "HS":
        result["Grade"] = "High Sulfur"
        idx += 1
    elif product_tok == "Jet" and idx < len(tokens) and tokens[idx] == "Fuel":
        idx += 1  # consume "Fuel" as part of product name

    # Suffix is the last token
    suffix = tokens[-1]
    if suffix not in _DTN_SUFFIXES:
        # No recognized suffix — use all remaining for geography
        geo_tokens = tokens[idx:]
        result["Geography"] = " ".join(geo_tokens) if geo_tokens else "Unspecified"
        return result

    # Append branded/unbranded to grade
    if suffix in _BRANDED_SUFFIXES:
        result["Grade"] += " (Branded)"
    elif suffix in _UNBRANDED_SUFFIXES:
        result["Grade"] += " (Unbranded)"
    elif suffix in _RACK_SUFFIXES:
        result["Grade"] += " (Rack Average)"

    # Remaining tokens between idx and last token = modifiers + city + state + supplier
    remaining = tokens[idx:-1]

    # Skip CARB / RFG modifiers
    while remaining and remaining[0] in ("CARB", "RFG"):
        remaining.pop(0)

    # Find state/province code scanning from right
    state_pos = None
    for i in range(len(remaining) - 1, -1, -1):
        if remaining[i] in _STATE_CODES:
            state_pos = i
            break

    if state_pos is not None:
        city = " ".join(remaining[:state_pos])
        state = remaining[state_pos]
        result["Geography"] = f"{city} {state}".strip()
    elif remaining:
        result["Geography"] = " ".join(remaining)

    return result


# ---------------------------------------------------------------------------
# NYMEX / CME Financial Parser
# ---------------------------------------------------------------------------

def _parse_nymex(desc: str, tokens: list[str]) -> dict:
    """Parse NYMEX / Nymex / CME financial descriptions."""
    result = _default()
    result["Delivery"] = "Financial Exchange"
    result["IS_SPOT"] = "No"

    # Extract timing from XX-Mo or MoXX pattern
    mo_match = _MO_PATTERN.search(desc)
    mo_match2 = _MO_PATTERN2.search(desc)
    my_match = _MONTH_YEAR.search(desc)

    if mo_match:
        result["Timing"] = f"{int(mo_match.group(1))}-Month Forward"
    elif mo_match2:
        result["Timing"] = f"{int(mo_match2.group(1))}-Month Forward"
    elif my_match:
        result["Timing"] = f"{my_match.group(1)} {my_match.group(2)} Forward"
    else:
        result["Timing"] = "Forward Month"

    # Detect product and geography
    d = desc.upper()
    if "WTI" in d:
        result["Product"] = "Crude Oil"
        result["Grade"] = "WTI"
        result["Geography"] = "Cushing OK"
    elif "RBOB" in d:
        result["Product"] = "RBOB Gasoline"
        result["Grade"] = "Generic"
        result["Geography"] = "New York"
    elif "NY ULSD" in d or "NY HO" in d:
        result["Product"] = "Diesel"
        result["Grade"] = "Ultra-Low Sulfur"
        result["Geography"] = "New York"
    elif "ULSD" in d:
        result["Product"] = "Diesel"
        result["Grade"] = "Ultra-Low Sulfur"
        result["Geography"] = "Unspecified"
    elif "NATURAL GAS" in d or "NAT GAS" in d or "HENRY HUB" in d:
        result["Product"] = "Natural Gas"
        result["Grade"] = "Generic"
        result["Geography"] = "Henry Hub LA"
    else:
        # Generic: take tokens between exchange name and Mo pattern as product
        skip = 1
        if tokens[0].upper() in ("INTRADAY",):
            skip = 2
        if len(tokens) > skip and tokens[skip].upper() == "GLOBEX":
            skip += 1
        product_parts = []
        for t in tokens[skip:]:
            if _MO_PATTERN.match(t) or _MO_PATTERN2.match(t) or _MONTH_YEAR.match(t):
                break
            if t in ("Floor", "Elect", "Comb", "Fin", "Pen", "Calendar", "Swap"):
                break
            product_parts.append(t)
        result["Product"] = " ".join(product_parts) if product_parts else "Unspecified"

    # Venue suffix
    last = tokens[-1] if tokens else ""
    if last in ("Floor", "Elect", "Comb"):
        result["Delivery"] = f"Financial Exchange ({last})"

    return result


# ---------------------------------------------------------------------------
# ICE Financial Parser
# ---------------------------------------------------------------------------

def _parse_ice(desc: str, tokens: list[str]) -> dict:
    """Parse ICE financial descriptions."""
    result = _default()
    result["Delivery"] = "Financial Exchange (ICE)"
    result["IS_SPOT"] = "No"
    result["Geography"] = "Europe"

    mo_match = _MO_PATTERN.search(desc)
    if mo_match:
        result["Timing"] = f"{int(mo_match.group(1))}-Month Forward"
    else:
        result["Timing"] = "Forward Month"

    d = desc.upper()
    if "GAS OIL" in d or "GASOIL" in d:
        result["Product"] = "Gas Oil"
    elif "BRENT" in d:
        result["Product"] = "Crude Oil"
        result["Grade"] = "Brent"
    else:
        # Take tokens between "ICE" and Mo pattern
        product_parts = []
        for t in tokens[1:]:
            if _MO_PATTERN.match(t) or t in ("Comb", "Floor", "Elect"):
                break
            product_parts.append(t)
        result["Product"] = " ".join(product_parts) if product_parts else "Unspecified"

    return result


# ---------------------------------------------------------------------------
# Enterprise Singapore Parser
# ---------------------------------------------------------------------------

def _parse_singapore(desc: str) -> dict:
    """Parse Enterprise Singapore / ES trade statistics descriptions."""
    result = _default()
    result["Delivery"] = "Trade Statistics"
    result["Timing"] = "Spot"
    result["IS_SPOT"] = "Yes"
    result["Geography"] = "Singapore"

    # Two prefix forms: "Enterprise Singapore {Product} Singapore ..."
    #                    "ES {Product} Singapore ..."
    if desc.startswith("Enterprise Singapore "):
        rest = desc[len("Enterprise Singapore "):]
    elif desc.startswith("ES "):
        rest = desc[len("ES "):]
    else:
        result["Product"] = desc
        return result

    # Product is everything before the first "Singapore" in rest
    parts = rest.split(" Singapore", 1)
    result["Product"] = parts[0].strip() if parts[0].strip() else "Unspecified"

    # If there's a trade partner after "Singapore"
    if len(parts) > 1 and parts[1].strip():
        trade_info = parts[1].strip().split()
        # First token(s) before Imp/Exp/ReExp are the country
        country_parts = []
        for t in trade_info:
            if t in ("Imp", "Exp", "ReExp", "Vol", "Val", "Dom", "Tot"):
                break
            country_parts.append(t)
        if country_parts:
            country = " ".join(country_parts)
            result["Geography"] = f"Singapore -> {country}"

    return result


# ---------------------------------------------------------------------------
# Pipeline Parser
# ---------------------------------------------------------------------------

def _parse_pipeline(desc: str) -> dict:
    """Parse Pipeline physical descriptions."""
    result = _default()
    result["Delivery"] = "Pipeline"
    result["IS_SPOT"] = "Yes"

    # Extract timing: Cycle + number or Prompt
    cycle_match = re.search(r"Cycle\s*(\d+)", desc)
    if cycle_match:
        result["Timing"] = f"Cycle {cycle_match.group(1)}"
    elif "prompt" in desc.lower():
        result["Timing"] = "Prompt"
    elif "cycle" in desc.lower():
        result["Timing"] = "Cycle"
    else:
        result["Timing"] = "Spot"

    # Geography
    if "USGC" in desc:
        result["Geography"] = "US Gulf Coast"
    elif "USAC" in desc:
        result["Geography"] = "US Atlantic Coast"
    elif "Colonial" in desc:
        result["Geography"] = "Colonial Pipeline"
    else:
        # Try to find City State pattern
        state_match = re.search(
            r"([A-Z][a-zA-Z/\s]+?)\s+(" + "|".join(_STATE_CODES) + r")\b", desc
        )
        if state_match:
            result["Geography"] = f"{state_match.group(1).strip()} {state_match.group(2)}"
        else:
            result["Geography"] = "Unspecified"

    # Product: everything before geography/pipeline keywords
    d = desc
    for kw in ("Pipeline", "USGC", "USAC", "Colonial", "Waterborne",
               "Assessment", "Cycle", "Prompt", "Differential"):
        d = d.split(kw)[0]
    # Also strip city/state if found
    for code in _STATE_CODES:
        d = re.sub(rf"\b[A-Z][a-zA-Z/\s]+?\s+{code}\b", "", d)
    product = d.strip().rstrip()
    result["Product"] = product if product else "Unspecified"

    # Grade
    if "ULS" in desc or "Ultra" in desc:
        result["Grade"] = "Ultra-Low Sulfur"
    elif "No.2" in desc or "No2" in desc:
        result["Grade"] = "No. 2"
    elif "54" in desc:
        result["Grade"] = "54"
    elif "55" in desc:
        result["Grade"] = "55"

    return result


# ---------------------------------------------------------------------------
# International Physical Cargo Parser
# ---------------------------------------------------------------------------

def _parse_intl_physical(desc: str, tokens: list[str]) -> dict:
    """Parse international physical cargo descriptions
    (Marine, Bunker, Jet Kero FOB, Gasoline, Gasoil, ULSD, Naphtha, etc.)"""
    result = _default()
    result["Delivery"] = "Physical Cargo"
    result["IS_SPOT"] = "Yes"
    result["Timing"] = "Spot"

    # Check for swap/forward patterns
    if "Swap" in desc:
        result["Delivery"] = "Swap"
        result["IS_SPOT"] = "No"
    mo_match = _MO_PATTERN.search(desc) or _MO_PATTERN2.search(desc)
    if mo_match:
        num = mo_match.group(1)
        result["Timing"] = f"{int(num)}-Month Forward"
        if result["IS_SPOT"] == "Yes":
            result["IS_SPOT"] = "No"

    # Detect delivery method
    for kw in ("FOB", "CIF", "C+F", "Dlvd", "DAP", "DES", "CFR"):
        if kw in desc:
            result["Delivery"] = f"Physical {kw}"
            break
    if "Waterborne" in desc:
        result["Delivery"] = "Physical Waterborne"
    elif "Barge" in desc:
        result["Delivery"] = "Physical Barge"
    elif "Cargo" in desc and result["Delivery"] == "Physical Cargo":
        pass  # already set

    # Detect geography
    for geo_key, geo_val in _GEO_KEYWORDS.items():
        if geo_key in desc:
            result["Geography"] = geo_val
            break

    # Product: take tokens before delivery/geography keywords
    product_parts = []
    stop_words = set(_DELIVERY_KEYWORDS) | {"Cargo", "Barge", "Waterborne",
        "Swap", "strip", "vs", "Global"} | set(_GEO_KEYWORDS.keys())
    for t in tokens:
        if t in stop_words or _MO_PATTERN.match(t) or _MO_PATTERN2.match(t):
            break
        product_parts.append(t)
    result["Product"] = " ".join(product_parts) if product_parts else desc.split()[0]

    # Grade: look for sulfur specs
    sulfur_match = re.search(r"(\d+\.?\d*%?\s*S(?:ulfur)?|\d+ppm)", desc)
    if sulfur_match:
        result["Grade"] = sulfur_match.group(1).strip()
    cst_match = re.search(r"(\d+)\s*CST", desc)
    if cst_match:
        result["Grade"] = f"{cst_match.group(1)} CST"

    return result


# ---------------------------------------------------------------------------
# Gov Statistics Parser
# ---------------------------------------------------------------------------

def _parse_gov_stats(desc: str, tokens: list[str]) -> dict:
    """Parse government statistics descriptions (CFTC, EIA, ECB)."""
    result = _default()
    result["Delivery"] = "Government Report"
    result["IS_SPOT"] = "Yes"
    result["Timing"] = "Spot"
    result["Product"] = " ".join(tokens[1:]) if len(tokens) > 1 else tokens[0]
    result["Geography"] = tokens[0]  # CFTC, EIA, ECB
    return result


# ---------------------------------------------------------------------------
# Fallback Parser
# ---------------------------------------------------------------------------

def _parse_fallback(desc: str, tokens: list[str]) -> dict:
    """Best-effort parse for unclassified descriptions."""
    result = _default()

    # Check for currency pairs
    if "Dollar" in desc or "EUR" in desc or "AUD" in desc:
        result["Product"] = "Currency"
        result["Grade"] = desc
        result["Delivery"] = "Financial Exchange"
        result["IS_SPOT"] = "No"
        result["Timing"] = "Spot"
        return result

    # Check for Mo patterns (financial)
    mo_match = _MO_PATTERN.search(desc) or _MO_PATTERN2.search(desc)
    if mo_match:
        result["IS_SPOT"] = "No"
        result["Timing"] = f"{int(mo_match.group(1))}-Month Forward"
        result["Delivery"] = "Financial Exchange"

    # Check for physical delivery keywords
    for kw in ("FOB", "CIF", "C+F", "Dlvd", "DAP", "Barge", "Cargo", "Waterborne"):
        if kw in desc:
            result["Delivery"] = f"Physical {kw}"
            result["IS_SPOT"] = "Yes"
            result["Timing"] = "Spot"
            break

    # Geography
    for geo_key, geo_val in _GEO_KEYWORDS.items():
        if geo_key in desc:
            result["Geography"] = geo_val
            break

    # Try to find City State pattern
    if result["Geography"] == "Unspecified":
        for code in _STATE_CODES:
            pat = re.search(rf"(\S+(?:\s+\S+)*?)\s+({code})\b", desc)
            if pat:
                result["Geography"] = f"{pat.group(1)} {pat.group(2)}"
                break

    # Product: first meaningful tokens
    product_parts = []
    stop = {"FOB", "CIF", "C+F", "Dlvd", "Barge", "Cargo", "Waterborne",
            "Swap", "vs", "Global", "Mo01", "Mo02"} | set(_GEO_KEYWORDS.keys())
    for t in tokens:
        if t in stop or _MO_PATTERN.match(t) or _MO_PATTERN2.match(t):
            break
        product_parts.append(t)
    result["Product"] = " ".join(product_parts) if product_parts else "Unspecified"

    return result


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def _parse_single(desc: str) -> dict:
    """Parse a single DESCRIPTION string into 6 categorical features.

    Returns a dict with keys:
        Product, Grade, Geography, Delivery, Timing, IS_SPOT
    """
    if not desc or not isinstance(desc, str):
        return _default()

    tokens = desc.split()
    first = tokens[0] if tokens else ""

    # 1. DTN Wholesale Rack
    if first == "DTN":
        return _parse_dtn(tokens)

    # 2. NYMEX / Nymex / CME Financial
    if first in ("NYMEX", "Nymex", "CME"):
        return _parse_nymex(desc, tokens)

    # 3. Intraday (NYMEX wrapper)
    if first == "Intraday":
        return _parse_nymex(desc, tokens)

    # 4. ICE Financial
    if first == "ICE":
        return _parse_ice(desc, tokens)

    # 5. Enterprise Singapore / ES
    if desc.startswith("Enterprise Singapore") or (first == "ES" and len(tokens) > 1):
        return _parse_singapore(desc)

    # 6. Pipeline
    if "Pipeline" in desc or "Colonial" in desc.split()[0] if tokens else False:
        return _parse_pipeline(desc)

    # 7. Gov Statistics
    if first in ("CFTC", "EIA", "ECB"):
        return _parse_gov_stats(desc, tokens)

    # 8. Swap (financial)
    if "Swap" in desc:
        r = _parse_intl_physical(desc, tokens)
        r["IS_SPOT"] = "No"
        return r

    # 9. International Physical Cargo (broad catch)
    physical_prefixes = (
        "Marine", "Bunker", "Jet", "Gasoline", "Gasoil", "ULSD", "FO",
        "Naphtha", "Propane", "Butane", "VGO", "LPG", "MTBE", "Low",
        "Straight", "ULS", "Natural", "US", "USGC",
    )
    if first in physical_prefixes:
        return _parse_intl_physical(desc, tokens)

    # 10. Fallback
    return _parse_fallback(desc, tokens)


# ---------------------------------------------------------------------------
# Vectorized batch handler (for local / Snowpark UDF)
# ---------------------------------------------------------------------------

def parse_description_batch(series: pd.Series) -> pd.Series:
    """Vectorized UDF handler: parse a batch of DESCRIPTION strings.

    Args:
        series: pandas Series of description strings.

    Returns:
        pandas Series of dicts, each with 6 keys.
    """
    return series.apply(_parse_single)


# ---------------------------------------------------------------------------
# Snowflake UDF SQL — self-contained Python code that runs server-side
# ---------------------------------------------------------------------------

def get_udf_sql(udf_fqn: str) -> str:
    """Return the CREATE FUNCTION SQL to register the parser UDF in Snowflake.

    The UDF runs entirely inside Snowflake's Python sandbox — no data leaves.
    The parsing logic here mirrors _parse_single() above.

    Args:
        udf_fqn: Fully-qualified UDF name, e.g. 'MYDB.PUBLIC.PARSE_DESCRIPTION'
    """
    return f"""
CREATE OR REPLACE FUNCTION {udf_fqn}(desc VARCHAR)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('regex')
HANDLER = 'parse'
AS
$$
import re, json

_US_STATES = {{
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY",
}}
_CA_PROV = {{
    "AB","BC","MB","NB","NL","NS","NT","NU","ON","PE","QC","SK","YT",
}}
_STATES = _US_STATES | _CA_PROV

_DTN_PROD = {{
    "Unl":"Unleaded Gasoline","ULSD":"Diesel","Kero":"Kerosene",
    "Jet":"Jet Fuel","Furnace":"Furnace Oil","No2/Diesel":"No. 2 Diesel",
    "Stove":"Stove Oil","AvGas":"Aviation Gasoline","No2":"No. 2 Oil",
    "Ethanol":"Ethanol","Diesel":"Diesel",
}}
_DTN_GRADE = {{"Reg":"Regular","Mid":"Midgrade","Prem":"Premium"}}
_DTN_SUFF = {{
    "BrAvg","UnbAvg","RkAvg","BrMAvg","UnbMAvg","RkMAvg","Br","Unb",
}}
_BR = {{"BrAvg","Br","BrMAvg"}}
_UNB = {{"UnbAvg","Unb","UnbMAvg"}}
_RK = {{"RkAvg","RkMAvg"}}

_MO = re.compile(r"(\\d+)-Mo")
_MO2 = re.compile(r"Mo(\\d+)")
_MY = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+(\\d{{4}})")

_GEO = {{
    "Arab Gulf":"Arab Gulf","NWE":"Northwest Europe","MED":"Mediterranean",
    "Med":"Mediterranean","USGC":"US Gulf Coast","USAC":"US Atlantic Coast",
    "Spore":"Singapore","Singapore":"Singapore","Japan":"Japan",
    "Korea":"South Korea","Australia":"Australia","Fujairah":"Fujairah",
    "South China":"South China","West Japan":"West Japan","Europe":"Europe",
    "FARAG":"FARAG (NW Europe)","Mt Belvieu":"Mt Belvieu TX",
}}

def _default():
    return {{"Product":"Unspecified","Grade":"Generic","Geography":"Unspecified",
            "Delivery":"Unspecified","Timing":"Spot","IS_SPOT":"Yes"}}

def _parse_dtn(tokens):
    r = _default()
    r["Delivery"] = "Rack Terminal"
    if len(tokens) < 3:
        r["Product"] = " ".join(tokens[1:])
        return r
    pt = tokens[1]
    r["Product"] = _DTN_PROD.get(pt, pt)
    idx = 2
    if pt == "Unl" and idx < len(tokens) and tokens[idx] in _DTN_GRADE:
        r["Grade"] = _DTN_GRADE[tokens[idx]]
        idx += 1
    elif pt == "ULSD":
        r["Grade"] = "Ultra-Low Sulfur"
    elif pt == "No2/Diesel" and idx < len(tokens) and tokens[idx] == "HS":
        r["Grade"] = "High Sulfur"
        idx += 1
    elif pt == "Jet" and idx < len(tokens) and tokens[idx] == "Fuel":
        idx += 1
    suf = tokens[-1]
    if suf not in _DTN_SUFF:
        r["Geography"] = " ".join(tokens[idx:]) if tokens[idx:] else "Unspecified"
        return r
    if suf in _BR:
        r["Grade"] += " (Branded)"
    elif suf in _UNB:
        r["Grade"] += " (Unbranded)"
    elif suf in _RK:
        r["Grade"] += " (Rack Average)"
    rem = tokens[idx:-1]
    while rem and rem[0] in ("CARB","RFG"):
        rem.pop(0)
    sp = None
    for i in range(len(rem)-1, -1, -1):
        if rem[i] in _STATES:
            sp = i
            break
    if sp is not None:
        city = " ".join(rem[:sp])
        r["Geography"] = (city + " " + rem[sp]).strip()
    elif rem:
        r["Geography"] = " ".join(rem)
    return r

def _parse_nymex(desc, tokens):
    r = _default()
    r["Delivery"] = "Financial Exchange"
    r["IS_SPOT"] = "No"
    m = _MO.search(desc)
    m2 = _MO2.search(desc)
    my = _MY.search(desc)
    if m:
        r["Timing"] = f"{{int(m.group(1))}}-Month Forward"
    elif m2:
        r["Timing"] = f"{{int(m2.group(1))}}-Month Forward"
    elif my:
        r["Timing"] = f"{{my.group(1)}} {{my.group(2)}} Forward"
    else:
        r["Timing"] = "Forward Month"
    d = desc.upper()
    if "WTI" in d:
        r["Product"],r["Grade"],r["Geography"] = "Crude Oil","WTI","Cushing OK"
    elif "RBOB" in d:
        r["Product"],r["Geography"] = "RBOB Gasoline","New York"
    elif "NY ULSD" in d:
        r["Product"],r["Grade"],r["Geography"] = "Diesel","Ultra-Low Sulfur","New York"
    elif "ULSD" in d:
        r["Product"],r["Grade"] = "Diesel","Ultra-Low Sulfur"
    else:
        skip = 1
        if tokens[0].upper() == "INTRADAY":
            skip = 2
        if len(tokens) > skip and tokens[skip].upper() == "GLOBEX":
            skip += 1
        pp = []
        for t in tokens[skip:]:
            if _MO.match(t) or _MO2.match(t) or t in ("Floor","Elect","Comb","Fin","Pen","Calendar","Swap"):
                break
            pp.append(t)
        r["Product"] = " ".join(pp) if pp else "Unspecified"
    last = tokens[-1] if tokens else ""
    if last in ("Floor","Elect","Comb"):
        r["Delivery"] = f"Financial Exchange ({{last}})"
    return r

def _parse_ice(desc, tokens):
    r = _default()
    r["Delivery"],r["IS_SPOT"],r["Geography"] = "Financial Exchange (ICE)","No","Europe"
    m = _MO.search(desc)
    r["Timing"] = f"{{int(m.group(1))}}-Month Forward" if m else "Forward Month"
    d = desc.upper()
    if "GAS OIL" in d or "GASOIL" in d:
        r["Product"] = "Gas Oil"
    elif "BRENT" in d:
        r["Product"],r["Grade"] = "Crude Oil","Brent"
    else:
        pp = []
        for t in tokens[1:]:
            if _MO.match(t) or t in ("Comb","Floor","Elect"):
                break
            pp.append(t)
        r["Product"] = " ".join(pp) if pp else "Unspecified"
    return r

def _parse_sg(desc):
    r = _default()
    r["Delivery"],r["Geography"] = "Trade Statistics","Singapore"
    if desc.startswith("Enterprise Singapore "):
        rest = desc[len("Enterprise Singapore "):]
    elif desc.startswith("ES "):
        rest = desc[3:]
    else:
        r["Product"] = desc
        return r
    parts = rest.split(" Singapore", 1)
    r["Product"] = parts[0].strip() or "Unspecified"
    if len(parts) > 1 and parts[1].strip():
        cp = []
        for t in parts[1].strip().split():
            if t in ("Imp","Exp","ReExp","Vol","Val","Dom","Tot"):
                break
            cp.append(t)
        if cp:
            r["Geography"] = f"Singapore -> {{' '.join(cp)}}"
    return r

def _parse_pipe(desc):
    r = _default()
    r["Delivery"],r["IS_SPOT"] = "Pipeline","Yes"
    cm = re.search(r"Cycle\\s*(\\d+)", desc)
    if cm:
        r["Timing"] = f"Cycle {{cm.group(1)}}"
    elif "prompt" in desc.lower():
        r["Timing"] = "Prompt"
    elif "cycle" in desc.lower():
        r["Timing"] = "Cycle"
    else:
        r["Timing"] = "Spot"
    if "USGC" in desc:
        r["Geography"] = "US Gulf Coast"
    elif "USAC" in desc:
        r["Geography"] = "US Atlantic Coast"
    elif "Colonial" in desc:
        r["Geography"] = "Colonial Pipeline"
    d = desc
    for kw in ("Pipeline","USGC","USAC","Colonial","Waterborne","Assessment","Cycle","Prompt","Differential"):
        d = d.split(kw)[0]
    r["Product"] = d.strip() or "Unspecified"
    if "ULS" in desc:
        r["Grade"] = "Ultra-Low Sulfur"
    return r

def _parse_phys(desc, tokens):
    r = _default()
    r["Delivery"],r["Timing"] = "Physical Cargo","Spot"
    if "Swap" in desc:
        r["Delivery"],r["IS_SPOT"] = "Swap","No"
    m = _MO.search(desc) or _MO2.search(desc)
    if m:
        r["Timing"] = f"{{int(m.group(1))}}-Month Forward"
        r["IS_SPOT"] = "No"
    for kw in ("FOB","CIF","C+F","Dlvd","DAP","DES","CFR"):
        if kw in desc:
            r["Delivery"] = f"Physical {{kw}}"
            break
    if "Waterborne" in desc:
        r["Delivery"] = "Physical Waterborne"
    elif "Barge" in desc:
        r["Delivery"] = "Physical Barge"
    for gk, gv in _GEO.items():
        if gk in desc:
            r["Geography"] = gv
            break
    stop = {{"FOB","CIF","C+F","Dlvd","DAP","Cargo","Barge","Waterborne","Swap","vs","Global","strip"}} | set(_GEO.keys())
    pp = []
    for t in tokens:
        if t in stop or _MO.match(t) or _MO2.match(t):
            break
        pp.append(t)
    r["Product"] = " ".join(pp) if pp else tokens[0]
    sm = re.search(r"(\\d+\\.?\\d*%?\\s*S(?:ulfur)?|\\d+ppm)", desc)
    if sm:
        r["Grade"] = sm.group(1).strip()
    cm = re.search(r"(\\d+)\\s*CST", desc)
    if cm:
        r["Grade"] = f"{{cm.group(1)}} CST"
    return r

def _parse_gov(tokens):
    r = _default()
    r["Delivery"],r["Geography"] = "Government Report",tokens[0]
    r["Product"] = " ".join(tokens[1:]) if len(tokens) > 1 else tokens[0]
    return r

def _parse_fb(desc, tokens):
    r = _default()
    if "Dollar" in desc or "EUR" in desc or "AUD" in desc:
        return {{"Product":"Currency","Grade":desc,"Geography":"Unspecified",
                "Delivery":"Financial Exchange","Timing":"Spot","IS_SPOT":"No"}}
    m = _MO.search(desc) or _MO2.search(desc)
    if m:
        r["IS_SPOT"],r["Timing"] = "No",f"{{int(m.group(1))}}-Month Forward"
        r["Delivery"] = "Financial Exchange"
    for kw in ("FOB","CIF","C+F","Dlvd","DAP","Barge","Cargo","Waterborne"):
        if kw in desc:
            r["Delivery"],r["IS_SPOT"],r["Timing"] = f"Physical {{kw}}","Yes","Spot"
            break
    for gk, gv in _GEO.items():
        if gk in desc:
            r["Geography"] = gv
            break
    pp = []
    stop = {{"FOB","CIF","C+F","Dlvd","Barge","Cargo","Waterborne","Swap","vs","Global"}} | set(_GEO.keys())
    for t in tokens:
        if t in stop or _MO.match(t) or _MO2.match(t):
            break
        pp.append(t)
    r["Product"] = " ".join(pp) if pp else "Unspecified"
    return r

def parse(desc):
    if not desc:
        return _default()
    tokens = desc.split()
    f = tokens[0]
    if f == "DTN":
        return _parse_dtn(tokens)
    if f in ("NYMEX","Nymex","CME","Intraday"):
        return _parse_nymex(desc, tokens)
    if f == "ICE":
        return _parse_ice(desc, tokens)
    if desc.startswith("Enterprise Singapore") or (f == "ES" and len(tokens) > 1):
        return _parse_sg(desc)
    if "Pipeline" in desc or f == "Colonial":
        return _parse_pipe(desc)
    if f in ("CFTC","EIA","ECB"):
        return _parse_gov(tokens)
    if "Swap" in desc:
        r = _parse_phys(desc, tokens)
        r["IS_SPOT"] = "No"
        return r
    phys = ("Marine","Bunker","Jet","Gasoline","Gasoil","ULSD","FO",
            "Naphtha","Propane","Butane","VGO","LPG","MTBE","Low",
            "Straight","ULS","Natural","US","USGC")
    if f in phys:
        return _parse_phys(desc, tokens)
    return _parse_fb(desc, tokens)
$$;
"""


def get_ctas_sql(
    udf_fqn: str,
    source_table: str,
    target_fqn: str,
    *,
    limit: int | None = None,
) -> str:
    """Return the CREATE TABLE AS SELECT SQL that applies the UDF.

    Args:
        udf_fqn:      Fully-qualified UDF name.
        source_table:  Fully-qualified source table.
        target_fqn:    Fully-qualified target table.
        limit:         Optional row limit.
    """
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    return f"""
    CREATE OR REPLACE TABLE {target_fqn} AS
    SELECT
        T.*,
        P.value:Product::VARCHAR   AS PRODUCT,
        P.value:Grade::VARCHAR     AS GRADE,
        P.value:Geography::VARCHAR AS GEOGRAPHY,
        P.value:Delivery::VARCHAR  AS DELIVERY,
        P.value:Timing::VARCHAR    AS TIMING,
        P.value:IS_SPOT::VARCHAR   AS IS_SPOT
    FROM (
        SELECT *, {udf_fqn}(DESCRIPTION) AS PARSED
        FROM {source_table}
        {limit_clause}
    ) T,
    LATERAL FLATTEN(input => ARRAY_CONSTRUCT(T.PARSED)) P
    """
