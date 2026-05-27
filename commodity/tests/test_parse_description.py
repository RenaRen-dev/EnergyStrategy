"""
Unit tests for parse_description_udf._parse_single().

Run:  python -m pytest tests/test_parse_description.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from utility.parse_description_udf import _parse_single


# ── DTN Wholesale Rack ───────────────────────────────────────────────────────

class TestDTN:
    def test_unl_reg_branded_avg(self):
        r = _parse_single("DTN Unl Reg Pittsburgh PA BrAvg")
        assert r["Product"] == "Unleaded Gasoline"
        assert r["Grade"] == "Regular (Branded)"
        assert r["Geography"] == "Pittsburgh PA"
        assert r["Delivery"] == "Rack Terminal"
        assert r["Timing"] == "Spot"
        assert r["IS_SPOT"] == "Yes"

    def test_unl_prem_unbranded(self):
        r = _parse_single("DTN Unl Prem Pittsburgh PA Gulf Unb")
        assert r["Product"] == "Unleaded Gasoline"
        assert r["Grade"] == "Premium (Unbranded)"
        assert r["Geography"] == "Pittsburgh PA"
        assert r["IS_SPOT"] == "Yes"

    def test_unl_mid_rack_avg(self):
        r = _parse_single("DTN Unl Mid Chattanooga TN RkAvg")
        assert r["Product"] == "Unleaded Gasoline"
        assert r["Grade"] == "Midgrade (Rack Average)"
        assert r["Geography"] == "Chattanooga TN"

    def test_ulsd_rack_avg(self):
        r = _parse_single("DTN ULSD Phoenix AZ RkAvg")
        assert r["Product"] == "Diesel"
        assert r["Grade"] == "Ultra-Low Sulfur (Rack Average)"
        assert r["Geography"] == "Phoenix AZ"

    def test_kero_branded_avg(self):
        r = _parse_single("DTN Kero Detroit MI BrAvg")
        assert r["Product"] == "Kerosene"
        assert "Branded" in r["Grade"]
        assert r["Geography"] == "Detroit MI"

    def test_jet_rack_avg(self):
        r = _parse_single("DTN Jet Anchorage AK RkAvg")
        assert r["Product"] == "Jet Fuel"
        assert r["Geography"] == "Anchorage AK"

    def test_jet_fuel_rack_avg(self):
        r = _parse_single("DTN Jet Fuel Lake Charles LA RkAvg")
        assert r["Product"] == "Jet Fuel"
        assert r["Geography"] == "Lake Charles LA"

    def test_rfg_prefix(self):
        r = _parse_single("DTN Unl Reg RFG Baltimore MD BrAvg")
        assert r["Geography"] == "Baltimore MD"

    def test_carb_rfg_prefix(self):
        r = _parse_single("DTN Unl Reg CARB RFG Los Angeles CA BrAvg")
        assert r["Geography"] == "Los Angeles CA"

    def test_canadian_province(self):
        r = _parse_single("DTN Unl Reg Calgary AB RkAvg")
        assert r["Geography"] == "Calgary AB"

    def test_multi_word_city(self):
        r = _parse_single("DTN Unl Reg Minn/StPaul MN BrAvg")
        assert r["Geography"] == "Minn/StPaul MN"

    def test_no2_diesel_hs(self):
        r = _parse_single("DTN No2/Diesel HS Hartford/Woodriver IL BrAvg")
        assert r["Product"] == "No. 2 Diesel"
        assert r["Grade"] == "High Sulfur (Branded)"

    def test_furnace_oil(self):
        r = _parse_single("DTN Furnace Oil Portland ME RkAvg")
        assert r["Product"] == "Furnace Oil"

    def test_supplier_name_stripped(self):
        """Supplier names should not pollute geography."""
        r = _parse_single("DTN Unl Reg Pittsburgh PA Exxon Br")
        assert r["Geography"] == "Pittsburgh PA"

    def test_ulsd_carb(self):
        r = _parse_single("DTN ULSD CARB Bakersfield CA RkAvg")
        assert r["Product"] == "Diesel"
        assert r["Geography"] == "Bakersfield CA"


# ── NYMEX Financial ──────────────────────────────────────────────────────────

class TestNYMEX:
    def test_ny_ulsd_forward(self):
        r = _parse_single("NYMEX NY ULSD 03-Mo Floor")
        assert r["Product"] == "Diesel"
        assert r["Timing"] == "3-Month Forward"
        assert r["IS_SPOT"] == "No"
        assert "Financial" in r["Delivery"]

    def test_rbob(self):
        r = _parse_single("NYMEX RBOB 01-Mo Floor")
        assert r["Product"] == "RBOB Gasoline"
        assert r["Timing"] == "1-Month Forward"
        assert r["IS_SPOT"] == "No"

    def test_wti(self):
        r = _parse_single("NYMEX WTI 05-Mo Floor")
        assert r["Product"] == "Crude Oil"
        assert r["Grade"] == "WTI"
        assert r["Geography"] == "Cushing OK"
        assert r["IS_SPOT"] == "No"

    def test_globex(self):
        r = _parse_single("NYMEX Globex NY ULSD 01-Mo Elect")
        assert r["Product"] == "Diesel"
        assert r["Timing"] == "1-Month Forward"
        assert "Financial" in r["Delivery"]

    def test_named_month(self):
        r = _parse_single("NYMEX NY ULSD Mar 2020 Comb")
        assert "Mar 2020" in r["Timing"]
        assert r["IS_SPOT"] == "No"

    def test_fin_penultimate(self):
        r = _parse_single("NYMEX NY ULSD Fin Pen 25-Mo Comb")
        assert r["Timing"] == "25-Month Forward"
        assert r["IS_SPOT"] == "No"

    def test_intraday(self):
        r = _parse_single("Intraday NYMEX RBOB Mo01")
        assert r["Product"] == "RBOB Gasoline"
        assert r["IS_SPOT"] == "No"


# ── ICE Financial ────────────────────────────────────────────────────────────

class TestICE:
    def test_gas_oil(self):
        r = _parse_single("ICE Gas Oil 02-Mo Comb")
        assert r["Product"] == "Gas Oil"
        assert r["Timing"] == "2-Month Forward"
        assert r["IS_SPOT"] == "No"
        assert r["Geography"] == "Europe"


# ── Enterprise Singapore ─────────────────────────────────────────────────────

class TestSingapore:
    def test_enterprise_singapore(self):
        r = _parse_single(
            "Enterprise Singapore Asphalt & Tar Pitch Singapore Malaysia Imp Val Tot"
        )
        assert r["Product"] == "Asphalt & Tar Pitch"
        assert "Singapore" in r["Geography"]
        assert r["Delivery"] == "Trade Statistics"
        assert r["IS_SPOT"] == "Yes"

    def test_es_prefix(self):
        r = _parse_single("ES Lube Grease Singapore Netherlands Exp Vol Dom")
        assert r["Product"] == "Lube Grease"
        assert "Singapore" in r["Geography"]
        assert r["IS_SPOT"] == "Yes"

    def test_es_trade_partner(self):
        r = _parse_single(
            "ES Other Med Raw Oils ex Biodsl & Waste Oil Singapore India Imp Vol Tot"
        )
        assert "India" in r["Geography"]


# ── Pipeline ─────────────────────────────────────────────────────────────────

class TestPipeline:
    def test_usgc_cycle(self):
        r = _parse_single("ULS Heating Oil USGC Pipeline Assessment Cycle01")
        assert r["Delivery"] == "Pipeline"
        assert r["Geography"] == "US Gulf Coast"
        assert r["Timing"] == "Cycle 01"
        assert r["IS_SPOT"] == "Yes"

    def test_prompt(self):
        r = _parse_single("ULS Heating Oil USGC Prompt Pipeline Cycle")
        assert r["Delivery"] == "Pipeline"
        assert "Prompt" in r["Timing"] or "Cycle" in r["Timing"]

    def test_jet_pipeline_city(self):
        r = _parse_single("Jet Kero San Francisco CA Pipeline")
        assert r["Delivery"] == "Pipeline"


# ── International Physical ───────────────────────────────────────────────────

class TestIntlPhysical:
    def test_jet_kero_fob(self):
        r = _parse_single("Jet Kero FOB Arab Gulf Cargo")
        assert "Jet" in r["Product"]
        assert "Arab Gulf" in r["Geography"]
        assert r["IS_SPOT"] == "Yes"

    def test_marine_gasoil(self):
        r = _parse_single("Marine Gasoil 0.5% Dlvd Singapore")
        assert "Marine" in r["Product"]
        assert r["Geography"] == "Singapore"

    def test_bunker_fo(self):
        r = _parse_single("Bunker FO 380 CST FOB Singapore")
        assert "Bunker" in r["Product"]
        assert r["Geography"] == "Singapore"

    def test_swap_is_not_spot(self):
        r = _parse_single("FO 180 CST MOPAG Swap 17:30 SGT Mo02")
        assert r["IS_SPOT"] == "No"


# ── Gov Statistics ───────────────────────────────────────────────────────────

class TestGovStats:
    def test_cftc(self):
        r = _parse_single("CFTC Nymex WTI Net Spec Positions")
        assert r["Delivery"] == "Government Report"
        assert r["Geography"] == "CFTC"

    def test_eia(self):
        r = _parse_single("EIA 9 US Crude Oil Production")
        assert r["Delivery"] == "Government Report"


# ── Fallback ─────────────────────────────────────────────────────────────────

class TestFallback:
    def test_currency(self):
        r = _parse_single("AUD-US Dollar")
        assert r["Product"] == "Currency"
        assert r["IS_SPOT"] == "No"

    def test_null_input(self):
        r = _parse_single("")
        assert r["Product"] == "Unspecified"

    def test_none_input(self):
        r = _parse_single(None)
        assert r["Product"] == "Unspecified"


# ── All results have all 6 keys ──────────────────────────────────────────────

SAMPLE_DESCS = [
    "DTN Unl Reg Pittsburgh PA BrAvg",
    "NYMEX NY ULSD 03-Mo Floor",
    "ICE Gas Oil 02-Mo Comb",
    "Enterprise Singapore Asphalt & Tar Pitch Singapore Malaysia Imp Val Tot",
    "ULS Heating Oil USGC Pipeline Assessment Cycle01",
    "Jet Kero FOB Arab Gulf Cargo",
    "CFTC Nymex WTI Net Spec Positions",
    "AUD-US Dollar",
    "",
]

REQUIRED_KEYS = {"Product", "Grade", "Geography", "Delivery", "Timing", "IS_SPOT"}


@pytest.mark.parametrize("desc", SAMPLE_DESCS)
def test_all_keys_present(desc):
    r = _parse_single(desc)
    assert set(r.keys()) == REQUIRED_KEYS


@pytest.mark.parametrize("desc", SAMPLE_DESCS)
def test_no_none_values(desc):
    r = _parse_single(desc)
    for k, v in r.items():
        assert v is not None, f"Key {k} is None for desc={desc!r}"
        assert v != "", f"Key {k} is empty for desc={desc!r}"
