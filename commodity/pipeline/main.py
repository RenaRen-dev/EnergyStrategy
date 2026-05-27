"""
Single entry point for the Energy Commodity Data Pipeline.
Use this orchestrator to run the entire pipeline or specific steps.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import data_decoding
from pipeline import normalization
from pipeline import revin_scaling

def parse_args():
    parser = argparse.ArgumentParser(description="Energy Commodity ELT Pipeline")
    parser.add_argument("--skip-decoding", action="store_true", help="Skip the initial 163M row string decoding phase")
    parser.add_argument("--skip-normalization", action="store_true", help="Skip the Physical & Financial Normalization phase")
    parser.add_argument("--skip-fx-ingest", action="store_true", help="Skip the FRED FX ingestion (reuse existing FX table)")
    parser.add_argument("--skip-scaling", action="store_true", help="Skip the dense grid RevIN scaling phase")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=================================================================")
    print("  ENERGY COMMODITY DATA PIPELINE")
    print("=================================================================")

    # 1. Data Decoding Phase
    if not args.skip_decoding:
        data_decoding.run()
    else:
        print("[SKIP] Bypassing Data Decoding Phase...")

    # 2. Normalization Phase (Physical & Financial)
    if not args.skip_normalization:
        normalization.run(skip_fx_ingest=args.skip_fx_ingest)
    else:
        print("[SKIP] Bypassing Normalization Phase...")

    # 3. Scaling Phase (Dense Grid + RevIN)
    if not args.skip_scaling:
        revin_scaling.run()
    else:
        print("[SKIP] Bypassing Scaling Phase...")

    print("=================================================================")
    print("  PIPELINE EXECUTION COMPLETE")
    print("=================================================================")

if __name__ == "__main__":
    main()
