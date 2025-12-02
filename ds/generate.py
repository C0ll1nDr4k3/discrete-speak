import argparse
import os
from datetime import datetime, timedelta
from typing import List

import torch
from alpaca.data.timeframe import TimeFrame

from ds.bars import Conversion
from ds.config import Config
from ds.data import CurveTokenizer
from ds.discretization import Discretizer
from ds.fit import fit
from ds.retrieval import Security
from ds.thresholds import Threshold


def save_dataset_from_results(results: dict, config: Config, output_path: str):
    """
    Tokenizes results and saves them to a file.
    """
    # Initialize tokenizer
    discretizer = Discretizer(
        min_power=config.discretization_log_min_power,
        max_power=config.discretization_log_max_power,
        points_per_magnitude=config.discretization_points_per_magnitude
    )
    tokenizer = CurveTokenizer(discretizer)
    
    dataset_data = []
    
    print("Tokenizing results...")
    for symbol, data in results.items():
        labels = data.get("labels", [])
        symbol_tokens = []
        for label in labels:
            try:
                tokens = tokenizer.tokenize(label)
                symbol_tokens.append(tokens)
            except ValueError as e:
                print(f"Skipping label due to error: {e}")
                continue
        
        if symbol_tokens:
            dataset_data.append({
                "symbol": symbol,
                "tokens": torch.tensor(symbol_tokens, dtype=torch.long) # [NumSegments, 5]
            })
            
    print(f"Saving dataset to {output_path}...")
    torch.save(dataset_data, output_path)
    print("Done.")

def generate_dataset(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    output_path: str,
    config_overrides: dict = None
):
    """
    Generates a dataset of tokenized curves for the given symbols and date range.
    """
    print(f"Generating dataset for {len(symbols)} symbols from {start_date} to {end_date}...")
    
    # Default config
    config_kwargs = {
        "start": start_date,
        "end": end_date,
        "step": TimeFrame.Minute,
        "plot_enabled": False, # Disable plotting for speed
        "conversion": Conversion.DOLLAR,
        "threshold_strategy": Threshold.EMA,
        "sg_window_length": 5,
        "sg_polyorder": 2,
    }
    
    if config_overrides:
        config_kwargs.update(config_overrides)
        
    config = Config(**config_kwargs)
    
    # Run fit pipeline
    print("Running fit pipeline...")
    # Note: fit returns a dict of symbol -> results
    results = fit(symbols, Security.EQUITIES, config)
    
    save_dataset_from_results(results, config, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate curve dataset.")
    parser.add_argument("--symbols", nargs="+", required=True, help="List of symbols")
    parser.add_argument("--start", type=str, required=False, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), defaults to now")
    parser.add_argument("--output", type=str, default="dataset.pt", help="Output file path")
    parser.add_argument("--days", type=int, default=100, help="Number of days of data to fetch")
    
    args = parser.parse_args()
    
    end_date = datetime.now()
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
        
    start_date = datetime.now() - timedelta(days=7)
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    elif args.days:
        start_date = end_date - timedelta(days=args.days)
        
    generate_dataset(args.symbols, start_date, end_date, args.output)
