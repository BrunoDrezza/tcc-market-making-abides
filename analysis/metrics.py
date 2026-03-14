"""
analysis/metrics.py
Computes inventory and pricing statistics from parsed AS quote data.
"""

import numpy as np
import pandas as pd


def calculate_inventory_stats(parsed_df: pd.DataFrame) -> dict:
    """Compute summary statistics for the inventory trajectory.

    Parameters
    ----------
    parsed_df : pd.DataFrame
        Output of :func:`analysis.parser.parse_as_metrics` with an ``inv`` column.

    Returns
    -------
    dict
        Keys: max_inventory, min_inventory, mean_inventory,
              std_inventory, final_inventory.
    """
    inv = parsed_df["inv"]
    return {
        "max_inventory": int(inv.max()),
        "min_inventory": int(inv.min()),
        "mean_inventory": float(inv.mean()),
        "std_inventory": float(inv.std(ddof=1)),
        "final_inventory": int(inv.iloc[-1]),
    }


def print_stats(stats_dict: dict) -> None:
    """Pretty-print a statistics dictionary to the console."""
    print("\n" + "=" * 50)
    print("  Avellaneda-Stoikov — Inventory Statistics")
    print("=" * 50)
    for key, value in stats_dict.items():
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            print(f"  {label:<25s}: {value:>12.4f}")
        else:
            print(f"  {label:<25s}: {value:>12}")
    print("=" * 50 + "\n")
