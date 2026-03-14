"""
run_analysis.py
CLI entry point for the Avellaneda-Stoikov post-simulation analysis pipeline.
"""

import argparse
import os

from analysis.parser import load_agent_log, parse_as_metrics
from analysis.metrics import calculate_inventory_stats, print_stats
from analysis.plotting import plot_inventory_dynamics, plot_pricing_dynamics, plot_spread


def main():
    parser = argparse.ArgumentParser(
        description="Avellaneda-Stoikov Market Maker — Post-Simulation Analysis"
    )
    parser.add_argument(
        "--log_dir",
        default="log/TCC_AS_Experiment",
        help="Path to the ABIDES log directory (default: log/TCC_AS_Experiment)",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis_output",
        help="Directory for output PNGs (default: analysis_output)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading agent log from: {args.log_dir}")
    raw_df = load_agent_log(args.log_dir)

    print("Parsing AS quote metrics...")
    parsed_df = parse_as_metrics(raw_df)
    print(f"  {len(parsed_df)} quote observations extracted.")

    stats = calculate_inventory_stats(parsed_df)
    print_stats(stats)

    print("Generating plots...")
    plot_inventory_dynamics(parsed_df, args.output_dir)
    plot_pricing_dynamics(parsed_df, args.output_dir)
    plot_spread(parsed_df, args.output_dir)

    print(f"All plots saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
