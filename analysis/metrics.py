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


def calculate_risk_metrics(
    parsed_df: pd.DataFrame, periods_per_year: float = 252.0 * 6.5 * 3600.0
) -> dict:
    """Compute risk-adjusted performance metrics from the equity curve.

    Parameters
    ----------
    parsed_df : pd.DataFrame
        Output of :func:`analysis.parser.parse_as_metrics`, containing the
        ``Equity`` and ``Returns`` columns.
    periods_per_year : float, optional
        Number of observations per year used to annualise the Sharpe ratio.
        The default assumes one observation per second over a 6.5-hour
        session across 252 trading days. The factor is constant across
        simulations, so parameter rankings are invariant to this choice.

    Returns
    -------
    dict
        Keys: sharpe_annualised, sharpe_per_obs, volatility_per_obs,
              max_drawdown_pct, total_return_pct, final_pnl_usd.

    Notes
    -----
    ``sharpe_per_obs`` is the raw mean-to-standard-deviation ratio of the
    log-return series and carries no annualisation assumption; prefer it
    when comparing configurations within a single experimental design.
    A degenerate return series (zero variance) yields a NaN Sharpe rather
    than an infinite one.
    """
    returns = parsed_df["Returns"]
    equity = parsed_df["Equity"]

    mean_r = float(returns.mean())
    std_r = float(returns.std(ddof=1))

    if std_r == 0.0 or np.isnan(std_r):
        sharpe_per_obs = float("nan")
    else:
        sharpe_per_obs = mean_r / std_r

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    return {
        "sharpe_annualised": sharpe_per_obs * np.sqrt(periods_per_year),
        "sharpe_per_obs": sharpe_per_obs,
        "volatility_per_obs": std_r,
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "total_return_pct": float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0),
        "final_pnl_usd": float(parsed_df["PnL"].iloc[-1]),
    }


def calculate_sigma2_stats(parsed_df: pd.DataFrame) -> dict:
    """Summarise the estimated variance series, for kill-switch calibration.

    Parameters
    ----------
    parsed_df : pd.DataFrame
        Output of :func:`analysis.parser.parse_as_metrics` with a ``sigma2``
        column, expressed in squared dollars.

    Returns
    -------
    dict
        Median, and the 90th, 99th and 99.9th percentiles of sigma^2, plus
        its maximum. The 99th percentile is the reference threshold for the
        entropy kill-switch (layer 3).
    """
    sigma2 = parsed_df["sigma2"]
    return {
        "sigma2_median": float(sigma2.median()),
        "sigma2_p90": float(sigma2.quantile(0.90)),
        "sigma2_p99": float(sigma2.quantile(0.99)),
        "sigma2_p999": float(sigma2.quantile(0.999)),
        "sigma2_max": float(sigma2.max()),
    }


def print_stats(stats_dict: dict) -> None:
    """Pretty-print a statistics dictionary to the console."""
    print("\n" + "=" * 50)
    print("  Avellaneda-Stoikov — Inventory Statistics")
    print("=" * 50)
    for key, value in stats_dict.items():
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            # Notacao cientifica para grandezas de magnitude muito pequena
            # (e.g. sigma^2), que a formatacao decimal fixa truncaria a zero.
            if value != 0.0 and abs(value) < 1e-3:
                print(f"  {label:<25s}: {value:>12.4e}")
            else:
                print(f"  {label:<25s}: {value:>12.4f}")
        else:
            print(f"  {label:<25s}: {value:>12}")
    print("=" * 50 + "\n")
