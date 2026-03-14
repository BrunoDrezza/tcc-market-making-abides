"""
analysis/plotting.py
Academic-grade Matplotlib/Seaborn charts for the Avellaneda-Stoikov analysis.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_inventory_dynamics(parsed_df, output_dir):
    """Plot inventory trajectory over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(parsed_df.index, parsed_df["inv"], linewidth=0.8)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.7)
    ax.set_title("Evolução do Inventário do Agente A-S", fontsize=14)
    ax.set_xlabel("Tempo")
    ax.set_ylabel(r"Inventário ($q_t$)")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "inventory_dynamics.png"), dpi=300)
    plt.close(fig)


def plot_pricing_dynamics(parsed_df, output_dir):
    """Plot mid-price, bid, and ask over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(parsed_df.index, parsed_df["mid"], color="black",
            linestyle="--", linewidth=0.7, label="Mid-Price")
    ax.plot(parsed_df.index, parsed_df["bid"], color="blue",
            linewidth=0.6, label="Bid")
    ax.plot(parsed_df.index, parsed_df["ask"], color="red",
            linewidth=0.6, label="Ask")
    ax.set_title("Dinâmica de Cotação: Mid-Price vs Bid/Ask", fontsize=14)
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Preço (Cents)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "pricing_dynamics.png"), dpi=300)
    plt.close(fig)


def plot_spread(parsed_df, output_dir):
    """Plot the optimal spread (ask - bid) over time."""
    spread = parsed_df["ask"] - parsed_df["bid"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(parsed_df.index, spread, linewidth=0.8)
    ax.set_title("Dinâmica do Spread Ótimo", fontsize=14)
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Spread (Cents)")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "spread_dynamics.png"), dpi=300)
    plt.close(fig)
