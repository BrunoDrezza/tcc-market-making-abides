"""Visualizacao da superficie de calibracao (gamma, k) com intervalos de confianca.

Le o CSV agregado produzido por `optimization_as.py`, que reporta media e
meia-largura do intervalo de confianca de 95% sobre multiplas sementes por
celula. A anotacao de cada celula exibe a media acompanhada da meia-largura,
de modo que a leitura visual nao sugira precisao que os dados nao sustentam.

Uso:
    python plot_optimization_results.py [--csv grid_recentered.csv]
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


def half_spread_cents(gamma: float, k: float) -> float:
    """Calcula o termo de liquidez do meio-spread otimo, em centimos.

    Parameters
    ----------
    gamma : float
        Coeficiente de aversao ao risco.
    k : float
        Sensibilidade da taxa de execucao.

    Returns
    -------
    float
        O valor de (1/gamma)*ln(1+gamma/k) convertido para centimos. Corresponde
        ao meio-spread praticado quando o termo de volatilidade e desprezivel.
    """
    return (1.0 / gamma) * math.log1p(gamma / k) * 100.0


def plot_metric(
    df: pd.DataFrame, value: str, err: Optional[str], title: str,
    cbar_label: str, outfile: str, fmt: str = ".1f", center: Optional[float] = 0.0,
) -> None:
    """Gera um mapa de calor de uma metrica sobre a grade (gamma, k).

    Parameters
    ----------
    df : pd.DataFrame
        Resultados agregados, com colunas `gamma`, `k` e a metrica.
    value : str
        Nome da coluna com a media da metrica.
    err : str or None
        Nome da coluna com a meia-largura do intervalo de confianca. Se
        informada, e anexada a anotacao de cada celula.
    title, cbar_label, outfile : str
        Titulo do grafico, rotulo da barra de cores e caminho de saida.
    fmt : str, optional
        Formato numerico das anotacoes.
    center : float or None, optional
        Valor central da paleta divergente.
    """
    pivot = df.pivot(index="gamma", columns="k", values=value)

    if err is not None and err in df.columns:
        pivot_err = df.pivot(index="gamma", columns="k", values=err)
        annot = pivot.copy().astype(object)
        for i in pivot.index:
            for j in pivot.columns:
                mean, half = pivot.loc[i, j], pivot_err.loc[i, j]
                annot.loc[i, j] = (
                    f"{mean:{fmt}}" if pd.isna(half) else f"{mean:{fmt}}\n±{half:{fmt}}"
                )
        annot_arg: object = annot.to_numpy()
        fmt_arg = ""
    else:
        annot_arg, fmt_arg = True, fmt

    plt.rcParams.update({"font.size": 11, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        pivot, annot=annot_arg, fmt=fmt_arg, cmap="RdYlGn", center=center,
        cbar_kws={"label": cbar_label}, linewidths=0.5, ax=ax,
        annot_kws={"fontsize": 9},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel(r"Aversão ao risco ($\gamma$)", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Sensibilidade de execução ($k$)", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()

    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  gravado: {outfile}")


def main() -> None:
    """Le o CSV agregado e emite os mapas de calor da superficie de calibracao."""
    parser = argparse.ArgumentParser(description="Superficie de calibracao (gamma, k)")
    parser.add_argument("--csv", type=str, default="grid_recentered.csv")
    parser.add_argument("--outdir", type=str, default="figuras")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV nao encontrado: {args.csv}")

    df = pd.read_csv(args.csv)
    required = {"gamma", "k", "pnl_mean"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Colunas ausentes em {args.csv}: {sorted(missing)}.\n"
            "Este script espera o formato agregado de optimization_as.py."
        )

    print(f"Lendo {args.csv}: {len(df)} celulas, "
          f"{df['n'].min():.0f}-{df['n'].max():.0f} sementes por celula")

    print("\n  meio-spread implicado por celula (termo de liquidez, em cents):")
    for _, r in df.sort_values(["gamma", "k"]).iterrows():
        print(f"    gamma={r.gamma:<6} k={r.k:<7} -> {half_spread_cents(r.gamma, r.k):5.2f}c")
    print()

    plot_metric(
        df, "pnl_mean", "pnl_ci95",
        r"Superfície de Calibração: PnL acumulado (média $\pm$ IC 95\%)",
        "PnL acumulado (USD)", os.path.join(args.outdir, "heatmap_pnl.png"),
    )
    if "sharpe_mean" in df.columns:
        plot_metric(
            df, "sharpe_mean", "sharpe_ci95",
            r"Superfície de Calibração: Índice de Sharpe (média $\pm$ IC 95\%)",
            "Sharpe por observação", os.path.join(args.outdir, "heatmap_sharpe.png"),
            fmt=".3f",
        )
    if "inv_max_mean" in df.columns:
        plot_metric(
            df, "inv_max_mean", "inv_max_ci95",
            r"Superfície de Calibração: exposição máxima de inventário",
            "Inventário máximo (ações)", os.path.join(args.outdir, "heatmap_inventory.png"),
            fmt=".0f", center=None,
        )

    best = df.loc[df["sharpe_mean"].idxmax()] if "sharpe_mean" in df else df.loc[df["pnl_mean"].idxmax()]
    print("\n  celula de melhor retorno ajustado ao risco:")
    print(f"    gamma={best.gamma}  k={best.k}  "
          f"Sharpe={best.get('sharpe_mean', float('nan')):.4f}  "
          f"PnL={best.pnl_mean:.2f} ± {best.get('pnl_ci95', float('nan')):.2f}")
    print("    ATENCAO: verificar se o IC separa esta celula das demais antes de "
          "declara-la vencedora.")


if __name__ == "__main__":
    main()
