"""Converte os CSVs agregados de resultados em tabelas LaTeX prontas para inclusao.

Emite `booktabs` sem regras verticais, conforme exigido pela classe acmart, e
sempre acompanha as medias das respectivas meias-larguras de intervalo de
confianca. Reportar a media isolada convidaria o leitor a discriminar celulas
que os dados nao separam.

Uso:
    python make_latex_table.py ablation --csv ablation_results.csv
    python make_latex_table.py grid --csv grid_recentered.csv
"""

from __future__ import annotations

import argparse
import math
import os
from typing import List

import pandas as pd


def fmt(mean: float, half: float, prec: int = 2) -> str:
    """Formata media e meia-largura do IC como celula LaTeX.

    Parameters
    ----------
    mean : float
        Valor medio.
    half : float
        Meia-largura do intervalo de confianca de 95%.
    prec : int, optional
        Casas decimais.

    Returns
    -------
    str
        Celula formatada; omite o termo de incerteza quando indisponivel.
    """
    if pd.isna(mean):
        return "--"
    if pd.isna(half):
        return f"${mean:,.{prec}f}$"
    return f"${mean:,.{prec}f} \\pm {half:,.{prec}f}$"


def ablation_table(df: pd.DataFrame) -> str:
    """Monta a tabela do estudo de ablacao.

    Parameters
    ----------
    df : pd.DataFrame
        Agregado produzido por `run_ablation_study.py`.

    Returns
    -------
    str
        Corpo LaTeX completo do ambiente `table*`.
    """
    yn = {True: r"\checkmark", False: "--"}
    lines: List[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ablation of the three defensive layers under a persistent "
        r"directional shock. Values are means over independent seeds with 95\% "
        r"confidence half-widths. Peak and terminal inventory are reported "
        r"together, since a layer may reduce one while increasing the other.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{@{}cccc rrrr@{}}",
        r"\toprule",
        r"\# & OBI & Hedge & Kill & PnL (USD) & Sharpe & Peak inv. & Final inv. \\",
        r"\midrule",
    ]
    for _, r in df.sort_values("test_id").iterrows():
        lines.append(
            f"{int(r.test_id)} & {yn[bool(r.obi)]} & {yn[bool(r.hedge)]} & "
            f"{yn[bool(r.kill)]} & {fmt(r.pnl_mean, r.pnl_ci95)} & "
            f"{fmt(r.sharpe_mean, r.sharpe_ci95, 4)} & "
            f"{fmt(r.inv_max_mean, r.inv_max_ci95, 0)} & "
            f"{fmt(r.inv_final_mean, r.inv_final_ci95, 0)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def grid_table(df: pd.DataFrame) -> str:
    """Monta a tabela da superficie de calibracao.

    Parameters
    ----------
    df : pd.DataFrame
        Agregado produzido por `optimization_as.py`.

    Returns
    -------
    str
        Corpo LaTeX completo do ambiente `table*`.
    """
    lines: List[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Calibration surface over $(\gamma, k)$ under directional calm. "
        r"Means over independent seeds with 95\% confidence half-widths. The "
        r"half-spread column is the liquidity term $\gamma^{-1}\ln(1+\gamma/k)$, "
        r"in ticks; the market's own half-spread is $0.5$.}",
        r"\label{tab:grid}",
        r"\begin{tabular}{@{}rrr rrr@{}}",
        r"\toprule",
        r"$\gamma$ & $k$ & $\delta^\ast$ & PnL (USD) & Sharpe & Peak inv. \\",
        r"\midrule",
    ]
    for _, r in df.sort_values(["gamma", "k"]).iterrows():
        half = (1.0 / r.gamma) * math.log1p(r.gamma / r.k) * 100.0
        # Sem casas decimais no PnL: a meia-largura do IC e da ordem de centenas
        # de dolares, de sorte que reportar centimos sugeriria precisao inexistente.
        lines.append(
            f"{r.gamma:g} & {r.k:g} & {half:.2f} & {fmt(r.pnl_mean, r.pnl_ci95, 0)} & "
            f"{fmt(r.sharpe_mean, r.sharpe_ci95, 3)} & "
            f"{fmt(r.inv_max_mean, r.inv_max_ci95, 0)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main() -> None:
    """Le o CSV indicado e grava a tabela LaTeX correspondente."""
    parser = argparse.ArgumentParser(description="Gerador de tabelas LaTeX")
    parser.add_argument("kind", choices=("ablation", "grid"))
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV nao encontrado: {args.csv}")

    df = pd.read_csv(args.csv)
    body = ablation_table(df) if args.kind == "ablation" else grid_table(df)

    out = args.out or f"../tcc-latex/icaif/table_{args.kind}.tex"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as handle:
        handle.write(body + "\n")
    print(f"Tabela gravada em {out} ({len(df)} linhas)")
    print("\n" + body)


if __name__ == "__main__":
    main()
