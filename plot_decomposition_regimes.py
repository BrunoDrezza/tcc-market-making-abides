"""Decomposicao do spread realizado, comparada entre os dois regimes de preco.

Produz a figura que sustenta o argumento central da secao de resultados: a
captura de spread e ancorada no tick e portanto praticamente invariante ao nivel
de preco, enquanto a selecao adversa e ancorada na volatilidade e escala com ele.
O cruzamento entre as duas e o que separa um venue viavel de um inviavel.

Uso:
    python plot_decomposition_regimes.py --gamma 0.5 --seed 20240101
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plot_decomposition import decompose  # noqa: E402


def collect_regime(
    log_prefix: str, gamma: float, ks: Sequence[float], r_bar: float, seed: int,
    horizon_s: int = 30,
) -> pd.DataFrame:
    """Percorre as celulas de um regime e devolve a decomposicao de cada uma.

    Parameters
    ----------
    log_prefix : str
        Prefixo dos diretorios de log.
    gamma : float
        Coeficiente de aversao ao risco.
    ks : sequence of float
        Valores de k a percorrer.
    r_bar : float
        Nivel de preco do regime, usado para montar o nome do diretorio.
    seed : int
        Semente estocastica.
    horizon_s : int, optional
        Horizonte de medicao do deslocamento adverso, em segundos.

    Returns
    -------
    pd.DataFrame
        Colunas k, half_spread, capture, adverse, realized, n_fills, shares.
        Celulas cujo log nao exista sao omitidas.
    """
    rows: List[Dict[str, Any]] = []
    for k in ks:
        d = f"{log_prefix}_g{gamma}_k{k}_rb{r_bar:.0e}_s{seed}"
        if not os.path.isdir(d):
            continue
        # parse_as_metrics escreve progresso em stdout; descartado para manter
        # esta funcao silenciosa.
        with contextlib.redirect_stdout(io.StringIO()):
            r = decompose(d, horizon_s=horizon_s)
        r["k"] = k
        r["half_spread"] = (1.0 / gamma) * math.log1p(gamma / k) * 100.0
        rows.append(r)
    return pd.DataFrame(rows).sort_values("half_spread")


LABELS = {
    "en": {
        "capture": "Captured spread", "adverse": "Adverse selection",
        "realized": "Realised spread", "xlabel": "Quoted half-spread (ticks)",
        "ylabel": "Ticks per share",
        "suptitle": "Capture is anchored to the tick; adverse selection scales with price",
        "r1": "Asset at \\$1,000  (relative tick 0.1 bp)",
        "r2": "Asset at \\$100  (relative tick 1 bp)",
    },
    "pt": {
        "capture": "Spread capturado", "adverse": "Seleção adversa",
        "realized": "Spread realizado", "xlabel": "Meio-spread cotado (ticks)",
        "ylabel": "Cêntimos por ação",
        "suptitle": "A captura é ancorada no tick; a seleção adversa escala com o preço",
        "r1": "Ativo a USD 1.000  (tick relativo 0,1 pb)",
        "r2": "Ativo a USD 100  (tick relativo 1 pb)",
    },
}


def draw_panel(ax: plt.Axes, df: pd.DataFrame, title: str,
               lab: Dict[str, str], annotate: bool = False) -> None:
    """Desenha, num eixo, a decomposicao de um regime por distancia de cotacao.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eixo de destino.
    df : pd.DataFrame
        Saida de :func:`collect_regime`.
    title : str
        Titulo do painel.
    """
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df.capture, w, label=lab["capture"], color="#2E7D32")
    ax.bar(x + w / 2, df.adverse, w, label=lab["adverse"], color="#C62828")
    ax.plot(x, df.realized, "o-", color="black", lw=1.8, label=lab["realized"])
    # O eixo y compartilhado torna o painel de menor magnitude ilegivel; os
    # rotulos numericos preservam a leitura sem sacrificar o contraste visual.
    if annotate:
        for xi, yi in zip(x, df.realized):
            ax.annotate(f"{yi:+.2f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:.2f}\n(k={k:.0f})" for h, k in zip(df.half_spread, df.k)])
    ax.set_xlabel(lab["xlabel"])
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    """Gera a figura de decomposicao comparada entre regimes."""
    parser = argparse.ArgumentParser(description="Decomposicao comparada entre regimes")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--ks", type=float, nargs="+", default=[20.0, 50.0, 100.0, 200.0])
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--log-prefix", type=str, default="log/TCC_Opt")
    parser.add_argument("--lang", type=str, default="en", choices=("en", "pt"))
    parser.add_argument("--out", type=str,
                        default="../tcc-latex/figuras/decomposition_regimes.png")
    args = parser.parse_args()

    lab = LABELS[args.lang]
    regimes: List[Tuple[float, str]] = [(1e5, lab["r1"]), (1e4, lab["r2"])]

    frames: List[Tuple[pd.DataFrame, str]] = []
    for r_bar, label in regimes:
        df = collect_regime(args.log_prefix, args.gamma, args.ks, r_bar, args.seed,
                            args.horizon)
        if df.empty:
            raise SystemExit(f"Nenhum log encontrado para r_bar={r_bar:.0e}")
        frames.append((df, label))
        print(f"\n  {label}")
        for _, r in df.iterrows():
            print(f"    k={r.k:<6.0f} 1/2spread={r.half_spread:5.2f}  "
                  f"captura={r.capture:+.3f}  adversa={r.adverse:+.3f}  "
                  f"realizado={r.realized:+.3f}")

    plt.rcParams.update({"font.size": 11, "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for i, (ax, (df, label)) in enumerate(zip(axes, frames)):
        draw_panel(ax, df, label, lab, annotate=(i == 1))

    axes[0].set_ylabel(lab["ylabel"])
    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle(lab["suptitle"], fontsize=13, fontweight="bold")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  gravado: {args.out}")


if __name__ == "__main__":
    main()
