"""Decomposicao do resultado por distancia de cotacao.

Para cada execucao do agente, separa o resultado em duas parcelas:

- captura de spread: quanto se ganha por ser passivo, isto e, a distancia entre
  o preco de execucao e o mid vigente no instante da execucao;
- selecao adversa: quanto o mid se desloca contra a posicao adquirida ao longo
  de um horizonte curto posterior.

A soma das duas e o spread realizado. Um formador de mercado so e viavel onde o
spread realizado e positivo; se ele permanece negativo em toda a faixa de
distancias praticaveis, nao existe parametrizacao lucrativa e a inviabilidade e
do mercado, nao da politica.

Uso:
    python plot_decomposition.py --gamma 1.0 --seed 20240101
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.parser import load_agent_log, parse_as_metrics  # noqa: E402


def decompose(log_dir: str, horizon_s: int = 30) -> Dict[str, Any]:
    """Decompoe o resultado das execucoes de uma simulacao.

    Parameters
    ----------
    log_dir : str
        Diretorio de log da simulacao.
    horizon_s : int, optional
        Horizonte, em segundos, sobre o qual se mede o deslocamento adverso.

    Returns
    -------
    dict
        Numero de execucoes, volume, e as medias por acao de captura, selecao
        adversa e spread realizado, em centimos.
    """
    raw = pd.read_pickle(os.path.join(log_dir, "AVELLANEDA_STOIKOV_AGENT.bz2"))
    ex = raw[raw.EventType == "ORDER_EXECUTED"].copy()
    if ex.empty:
        return {"n_fills": 0, "shares": 0, "capture": np.nan,
                "adverse": np.nan, "realized": np.nan}

    ex["qty"] = ex.Event.apply(lambda d: d["quantity"])
    ex["buy"] = ex.Event.apply(lambda d: d["is_buy_order"])
    ex["px"] = ex.Event.apply(lambda d: d["fill_price"])

    mid = parse_as_metrics(load_agent_log(log_dir))["mid"].sort_index()
    idx = mid.index

    def mid_at(t: pd.Timestamp, offset: int) -> float:
        i = idx.searchsorted(t + pd.Timedelta(seconds=offset))
        return float(mid.iloc[min(i, len(mid) - 1)])

    cap: List[float] = []
    adv: List[float] = []
    for t, r in ex.iterrows():
        m0, m1 = mid_at(t, 0), mid_at(t, horizon_s)
        sign = 1.0 if r.buy else -1.0
        cap.append((m0 - r.px) * sign)
        adv.append((m1 - m0) * sign)

    cap_a, adv_a = np.array(cap), np.array(adv)
    return {
        "n_fills": int(len(ex)),
        "shares": int(ex.qty.sum()),
        "capture": float(cap_a.mean()),
        "adverse": float(adv_a.mean()),
        "realized": float((cap_a + adv_a).mean()),
    }


def main() -> None:
    """Percorre as celulas de um gamma fixo e emite a figura de decomposicao."""
    parser = argparse.ArgumentParser(description="Decomposicao por distancia de cotacao")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--ks", type=float, nargs="+", default=[20.0, 50.0, 100.0, 200.0])
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--log-prefix", type=str, default="log/TCC_Opt")
    parser.add_argument(
        "--r-bar", type=float, default=None,
        help="Nivel de preco usado no grid. optimization_as.py grava o infixo "
             "_rb<r_bar> no nome do diretorio; informe o mesmo valor para que "
             "os logs sejam localizados.",
    )
    parser.add_argument("--out", type=str, default="../tcc-latex/figuras/decomposition.png")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for k in args.ks:
        rb = "" if args.r_bar is None else f"_rb{args.r_bar:.0e}"
        d = f"{args.log_prefix}_g{args.gamma}_k{k}{rb}_s{args.seed}"
        if not os.path.isdir(d):
            print(f"  log ausente, ignorado: {d}")
            continue
        r = decompose(d)
        r["k"] = k
        r["half_spread"] = (1.0 / args.gamma) * math.log1p(args.gamma / k) * 100.0
        rows.append(r)
        print(f"  k={k:<6.0f} 1/2spread={r['half_spread']:5.2f}c  fills={r['n_fills']:>6} "
              f"captura={r['capture']:+.2f}c  adversa={r['adverse']:+.2f}c  "
              f"realizado={r['realized']:+.2f}c")

    if not rows:
        raise SystemExit("Nenhum log encontrado. Rode o grid antes.")

    df = pd.DataFrame(rows).sort_values("half_spread")
    x = np.arange(len(df))

    plt.rcParams.update({"font.size": 11, "font.family": "serif"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    w = 0.35
    ax1.bar(x - w / 2, df.capture, w, label="Captura de spread", color="#2E7D32")
    ax1.bar(x + w / 2, df.adverse, w, label="Seleção adversa", color="#C62828")
    ax1.plot(x, df.realized, "o-", color="black", lw=1.6, label="Spread realizado")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{h:.2f}c\n(k={k:.0f})" for h, k in zip(df.half_spread, df.k)])
    ax1.set_xlabel("Meio-spread cotado")
    ax1.set_ylabel("Cêntimos por ação")
    ax1.set_title("Decomposição por ação")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, df.shares, 0.6, color="#1565C0")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{h:.2f}c" for h in df.half_spread])
    ax2.set_xlabel("Meio-spread cotado")
    ax2.set_ylabel("Volume executado (ações)")
    ax2.set_title("Volume executado")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Aproximar-se do topo reduz a perda por ação mas amplia o volume exposto",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  gravado: {args.out}")


if __name__ == "__main__":
    main()
