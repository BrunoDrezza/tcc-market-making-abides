"""Sondagem empirica do spread de mercado sob diferentes ecologias de liquidez.

O meio-spread otimo do modelo de Avellaneda-Stoikov, delta* = (1/gamma)ln(1+gamma/k),
so faz sentido calibrado contra o spread que o mercado efetivamente pratica. Antes
de varrer k, portanto, e preciso medir o spread que cada configuracao de liquidez
incumbente produz.

Este script executa sessoes curtas sob ecologias candidatas, mede o spread do topo
do livro e a distribuicao das distancias dos negocios ao mid, e deriva a faixa de k
que posicionaria o agente A-S no topo do livro.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def run_probe(eco: Dict[str, Any], seed: int, end_time: str) -> Optional[Dict[str, Any]]:
    """Executa uma sessao de sondagem e mede a microestrutura resultante.

    Parameters
    ----------
    eco : dict
        Especificacao da ecologia, com as chaves `mm_count`, `backstop`,
        `num_noise` e `fund_vol`.
    seed : int
        Semente estocastica.
    end_time : str
        Horario de encerramento do pregao simulado.

    Returns
    -------
    dict or None
        Estatisticas de spread e distancia dos negocios, ou None em caso de falha.
    """
    tag = (
        f"PROBE_mm{eco['mm_count']}_bs{int(eco['backstop'])}"
        f"_n{eco['num_noise']}_fv{eco['fund_vol']:.0e}_rb{eco['r_bar']:.0e}_s{seed}"
    )
    log_dir = os.path.join("log", tag)

    cmd = [
        sys.executable, "-u", "abides.py",
        "-c", "rmsc03_as", "-t", "ABM", "-d", "20240101",
        "-s", str(seed), "-l", tag,
        "--end-time", end_time,
        "--mm-count", str(eco["mm_count"]),
        "--mm-backstop-quantity", str(eco["backstop"]),
        "--num-noise", str(eco["num_noise"]),
        "--fund-vol", str(eco["fund_vol"]),
        "--r-bar", str(eco["r_bar"]),
        "--as-colocate",
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"  [ERRO] {tag}: {str(exc.stderr)[-300:]}")
        return None

    try:
        ob_path = os.path.join(log_dir, "ORDERBOOK_ABM_FREQ_1S.bz2")
        ob = pd.read_pickle(ob_path)
        prices = np.array(ob.columns, dtype=float)
        vol = np.nan_to_num(ob.to_numpy(dtype=float))

        # Convencao verificada empiricamente: volume negativo = bid, positivo = ask.
        bid_m, ask_m = vol < 0, vol > 0
        best_bid = np.where(
            bid_m.any(1), np.nanmax(np.where(bid_m, prices, np.nan), axis=1), np.nan
        )
        best_ask = np.where(
            ask_m.any(1), np.nanmin(np.where(ask_m, prices, np.nan), axis=1), np.nan
        )
        spread = pd.Series(best_ask - best_bid).dropna()
        spread = spread[spread > 0]

        exch = pd.read_pickle(os.path.join(log_dir, "EXCHANGE_AGENT.bz2"))
        trades = exch[exch.EventType == "ORDER_EXECUTED"].copy()
        trades["px"] = trades.Event.apply(lambda d: d.get("fill_price"))
        trades = trades.dropna(subset=["px"]).sort_index()

        mid = pd.Series((best_bid + best_ask) / 2.0, index=ob.index).dropna()
        mid_at = mid.reindex(trades.index, method="ffill")
        dist = (trades.px - mid_at).abs().dropna()

        return {
            "mm_count": eco["mm_count"],
            "backstop": eco["backstop"],
            "num_noise": eco["num_noise"],
            "fund_vol": eco["fund_vol"],
            "r_bar": eco["r_bar"],
            "seed": seed,
            "spread_median": float(spread.median()),
            "spread_p25": float(spread.quantile(0.25)),
            "spread_p75": float(spread.quantile(0.75)),
            "trade_dist_median": float(dist.median()),
            "trade_dist_p90": float(dist.quantile(0.90)),
            "trade_dist_p99": float(dist.quantile(0.99)),
            "n_trades": int(len(dist)),
        }
    except (FileNotFoundError, ValueError, KeyError, IndexError) as exc:
        print(f"  [ERRO PARSER] {tag}: {exc}")
        return None


def k_for_half_spread(half_spread_cents: float, gamma: float) -> float:
    """Inverte delta* = (1/gamma)ln(1+gamma/k) para obter k.

    Parameters
    ----------
    half_spread_cents : float
        Meio-spread desejado, em centimos.
    gamma : float
        Coeficiente de aversao ao risco.

    Returns
    -------
    float
        Valor de k que produz o meio-spread desejado, ou infinito se inatingivel.
    """
    delta = half_spread_cents / 100.0
    denom = math.exp(gamma * delta) - 1.0
    if denom <= 0:
        return float("inf")
    return gamma / denom


def main() -> None:
    """Executa a sondagem sobre as ecologias candidatas e reporta a faixa de k."""
    parser = argparse.ArgumentParser(description="Sondagem de ecologia de liquidez")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--end-time", type=str, default="10:00:00")
    parser.add_argument("--base-seed", type=int, default=20240101)
    parser.add_argument(
        "--preset", type=str, default="mm", choices=("mm", "liquidity", "rbar"),
        help="'mm' varia os market makers incumbentes; 'liquidity' varia a "
             "densidade de fluxo nao informado e a volatilidade do fundamental.",
    )
    args = parser.parse_args()

    base = {"mm_count": 2, "backstop": 50000.0, "num_noise": 5000, "fund_vol": 1e-8, "r_bar": 1e5}

    if args.preset == "mm":
        # Rarefacao progressiva da provisao de liquidez incumbente.
        ecologies = [
            {**base},
            {**base, "mm_count": 1},
            {**base, "mm_count": 1, "backstop": 5000.0},
            {**base, "mm_count": 0},
        ]
    elif args.preset == "rbar":
        # Nivel de preco: unica alavanca que altera o TICK RELATIVO.
        ecologies = [
            {**base, "r_bar": 1e5},   # canonico, USD 1.000  -> tick = 0,1 bp
            {**base, "r_bar": 3e4},   # USD 300              -> tick = 0,33 bp
            {**base, "r_bar": 1e4},   # USD 100              -> tick = 1 bp
            {**base, "r_bar": 3e3},   # USD 30               -> tick = 3,3 bp
        ]
    else:
        # Rarefacao do fluxo nao informado e elevacao da volatilidade. O spread de
        # equilibrio de um formador de mercado cresce com a volatilidade e cai com
        # a densidade de contrapartes; se o tick e o limitante, apenas estas
        # alavancas podem abrir premio disputavel.
        ecologies = [
            {**base, "mm_count": 1, "num_noise": 1000},
            {**base, "mm_count": 1, "num_noise": 200},
            {**base, "mm_count": 1, "fund_vol": 1e-6},
            {**base, "mm_count": 1, "fund_vol": 1e-4},
            {**base, "mm_count": 1, "num_noise": 1000, "fund_vol": 1e-6},
            {**base, "mm_count": 0, "num_noise": 1000, "fund_vol": 1e-6},
        ]

    seeds = [args.base_seed + i for i in range(args.seeds)]
    tasks = [(e, s) for e in ecologies for s in seeds]

    print("=" * 78)
    print(f" SONDAGEM DE ECOLOGIA ({args.preset}): qual spread cada configuracao produz")
    print("=" * 78)
    print(f" ecologias: {len(ecologies)} | sementes: {args.seeds} | sessao ate {args.end_time}")
    print("=" * 78)

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_probe, e, s, args.end_time): (e, s) for e, s in tasks}
        for fut in as_completed(futs):
            row = fut.result()
            e, s = futs[fut]
            desc = (f"mm={e['mm_count']} bs={int(e['backstop']):>5} "
                    f"noise={e['num_noise']:>4} fv={e['fund_vol']:.0e} rb={e['r_bar']:.0e}")
            if row is None:
                print(f"  FALHA  {desc} seed={s}")
                continue
            results.append(row)
            print(
                f"  OK  {desc} seed={s} -> spread={row['spread_median']:5.1f}c  "
                f"dist_p50={row['trade_dist_median']:4.1f}c  n={row['n_trades']:,}"
            )

    if not results:
        print("\nNenhuma sondagem concluiu.")
        sys.exit(1)

    df = pd.DataFrame(results)
    out = f"probe_ecology_{args.preset}.csv"
    df.to_csv(out, index=False)

    keys = ["mm_count", "backstop", "num_noise", "fund_vol", "r_bar"]
    agg = df.groupby(keys).mean(numeric_only=True).reset_index()

    print()
    print("=" * 78)
    print(" RESUMO POR ECOLOGIA (media entre sementes)")
    print("=" * 78)
    print(f"  {'MMs':>4} {'backstop':>9} {'noise':>6} {'fund_vol':>9} "
          f"{'spread':>8} {'p99 dist':>9} {'negocios':>10}")
    for _, r in agg.iterrows():
        print(
            f"  {int(r.mm_count):>4} {r.backstop:>9.0f} {int(r.num_noise):>6} "
            f"{r.fund_vol:>9.0e} {r.spread_median:>7.1f}c {r.trade_dist_p99:>8.1f}c "
            f"{r.n_trades:>10,.0f}"
        )

    print()
    print("=" * 78)
    print(" FAIXA DE k IMPLICADA  (meio-spread alvo = metade do spread de mercado)")
    print("=" * 78)
    for _, r in agg.iterrows():
        target = r.spread_median / 2.0
        ks = [k_for_half_spread(target, g) for g in (0.5, 1.0, 5.0)]
        print(
            f"  mm={int(r.mm_count)} noise={int(r.num_noise):>4} fv={r.fund_vol:.0e}: "
            f"alvo={target:.2f}c -> k~{ks[0]:.0f} (g=0.5), {ks[1]:.0f} (g=1), {ks[2]:.0f} (g=5)"
        )
    print("=" * 78)
    print(f" Resultados gravados em {out}")


if __name__ == "__main__":
    main()
