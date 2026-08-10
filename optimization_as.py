"""Varredura espacial (Grid Search) dos parametros de controlo otimo do agente A-S.

Cada combinacao (gamma, k) e executada sob multiplas sementes estocasticas
independentes. Isto e indispensavel: na ausencia do argumento `-s`, a
configuracao `rmsc03_as` deriva a semente do marcador temporal de execucao, de
modo que cada celula da grade rodaria sobre uma realizacao distinta do processo
gerador de precos e as diferencas observadas confundiriam efeito parametrico com
ruido amostral.

A selecao final e feita por retorno ajustado ao risco medio com intervalo de
confianca, e nao por PnL pontual.
"""

from __future__ import annotations

import argparse
import csv
import json
import itertools
import math
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analysis.checkpoint import ResultCheckpoint, write_aggregate
from analysis.metrics import calculate_risk_metrics
from analysis.parser import load_agent_log, parse_as_metrics

STARTING_CASH_CENTS: int = 10_000_000

RAW_FIELDS: Tuple[str, ...] = (
    "gamma", "k", "seed", "pnl_usd", "return_pct", "sharpe", "max_drawdown_pct",
    "inv_max", "inv_final", "spread_mean", "spread_std",
)
KEY_FIELDS: Tuple[str, ...] = ("gamma", "k", "seed")

# Grade de calibracao, espacada de forma aproximadamente logaritmica.
#
# A faixa de gamma foi ancorada na variancia empiricamente observada no cenario
# de referencia (sigma^2 mediano de 2,9e-4 USD^2). O deslocamento de inventario
# q*gamma*sigma^2*tau, avaliado num inventario tipico de 100 acoes e tau=0,5,
# percorre nesta faixa desde valor inferior ao tick ate valor superior ao proprio
# spread cotado:
#
#   gamma=0,5  -> 0,7 cent   (praticamente inerte)
#   gamma=1,0  -> 1,5 cent   (da ordem de um tick)
#   gamma=5,0  -> 7,3 cents  (fracao relevante do spread de ~20 cents)
#   gamma=25,0 -> 36 cents   (excede o spread; controlo de inventario agressivo)
#
# Faixas centradas em valores muito menores nao exercitariam o termo de
# inventario, que e justamente o mecanismo distintivo do modelo.
GAMMAS: Tuple[float, ...] = (0.5, 1.0, 5.0, 25.0)
KS: Tuple[float, ...] = (5.0, 10.0, 50.0)


def run_cell(
    gamma: float, k: float, seed: int, log_root: str, lean_logs: bool, end_time: str,
    mm_count: int, backstop: float, colocate: bool, fund_vol: float, num_noise: int,
    r_bar: float,
) -> Optional[Dict[str, Any]]:
    """Executa uma simulacao e devolve as metricas de risco-retorno.

    Parameters
    ----------
    gamma : float
        Coeficiente de aversao ao risco absoluto.
    k : float
        Sensibilidade da taxa de execucao a distancia do mid-price.
    seed : int
        Semente estocastica, propagada explicitamente via `-s`.
    log_root : str
        Prefixo do diretorio de logs.
    lean_logs : bool
        Suprime os logs pesados da bolsa, reduzindo o consumo de memoria.
    end_time : str
        Horario de encerramento do pregao simulado.
    mm_count : int
        Numero de market makers incumbentes. Determina o spread praticado pelo
        mercado e, por consequencia, a faixa de k que faz sentido calibrar.
    backstop : float
        Quantidade de backstop de cada incumbente.
    colocate : bool
        Iguala o perfil de latencia do agente A-S ao do primeiro incumbente.
    fund_vol : float
        Volatilidade do processo fundamental. E a alavanca que efetivamente
        determina o spread de equilibrio: a densidade de agentes nao altera o
        topo do livro, que permanece fixado no tick.
    num_noise : int
        Numero de agentes de ruido.

    Returns
    -------
    dict or None
        Metricas da simulacao, ou None se a execucao ou o parsing falharem.
    """
    log_name = f"{log_root}_g{gamma}_k{k}_rb{r_bar:.0e}_s{seed}"
    log_dir = os.path.join("log", log_name)

    comando = [
        sys.executable, "-u", "abides.py",
        "-c", "rmsc03_as",
        "-t", "ABM",
        "-d", "20240101",
        "-s", str(seed),
        "-l", log_name,
        "--gamma", str(gamma),
        "--k", str(k),
        "--end-time", end_time,
        "--mm-count", str(mm_count),
        "--mm-backstop-quantity", str(backstop),
        "--fund-vol", str(fund_vol),
        "--num-noise", str(num_noise),
        "--r-bar", str(r_bar),
    ]

    if lean_logs:
        comando.append("--lean-logs")
    if colocate:
        comando.append("--as-colocate")

    try:
        subprocess.run(comando, capture_output=True, text=True, check=True)
        df = parse_as_metrics(load_agent_log(log_dir), starting_cash=STARTING_CASH_CENTS)
        risk = calculate_risk_metrics(df)
        spread = df["ask"] - df["bid"]

        return {
            "gamma": gamma,
            "k": k,
            "seed": seed,
            "pnl_usd": risk["final_pnl_usd"],
            "return_pct": risk["total_return_pct"],
            "sharpe": risk["sharpe_per_obs"],
            "max_drawdown_pct": risk["max_drawdown_pct"],
            "inv_max": int(df["inv"].abs().max()),
            "inv_final": int(df["inv"].iloc[-1]),
            "spread_mean": float(spread.mean()),
            "spread_std": float(spread.std(ddof=1)),
        }

    except subprocess.CalledProcessError as exc:
        print(f"  [ERRO ABIDES] gamma={gamma} k={k} seed={seed}: {str(exc.stderr)[-300:]}")
        return None
    except (FileNotFoundError, ValueError, KeyError, IndexError) as exc:
        print(f"  [ERRO PARSER] gamma={gamma} k={k} seed={seed}: {exc}")
        return None


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Consolida repeticoes de uma celula em media e meia-largura do IC de 95%.

    Parameters
    ----------
    rows : sequence of dict
        Resultados individuais de uma mesma combinacao (gamma, k).

    Returns
    -------
    dict
        Medias e meias-larguras do intervalo de confianca de 95% para PnL,
        Sharpe e inventario maximo, alem do numero de repeticoes validas.

    Notes
    -----
    A meia-largura usa a aproximacao normal (1,96 * erro padrao). Com poucas
    repeticoes trata-se de estimativa apenas indicativa; o numero de sementes
    deve ser suficiente para que a comparacao entre celulas seja informativa.
    """
    def mean_ci(key: str) -> Tuple[float, float]:
        vals = [float(r[key]) for r in rows if not math.isnan(float(r[key]))]
        if not vals:
            return float("nan"), float("nan")
        mean = statistics.fmean(vals)
        if len(vals) < 2:
            return mean, float("nan")
        half = 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))
        return mean, half

    pnl_mean, pnl_ci = mean_ci("pnl_usd")
    sharpe_mean, sharpe_ci = mean_ci("sharpe")
    inv_mean, inv_ci = mean_ci("inv_max")
    dd_mean, _ = mean_ci("max_drawdown_pct")
    spread_mean, _ = mean_ci("spread_mean")
    spread_std_mean, _ = mean_ci("spread_std")

    return {
        "n": float(len(rows)),
        "pnl_mean": pnl_mean, "pnl_ci95": pnl_ci,
        "sharpe_mean": sharpe_mean, "sharpe_ci95": sharpe_ci,
        "inv_max_mean": inv_mean, "inv_max_ci95": inv_ci,
        "max_drawdown_pct_mean": dd_mean,
        "spread_mean": spread_mean, "spread_std_mean": spread_std_mean,
    }


def write_manifest(out_csv: str, args: argparse.Namespace, n_cells: int, n_sims: int) -> str:
    """Grava, ao lado do CSV de resultados, o conjunto completo de parametros da execucao.

    O nome do diretorio de log codifica apenas as variaveis que costumam variar
    entre execucoes. Parametros como `mm_count`, `fund_vol` e `num_noise` ficariam
    de fora, e um artefato que nao registra a configuracao que o produziu nao e
    rastreavel. O manifesto fecha essa lacuna sem inchar os nomes de diretorio.

    Parameters
    ----------
    out_csv : str
        Caminho do CSV agregado; o manifesto recebe o mesmo nome com sufixo
        `.manifest.json`.
    args : argparse.Namespace
        Argumentos efetivamente usados na execucao.
    n_cells, n_sims : int
        Numero de celulas experimentais e de simulacoes previstas.

    Returns
    -------
    str
        Caminho do manifesto gravado.
    """
    path = os.path.splitext(out_csv)[0] + ".manifest.json"
    payload = {
        "parametros": {k: v for k, v in sorted(vars(args).items())},
        "n_celulas": n_cells,
        "n_simulacoes": n_sims,
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    return path


def main() -> None:
    """Executa a grade completa e consolida os resultados em CSV."""
    parser = argparse.ArgumentParser(description="Grid Search do agente Avellaneda-Stoikov")
    parser.add_argument("--seeds", type=int, default=5, help="Repeticoes por celula")
    parser.add_argument("--workers", type=int, default=3, help="Simulacoes simultaneas")
    parser.add_argument("--base-seed", type=int, default=20240101, help="Semente inicial")
    parser.add_argument("--out", type=str, default="optimization_results.csv")
    parser.add_argument("--raw-out", type=str, default="optimization_results_raw.csv")
    parser.add_argument(
        "--lean-logs", action="store_true",
        help="Suprime os logs pesados da bolsa. Recomendado: reduz o consumo de "
             "memoria por simulacao e viabiliza mais workers simultaneos.",
    )
    parser.add_argument(
        "--end-time", type=str, default="11:30:00",
        help="Encerramento do pregao simulado. O default reproduz a sessao de 2h "
             "do RMSC03.",
    )
    parser.add_argument(
        "--mm-count", type=int, default=2,
        help="Market makers incumbentes. O default 2 reproduz o RMSC03 canonico, "
             "no qual o spread fica fixado em um tick e nao ha premio disputavel.",
    )
    parser.add_argument("--mm-backstop-quantity", type=float, default=50000.0)
    parser.add_argument(
        "--fund-vol", type=float, default=1e-8,
        help="Volatilidade do fundamental. O default do RMSC03 (1e-8) produz spread "
             "fixado em um tick, sem premio disputavel; 1e-6 abre o spread para "
             "cerca de dez ticks.",
    )
    parser.add_argument("--num-noise", type=int, default=5000)
    parser.add_argument(
        "--r-bar", type=float, default=1e5,
        help="Nivel de preco do fundamental, em centimos. Determina o tick relativo.",
    )
    parser.add_argument(
        "--as-colocate", action="store_true",
        help="Iguala a latencia do agente A-S a do primeiro incumbente.",
    )
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=list(GAMMAS),
        help="Valores de gamma da grade.",
    )
    parser.add_argument(
        "--ks", type=float, nargs="+", default=list(KS),
        help="Valores de k da grade. Devem ser calibrados contra o spread que a "
             "ecologia escolhida efetivamente produz (ver probe_ecology.py).",
    )
    args = parser.parse_args()

    seeds: List[int] = [args.base_seed + i for i in range(args.seeds)]
    cells = list(itertools.product(args.gammas, args.ks))
    tasks = [(g, k, s) for (g, k) in cells for s in seeds]

    checkpoint = ResultCheckpoint(args.raw_out, RAW_FIELDS, KEY_FIELDS)
    pending = [
        t for t in tasks
        if not checkpoint.is_done({"gamma": t[0], "k": t[1], "seed": t[2]})
    ]

    print("=" * 70)
    print(" Grid Search do agente Avellaneda-Stoikov")
    print("=" * 70)
    print(f" celulas      : {len(cells)} ({len(args.gammas)} gamma x {len(args.ks)} k)")
    print(f"   gamma      : {args.gammas}")
    print(f"   k          : {args.ks}")
    print(f" sementes     : {args.seeds} por celula -> {len(tasks)} simulacoes")
    print(f" ja concluidas: {len(tasks) - len(pending)} (retomada)")
    print(f" a executar   : {len(pending)}")
    print(f" ecologia     : {args.mm_count} MM incumbente(s), {args.num_noise} ruido, "
          f"fund_vol={args.fund_vol:.0e}{', co-locado' if args.as_colocate else ''}")
    print(f" workers      : {args.workers} | lean-logs: {args.lean_logs}")
    print("=" * 70)

    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_cell, g, k, s, "TCC_Opt", args.lean_logs, args.end_time,
                args.mm_count, args.mm_backstop_quantity, args.as_colocate,
                args.fund_vol, args.num_noise, args.r_bar,
            ): (g, k, s)
            for g, k, s in pending
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            row = future.result()
            g, k, s = futures[future]
            if row is None:
                print(f"[{done}/{len(pending)}] FALHA  gamma={g} k={k} seed={s}")
                continue
            checkpoint.append(row)
            print(
                f"[{done}/{len(pending)}] OK     gamma={g:<6} k={k:<5} seed={s} -> "
                f"PnL ${row['pnl_usd']:>10.2f} | Sharpe {row['sharpe']:>8.4f} | "
                f"InvMax {row['inv_max']:>5}"
            )

    results = checkpoint.load_all()
    if not results:
        print("\nNenhuma simulacao concluiu. Abortando consolidacao.")
        sys.exit(1)

    fieldnames = [
        "gamma", "k", "n", "pnl_mean", "pnl_ci95", "sharpe_mean", "sharpe_ci95",
        "inv_max_mean", "inv_max_ci95", "max_drawdown_pct_mean",
        "spread_mean", "spread_std_mean",
    ]
    aggregated: List[Dict[str, Any]] = []
    for gamma, k in sorted(cells):
        rows = [r for r in results if r["gamma"] == gamma and r["k"] == k]
        if not rows:
            continue
        agg = aggregate(rows)
        aggregated.append({"gamma": gamma, "k": k, **{f: agg[f] for f in fieldnames[2:]}})
    write_aggregate(args.out, fieldnames, aggregated)

    manifest_path = write_manifest(args.out, args, len(cells), len(tasks))

    elapsed = (time.time() - start) / 60.0
    print(f"\nConcluido em {elapsed:.1f} min. {len(results)}/{len(tasks)} simulacoes.")
    print(f"  agregado: {args.out}")
    print(f"  bruto   : {args.raw_out}")
    print(f"  manifesto: {manifest_path}")


if __name__ == "__main__":
    main()
