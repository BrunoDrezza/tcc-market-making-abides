"""Estudo de ablacao das tres camadas de defesa sob choque direcional.

Itera as 2^3 combinacoes de ativacao das camadas (OBI, Hedge, Kill-Switch) em
condicao Ceteris Paribus, sob a agressao de um POV Execution Agent. Cada
combinacao e repetida sob multiplas sementes estocasticas, de modo que as
diferencas relatadas venham acompanhadas de intervalo de confianca e nao sejam
confundidas com ruido amostral.

O limiar do Kill-Switch deve ser informado em unidade de preco ao quadrado e
calibrado sobre a distribuicao empirica de sigma^2 de um cenario de referencia
(ver `analysis.metrics.calculate_sigma2_stats`). Um limiar arbitrario nunca e
atingido, e as celulas com a camada ligada tornam-se duplicatas exatas das
celulas com ela desligada.
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
    "obi", "hedge", "kill", "seed", "pnl_usd", "return_pct", "sharpe",
    "max_drawdown_pct", "inv_max", "inv_final", "sigma2_max",
)
KEY_FIELDS: Tuple[str, ...] = ("obi", "hedge", "kill", "seed")


def run_cell(
    obi: bool, hedge: bool, kill: bool, seed: int, pov: float, kill_sigma2: float,
    lean_logs: bool, end_time: str, gamma: float, k: float,
    mm_count: int, backstop: float, colocate: bool, fund_vol: float, num_noise: int,
    eta_obi: float, r_bar: float,
) -> Optional[Dict[str, Any]]:
    """Executa um cenario da matriz de ablacao.

    Parameters
    ----------
    obi, hedge, kill : bool
        Ativacao das camadas 1, 2 e 3, respectivamente.
    seed : int
        Semente estocastica, propagada explicitamente via `-s`.
    pov : float
        Participacao de volume do agente institucional de execucao.
    kill_sigma2 : float
        Limiar de variancia da camada 3, em dolares ao quadrado.
    lean_logs : bool
        Suprime os logs pesados da bolsa, reduzindo o consumo de memoria.
    end_time : str
        Horario de encerramento do pregao simulado. Encurtar a sessao reduz
        fortemente o tempo de execucao, mas comprime a janela de atuacao do
        agente POV, que opera de mkt_open+30min ate mkt_close-30min.
    gamma : float
        Coeficiente de aversao ao risco, oriundo da calibracao por grid search.
    k : float
        Sensibilidade da taxa de execucao, oriunda da calibracao por grid search.
    mm_count : int
        Numero de market makers incumbentes. Deve coincidir com a ecologia sob a
        qual gamma e k foram calibrados, sob pena de aplicar parametros ajustados
        a um mercado num mercado de spread distinto.
    backstop : float
        Quantidade de backstop de cada incumbente.
    colocate : bool
        Iguala o perfil de latencia do agente A-S ao do primeiro incumbente.

    Returns
    -------
    dict or None
        Metricas do cenario, ou None se a execucao ou o parsing falharem.
    """
    # O nome do diretorio precisa codificar TODA variavel que distingue uma
    # execucao de outra, e nao apenas as camadas ablacionadas. Sem r_bar e
    # eta_obi no nome, uma rodada em ecologia ou ganho de preditor distinto
    # sobrescreve os logs da anterior sem qualquer aviso, e os artefatos deixam
    # de ser rastreaveis ate o experimento que os produziu.
    log_name = (
        f"TCC_Ablation_OBI_{obi}_HEDGE_{hedge}_KILL_{kill}"
        f"_rb{r_bar:.0e}_eta{eta_obi}_s{seed}"
    )
    log_dir = os.path.join("log", log_name)

    comando = [
        sys.executable, "-u", "abides.py",
        "-c", "rmsc03_as",
        "-t", "ABM",
        "-d", "20240101",
        "-s", str(seed),
        "-l", log_name,
        "-e", "-p", str(pov),
        "--kill-sigma2", str(kill_sigma2),
        "--end-time", end_time,
        "--gamma", str(gamma),
        "--k", str(k),
        "--mm-count", str(mm_count),
        "--mm-backstop-quantity", str(backstop),
        "--fund-vol", str(fund_vol),
        "--num-noise", str(num_noise),
        "--eta-obi", str(eta_obi),
        "--r-bar", str(r_bar),
    ]

    if colocate:
        comando.append("--as-colocate")

    if obi:
        comando.append("--use-obi")
    if hedge:
        comando.append("--use-hedge")
    if kill:
        comando.append("--use-kill-switch")
    if lean_logs:
        comando.append("--lean-logs")

    try:
        subprocess.run(comando, capture_output=True, text=True, check=True)
        df = parse_as_metrics(load_agent_log(log_dir), starting_cash=STARTING_CASH_CENTS)
        risk = calculate_risk_metrics(df)

        return {
            "obi": obi,
            "hedge": hedge,
            "kill": kill,
            "seed": seed,
            "pnl_usd": risk["final_pnl_usd"],
            "return_pct": risk["total_return_pct"],
            "sharpe": risk["sharpe_per_obs"],
            "max_drawdown_pct": risk["max_drawdown_pct"],
            "inv_max": int(df["inv"].abs().max()),
            "inv_final": int(df["inv"].iloc[-1]),
            "sigma2_max": float(df["sigma2"].max()),
        }

    except subprocess.CalledProcessError as exc:
        print(f"  [ERRO ABIDES] {log_name}: {str(exc.stderr)[-300:]}")
        return None
    except (FileNotFoundError, ValueError, KeyError, IndexError) as exc:
        print(f"  [ERRO PARSER] {log_name}: {exc}")
        return None


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Consolida repeticoes de um cenario em media e meia-largura do IC de 95%.

    Parameters
    ----------
    rows : sequence of dict
        Resultados individuais de uma mesma combinacao de camadas.

    Returns
    -------
    dict
        Medias e meias-larguras do intervalo de confianca de 95%.
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
    inv_max_mean, inv_max_ci = mean_ci("inv_max")
    inv_fin_mean, inv_fin_ci = mean_ci("inv_final")
    sharpe_mean, sharpe_ci = mean_ci("sharpe")
    dd_mean, _ = mean_ci("max_drawdown_pct")

    return {
        "n": float(len(rows)),
        "pnl_mean": pnl_mean, "pnl_ci95": pnl_ci,
        "sharpe_mean": sharpe_mean, "sharpe_ci95": sharpe_ci,
        "inv_max_mean": inv_max_mean, "inv_max_ci95": inv_max_ci,
        "inv_final_mean": inv_fin_mean, "inv_final_ci95": inv_fin_ci,
        "max_drawdown_pct_mean": dd_mean,
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
    """Executa a matriz de ablacao completa e consolida os resultados."""
    parser = argparse.ArgumentParser(description="Estudo de ablacao do agente A-S")
    parser.add_argument("--seeds", type=int, default=5, help="Repeticoes por cenario")
    parser.add_argument("--workers", type=int, default=3, help="Simulacoes simultaneas")
    parser.add_argument("--base-seed", type=int, default=20240101, help="Semente inicial")
    parser.add_argument("--pov", type=float, default=0.25, help="Participacao do POV Agent")
    parser.add_argument(
        "--kill-sigma2", type=float, required=True,
        help="Limiar de variancia da camada 3, em dolares^2. Calibrar sobre o "
             "percentil da distribuicao empirica de sigma^2 de um cenario de referencia.",
    )
    parser.add_argument("--out", type=str, default="ablation_results.csv")
    parser.add_argument("--raw-out", type=str, default="ablation_results_raw.csv")
    parser.add_argument(
        "--lean-logs", action="store_true",
        help="Suprime os logs pesados da bolsa. Recomendado: reduz o consumo de "
             "memoria por simulacao e viabiliza mais workers simultaneos.",
    )
    parser.add_argument(
        "--gamma", type=float, required=True,
        help="Aversao ao risco. Deve provir da celula vencedora do grid search, "
             "e nao de escolha heuristica.",
    )
    parser.add_argument(
        "--k", type=float, required=True,
        help="Sensibilidade da taxa de execucao, oriunda do grid search.",
    )
    parser.add_argument(
        "--end-time", type=str, default="11:30:00",
        help="Encerramento do pregao simulado. O default reproduz a sessao de 2h "
             "do RMSC03. Encurtar acelera muito a execucao, mas comprime a janela "
             "do agente POV (mkt_open+30min a mkt_close-30min): abaixo de 1h de "
             "sessao o estressor deixa de operar.",
    )
    parser.add_argument(
        "--mm-count", type=int, default=2,
        help="Market makers incumbentes. Deve coincidir com a ecologia usada na "
             "calibracao de gamma e k.",
    )
    parser.add_argument("--mm-backstop-quantity", type=float, default=50000.0)
    parser.add_argument(
        "--fund-vol", type=float, default=1e-8,
        help="Volatilidade do fundamental. Deve coincidir com a ecologia usada "
             "na calibracao de gamma e k.",
    )
    parser.add_argument("--num-noise", type=int, default=5000)
    parser.add_argument(
        "--r-bar", type=float, default=1e5,
        help="Nivel de preco do fundamental, em centimos. PRECISA coincidir com o "
             "valor usado na calibracao de gamma e k: o tick relativo determina "
             "o spread disputavel, e calibrar num nivel de preco para ablacionar "
             "noutro aplica parametros ajustados a um mercado num mercado distinto.",
    )
    parser.add_argument(
        "--eta-obi", type=float, default=0.5,
        help="Sensibilidade do preditor OBI, em dolares. Precisa ser calibrada "
             "contra o spread do venue: o deslocamento tipico de r_t vale "
             "eta*|OBI|, e valores da ordem de 0.5 deslocam a cotacao dezenas "
             "de vezes o spread de mercado, o que descaracteriza o preditor.",
    )
    parser.add_argument(
        "--as-colocate", action="store_true",
        help="Iguala a latencia do agente A-S a do primeiro incumbente.",
    )
    args = parser.parse_args()

    seeds: List[int] = [args.base_seed + i for i in range(args.seeds)]
    flags = (False, True)
    cells = list(itertools.product(flags, flags, flags))
    tasks = [(o, h, k, s) for (o, h, k) in cells for s in seeds]

    checkpoint = ResultCheckpoint(args.raw_out, RAW_FIELDS, KEY_FIELDS)
    pending = [
        t for t in tasks
        if not checkpoint.is_done({"obi": t[0], "hedge": t[1], "kill": t[2], "seed": t[3]})
    ]

    print("=" * 70)
    print(" Estudo de Ablacao sob choque direcional")
    print("=" * 70)
    print(f" cenarios    : {len(cells)}")
    print(f" sementes    : {args.seeds} por cenario -> {len(tasks)} simulacoes")
    print(f" ja concluidas: {len(tasks) - len(pending)} (retomada)")
    print(f" a executar  : {len(pending)}")
    print(f" POV         : {args.pov}")
    print(f" gamma / k   : {args.gamma} / {args.k} | eta_obi: {args.eta_obi}")
    print(f" ecologia    : {args.mm_count} MM incumbente(s), backstop "
          f"{args.mm_backstop_quantity:.0f}, r_bar={args.r_bar:.0e}"
          f"{', co-locado' if args.as_colocate else ''}")
    print(f" kill-sigma2 : {args.kill_sigma2:.6e} USD^2")
    print(f" workers     : {args.workers} | lean-logs: {args.lean_logs}")
    print("=" * 70)

    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_cell, o, h, k, s, args.pov, args.kill_sigma2, args.lean_logs,
                args.end_time, args.gamma, args.k,
                args.mm_count, args.mm_backstop_quantity, args.as_colocate,
                args.fund_vol, args.num_noise, args.eta_obi, args.r_bar,
            ): (o, h, k, s)
            for o, h, k, s in pending
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            row = future.result()
            o, h, k, s = futures[future]
            if row is None:
                print(f"[{done}/{len(pending)}] FALHA  OBI={o} HEDGE={h} KILL={k} seed={s}")
                continue
            checkpoint.append(row)
            print(
                f"[{done}/{len(pending)}] OK     OBI={str(o):<5} HEDGE={str(h):<5} "
                f"KILL={str(k):<5} seed={s} -> PnL ${row['pnl_usd']:>10.2f} | "
                f"InvMax {row['inv_max']:>5} | InvFim {row['inv_final']:>6}"
            )

    results = checkpoint.load_all()
    if not results:
        print("\nNenhuma simulacao concluiu. Abortando consolidacao.")
        sys.exit(1)

    fieldnames = [
        "test_id", "obi", "hedge", "kill", "n",
        "pnl_mean", "pnl_ci95", "sharpe_mean", "sharpe_ci95",
        "inv_max_mean", "inv_max_ci95", "inv_final_mean", "inv_final_ci95",
        "max_drawdown_pct_mean",
    ]
    aggregated: List[Dict[str, Any]] = []
    for idx, (obi, hedge, kill) in enumerate(cells, start=1):
        rows = [
            r for r in results
            if r["obi"] == obi and r["hedge"] == hedge and r["kill"] == kill
        ]
        if not rows:
            continue
        agg = aggregate(rows)
        aggregated.append({
            "test_id": idx, "obi": obi, "hedge": hedge, "kill": kill,
            **{f: agg[f] for f in fieldnames[4:]},
        })
    write_aggregate(args.out, fieldnames, aggregated)

    manifest_path = write_manifest(args.out, args, len(cells), len(tasks))

    elapsed = (time.time() - start) / 60.0
    print(f"\nConcluido em {elapsed:.1f} min. {len(results)}/{len(tasks)} simulacoes.")
    print(f"  agregado: {args.out}")
    print(f"  bruto   : {args.raw_out}")
    print(f"  manifesto: {manifest_path}")


if __name__ == "__main__":
    main()
