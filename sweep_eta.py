"""Varredura da sensibilidade do preditor OBI (eta) sob choque direcional.

O termo do preditor desloca o preco de reserva em `eta * OBI`, grandeza expressa
em unidade de preco. Um valor nao calibrado contra o spread do venue deixa de ser
um vies direcional e passa a ser um reposicionamento violento da cotacao: com
eta = 0,5 USD e |OBI| tipico de 0,42, o deslocamento mediano alcanca 21 centimos
num mercado cujo spread e de 1 centimo.

Esta varredura isola o efeito de eta mantendo as demais camadas desligadas, de
modo que a contribuicao do preditor possa ser avaliada em escala apropriada.

Uso:
    python sweep_eta.py --etas 0.001 0.005 0.02 0.1 0.5 --seeds 4
"""

from __future__ import annotations

import argparse
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
    "eta", "seed", "pnl_usd", "sharpe", "max_drawdown_pct",
    "inv_max", "inv_final", "obi_abs_median", "displacement_cents",
)
KEY_FIELDS: Tuple[str, ...] = ("eta", "seed")


def run_cell(
    eta: float, seed: int, gamma: float, k: float, pov: float, kill_sigma2: float,
    r_bar: float,
) -> Optional[Dict[str, Any]]:
    """Executa uma simulacao com o preditor OBI ativo e eta informado.

    Parameters
    ----------
    eta : float
        Sensibilidade do preditor, em dolares.
    seed : int
        Semente estocastica.
    gamma, k : float
        Parametros de controlo otimo, oriundos da calibracao.
    pov : float
        Participacao de volume do agente institucional de execucao.
    kill_sigma2 : float
        Limiar da camada 3 (mantida desligada aqui, mas exigido pela config).

    Returns
    -------
    dict or None
        Metricas da simulacao, ou None em caso de falha.
    """
    tag = f"SWEEP_eta{eta}_rb{r_bar:.0e}_s{seed}"
    log_dir = os.path.join("log", tag)

    cmd = [
        sys.executable, "-u", "abides.py",
        "-c", "rmsc03_as", "-t", "ABM", "-d", "20240101",
        "-s", str(seed), "-l", tag,
        "-e", "-p", str(pov),
        "--gamma", str(gamma), "--k", str(k),
        "--eta-obi", str(eta),
        "--kill-sigma2", str(kill_sigma2),
        "--use-obi",
        "--mm-count", "2", "--as-colocate", "--lean-logs",
        "--r-bar", str(r_bar),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        df = parse_as_metrics(load_agent_log(log_dir), starting_cash=STARTING_CASH_CENTS)
        risk = calculate_risk_metrics(df)
        obi_med = float(df["obi"].abs().median())

        return {
            "eta": eta,
            "seed": seed,
            "pnl_usd": risk["final_pnl_usd"],
            "sharpe": risk["sharpe_per_obs"],
            "max_drawdown_pct": risk["max_drawdown_pct"],
            "inv_max": int(df["inv"].abs().max()),
            "inv_final": int(df["inv"].iloc[-1]),
            "obi_abs_median": obi_med,
            "displacement_cents": eta * obi_med * 100.0,
        }
    except subprocess.CalledProcessError as exc:
        print(f"  [ERRO ABIDES] {tag}: {str(exc.stderr)[-250:]}")
        return None
    except (FileNotFoundError, ValueError, KeyError, IndexError) as exc:
        print(f"  [ERRO PARSER] {tag}: {exc}")
        return None


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Consolida repeticoes em media e meia-largura do IC de 95%."""
    def mean_ci(key: str) -> Tuple[float, float]:
        vals = [float(r[key]) for r in rows if not math.isnan(float(r[key]))]
        if not vals:
            return float("nan"), float("nan")
        mean = statistics.fmean(vals)
        if len(vals) < 2:
            return mean, float("nan")
        return mean, 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))

    pnl_m, pnl_c = mean_ci("pnl_usd")
    sh_m, sh_c = mean_ci("sharpe")
    iv_m, iv_c = mean_ci("inv_max")
    disp_m, _ = mean_ci("displacement_cents")
    return {
        "n": float(len(rows)),
        "pnl_mean": pnl_m, "pnl_ci95": pnl_c,
        "sharpe_mean": sh_m, "sharpe_ci95": sh_c,
        "inv_max_mean": iv_m, "inv_max_ci95": iv_c,
        "displacement_cents_mean": disp_m,
    }


def main() -> None:
    """Executa a varredura de eta e consolida os resultados."""
    parser = argparse.ArgumentParser(description="Varredura de eta do preditor OBI")
    parser.add_argument("--etas", type=float, nargs="+",
                        default=[0.001, 0.005, 0.02, 0.1, 0.5])
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=20240101)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--k", type=float, default=20.0)
    parser.add_argument("--pov", type=float, default=0.25)
    parser.add_argument("--kill-sigma2", type=float, default=2.098067e-03)
    parser.add_argument("--r-bar", type=float, default=1e5)
    parser.add_argument("--out", type=str, default="eta_sweep.csv")
    parser.add_argument("--raw-out", type=str, default="eta_sweep_raw.csv")
    args = parser.parse_args()

    seeds = [args.base_seed + i for i in range(args.seeds)]
    tasks = [(e, s) for e in args.etas for s in seeds]

    ckpt = ResultCheckpoint(args.raw_out, RAW_FIELDS, KEY_FIELDS)
    pending = [t for t in tasks if not ckpt.is_done({"eta": t[0], "seed": t[1]})]

    print("=" * 72)
    print(" VARREDURA DE ETA (preditor OBI) sob choque direcional")
    print("=" * 72)
    print(f" etas         : {args.etas}")
    print(f" sementes     : {args.seeds} -> {len(tasks)} simulacoes")
    print(f" ja concluidas: {len(tasks) - len(pending)} | a executar: {len(pending)}")
    print(f" gamma/k      : {args.gamma}/{args.k} | POV {args.pov}")
    print("=" * 72)

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(run_cell, e, s, args.gamma, args.k, args.pov, args.kill_sigma2, args.r_bar): (e, s)
            for e, s in pending
        }
        done = 0
        for fut in as_completed(futs):
            done += 1
            row = fut.result()
            e, s = futs[fut]
            if row is None:
                print(f"[{done}/{len(pending)}] FALHA eta={e} seed={s}")
                continue
            ckpt.append(row)
            print(f"[{done}/{len(pending)}] OK eta={e:<7} seed={s} -> "
                  f"PnL ${row['pnl_usd']:>10,.0f} | desloc {row['displacement_cents']:>6.2f}c "
                  f"| InvMax {row['inv_max']:>6}")

    results = ckpt.load_all()
    if not results:
        print("\nNenhuma simulacao concluiu.")
        sys.exit(1)

    fields = ["eta", "n", "pnl_mean", "pnl_ci95", "sharpe_mean", "sharpe_ci95",
              "inv_max_mean", "inv_max_ci95", "displacement_cents_mean"]
    agg = []
    for e in args.etas:
        rows = [r for r in results if float(r["eta"]) == e]
        if rows:
            agg.append({"eta": e, **{f: aggregate(rows)[f] for f in fields[1:]}})
    write_aggregate(args.out, fields, agg)

    print(f"\nConcluido em {(time.time()-start)/60:.1f} min.")
    print(f"\n{'eta':>8} {'desloc':>9} {'PnL medio':>13} {'IC95':>9} {'InvMax':>9}")
    for r in agg:
        print(f"{r['eta']:>8} {r['displacement_cents_mean']:>8.2f}c "
              f"{r['pnl_mean']:>13,.0f} {r['pnl_ci95']:>9,.0f} {r['inv_max_mean']:>9,.0f}")


if __name__ == "__main__":
    main()
