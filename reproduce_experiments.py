"""Sequencia completa do experimento, da simulacao zero as figuras do artigo.

Este script e a especificacao executavel da metodologia. Ele nao acrescenta
nenhuma logica de simulacao: apenas encadeia, na ordem correta e com os
parametros corretos, os orquestradores que ja existem no repositorio, e explica
por que cada passo esta onde esta.

Por default **nao executa nada**: imprime a sequencia exata de comandos, com os
parametros resolvidos, para conferencia. Passe ``--execute`` para rodar de fato.

    # inspecao (default)
    python reproduce_experiments.py

    # execucao completa (varias horas)
    python reproduce_experiments.py --execute

O modo de inspecao nao importa NumPy nem Pandas e roda com qualquer Python 3;
apenas ``--execute`` exige o venv do ABIDES (``./ABIDES_Enviroment/bin/python``).

Dependencias entre os passos
----------------------------
Dois valores atravessam a sequencia e **precisam** ser medidos, nao arbitrados:

1. o limiar da Camada 3 (``--kill-sigma2``), que sai do percentil 99 da
   distribuicao empirica de sigma^2 da rodada de referencia;
2. o par ``(gamma, k)`` da ablacao, que sai da celula de melhor Sharpe medio do
   grid search.

Em modo de inspecao esses valores aparecem como marcadores
(``<KILL_SIGMA2>``, ``<GAMMA*>``, ``<K*>``), justamente porque so existem depois
de rodar os passos que os produzem.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

#: Minutos de relogio por simulacao de 2 h de pregao, medido no hardware de
#: referencia (4 nucleos, 3 workers simultaneos). Ver README/REPRODUCIBILITY.
MINUTES_PER_SIM: float = 12.0

#: Nivel de fundamental assumido por `run_ablation_study.py`. Ver `_check_rbar`.
ABLATION_RBAR_CENTS: float = 1.0e5

PLACEHOLDER_KILL: str = "<KILL_SIGMA2>"
PLACEHOLDER_GAMMA: str = "<GAMMA*>"
PLACEHOLDER_K: str = "<K*>"


class Step(NamedTuple):
    """Um passo da sequencia experimental.

    Attributes
    ----------
    number : int
        Posicao na sequencia.
    title : str
        Nome curto do passo.
    rationale : str
        Por que o passo existe e por que esta nesta posicao. E a parte que o
        leitor do artigo precisa; os comandos ele consegue reconstruir sozinho.
    commands : tuple of list of str
        Comandos a executar, na ordem. Vazio para passos puramente de leitura.
    outputs : tuple of str
        Artefactos produzidos, para conferencia manual.
    n_sims : int
        Numero de simulacoes ABIDES disparadas, usado na estimativa de tempo.
    warning : str, optional
        Armadilha conhecida do passo.
    """

    number: int
    title: str
    rationale: str
    commands: Tuple[List[str], ...]
    outputs: Tuple[str, ...]
    n_sims: int
    warning: Optional[str] = None


# --- Construcao dos passos --------------------------------------------------


def _py() -> str:
    """Devolve o interpretador a usar nos subprocessos.

    Returns
    -------
    str
        ``sys.executable``. Os orquestradores do repositorio ja propagam o
        interpretador ativo para os subprocessos; rodar fora do venv faz todos
        os cenarios falharem silenciosamente para ``Status=CRASHED`` no CSV.
    """
    return sys.executable


def step_reference(args: argparse.Namespace) -> Step:
    """Passo 1: rodada de referencia em calmaria, com log completo."""
    command = [
        _py(), "-u", "abides.py",
        "-c", "rmsc03_as",
        "-t", args.ticker,
        "-d", args.date,
        "-s", str(args.base_seed),
        "-l", args.ref_log,
        "--end-time", args.end_time,
        "--gamma", f"{args.gammas[0]:g}",
        "--k", f"{args.ks[0]:g}",
        "--mm-count", str(args.mm_count),
        "--num-noise", str(args.num_noise),
        "--fund-vol", f"{args.fund_vol:g}",
        "--r-bar", f"{args.r_bar:g}",
    ]
    if args.as_colocate:
        command.append("--as-colocate")

    return Step(
        number=1,
        title="Rodada de referencia em calmaria (log completo)",
        rationale=(
            "Sem estressor: nao ha `-e/-p`. Esta rodada nao mede desempenho, mede o\n"
            "  MERCADO. Dela saem dois insumos que nenhum outro passo produz:\n"
            "   (a) a distribuicao empirica de sigma^2, base do limiar da Camada 3;\n"
            "   (b) a serie do livro de ofertas, unico insumo do diagnostico de\n"
            "       posicionamento (analysis/positioning.py).\n"
            "  Roda deliberadamente SEM --lean-logs: a flag zera `book_freq` no config\n"
            "  e o ORDERBOOK_<sym>_FREQ_1S.bz2 deixa de ser gravado. E o unico passo da\n"
            "  sequencia que paga o custo do log pesado, e paga por causa de (b).\n"
            "  A ecologia (--mm-count, --num-noise, --fund-vol, --r-bar, --as-colocate)\n"
            "  tem de ser IDENTICA a dos passos 3 e 5: um limiar de sigma^2 calibrado\n"
            "  num mercado e aplicado noutro nao e calibracao, e coincidencia."
        ),
        commands=(command,),
        outputs=(
            f"log/{args.ref_log}/AVELLANEDA_STOIKOV_AGENT.bz2",
            f"log/{args.ref_log}/ORDERBOOK_{args.ticker}_FREQ_1S.bz2",
        ),
        n_sims=1,
    )


def step_diagnose(args: argparse.Namespace) -> Step:
    """Passo 2: extrai o percentil 99 de sigma^2 e o diagnostico de posicionamento."""
    log_dir = os.path.join("log", args.ref_log)
    return Step(
        number=2,
        title="Diagnostico da referencia -> limiar do kill-switch",
        rationale=(
            "O percentil 99 de sigma^2 vira o `--kill-sigma2` do passo 5. Precisa vir\n"
            "  de MEDICAO e nao de constante porque o limiar tem de ser raro mas\n"
            "  alcancavel: acima da cauda da distribuicao a camada nunca dispara e as\n"
            "  quatro celulas com KILL=True viram duplicatas exatas das celulas com\n"
            "  KILL=False — a ablacao mede oito cenarios e reporta quatro. Abaixo da\n"
            "  mediana ela dispara o tempo todo e o agente simplesmente para de cotar.\n"
            "  A escala de sigma^2 muda com r_bar, com fund_vol e com a densidade de\n"
            "  agentes; nao existe valor universal. `run_ablation_study.py` torna o\n"
            "  argumento obrigatorio exatamente para proibir a constante arbitraria.\n"
            "  O segundo comando responde a pergunta que o PnL nao responde: onde o\n"
            "  agente ficou no livro. E o insumo do achado central do trabalho."
        ),
        commands=(
            [_py(), "run_analysis.py", "--log_dir", log_dir],
            [_py(), "-m", "analysis.positioning", "--log-dir", log_dir],
        ),
        outputs=(
            "sigma2_p99 (impresso; consumido pelo passo 5)",
            "analysis_output/*.png",
        ),
        n_sims=0,
    )


def step_grid(args: argparse.Namespace) -> Step:
    """Passo 3: grid search de (gamma, k) sob calmaria."""
    command = [
        _py(), "-u", "optimization_as.py",
        "--gammas", *[f"{g:g}" for g in args.gammas],
        "--ks", *[f"{k:g}" for k in args.ks],
        "--seeds", str(args.seeds),
        "--workers", str(args.workers),
        "--base-seed", str(args.base_seed),
        "--end-time", args.end_time,
        "--mm-count", str(args.mm_count),
        "--num-noise", str(args.num_noise),
        "--fund-vol", f"{args.fund_vol:g}",
        "--r-bar", f"{args.r_bar:g}",
        "--out", args.grid_csv,
        "--raw-out", args.grid_raw_csv,
        "--lean-logs",
    ]
    if args.as_colocate:
        command.append("--as-colocate")

    n_sims = len(args.gammas) * len(args.ks) * args.seeds
    return Step(
        number=3,
        title="Grid search de (gamma, k) — SEM o agente institucional",
        rationale=(
            "O grid roda em calmaria, sem `-e -p`, de proposito. Calibrar sob choque\n"
            "  confundiria duas coisas distintas: o efeito proprio dos parametros de\n"
            "  controlo otimo e a reacao ao estressor. Se (gamma, k) fossem escolhidos\n"
            "  ja sob o POV, a ablacao do passo 5 nao teria o que atribuir as camadas —\n"
            "  parte da resposta ao choque ja estaria embutida na parametrizacao.\n"
            "  Aqui --lean-logs E usado: sao "
            f"{n_sims} simulacoes, e manter o log da bolsa e a\n"
            "  serie do livro em "
            f"{args.workers} processos simultaneos estoura a memoria. O pipeline\n"
            "  de risco-retorno le apenas o log do agente, que --lean-logs preserva.\n"
            "  Cada celula e repetida em "
            f"{args.seeds} sementes com `-s` explicito: sem isso o\n"
            "  config deriva a semente do timestamp e cada celula roda sobre uma\n"
            "  realizacao diferente do oraculo, o que compara ruido com ruido."
        ),
        commands=(command,),
        outputs=(args.grid_csv, args.grid_raw_csv, "log/TCC_Opt_*/"),
        n_sims=n_sims,
    )


def step_select(args: argparse.Namespace) -> Step:
    """Passo 4: selecao da celula vencedora por Sharpe medio."""
    return Step(
        number=4,
        title="Selecao da celula vencedora por Sharpe medio",
        rationale=(
            "A selecao e por Sharpe MEDIO entre sementes, e nao por PnL pontual: PnL\n"
            "  de uma unica realizacao premia a celula que teve sorte na trajetoria do\n"
            "  fundamental. O CSV agregado tras `sharpe_ci95` ao lado de `sharpe_mean`;\n"
            "  se o intervalo da segunda melhor celula cobre a media da primeira, os\n"
            "  dados nao separam as duas e a escolha entre elas e convencao, nao\n"
            "  resultado. O script reporta essa sobreposicao quando ela ocorre."
        ),
        commands=(
            [_py(), "-c", (
                "import pandas as pd; "
                f"d=pd.read_csv({args.grid_csv!r}); "
                "print(d.sort_values('sharpe_mean', ascending=False).head())"
            )],
        ),
        outputs=("(gamma*, k*) para o passo 5",),
        n_sims=0,
    )


def step_ablation(
    args: argparse.Namespace, gamma: str, k: str, kill_sigma2: str
) -> Step:
    """Passo 5: ablacao 2^3 das camadas de defesa sob choque direcional."""
    command = [
        _py(), "-u", "run_ablation_study.py",
        "--gamma", gamma,
        "--k", k,
        "--kill-sigma2", kill_sigma2,
        "--pov", f"{args.pov:g}",
        "--eta-obi", f"{args.eta_obi:g}",
        "--seeds", str(args.seeds),
        "--workers", str(args.workers),
        "--base-seed", str(args.base_seed),
        "--end-time", args.end_time,
        "--mm-count", str(args.mm_count),
        "--num-noise", str(args.num_noise),
        "--fund-vol", f"{args.fund_vol:g}",
        "--out", args.ablation_csv,
        "--raw-out", args.ablation_raw_csv,
        "--lean-logs",
    ]
    if args.as_colocate:
        command.append("--as-colocate")

    n_sims = 8 * args.seeds
    return Step(
        number=5,
        title="Ablacao 2^3 das camadas — COM o agente institucional",
        rationale=(
            "Aqui o POV entra (`--pov`), e o inverso do passo 3. As tres camadas sao\n"
            "  mecanismos de defesa: so existem para responder a um choque direcional\n"
            "  persistente. Rodar a ablacao em calmaria mediria oito variantes de um\n"
            "  agente que nunca precisou se defender, e as diferencas seriam ruido.\n"
            "  gamma e k vem do passo 4 — nao sao reotimizados aqui, porque o objeto de\n"
            "  estudo passa a ser a camada, com a politica de cotacao mantida fixa.\n"
            "  --kill-sigma2 vem do passo 2 pelo motivo ja exposto.\n"
            "  --eta-obi precisa ser calibrado contra o spread do venue: o deslocamento\n"
            "  tipico de r_t vale eta*|OBI|, de modo que eta=0.5 desloca a cotacao\n"
            "  dezenas de vezes o spread de mercado e descaracteriza o preditor."
        ),
        commands=(command,),
        outputs=(args.ablation_csv, args.ablation_raw_csv, "log/TCC_Ablation_*/"),
        n_sims=n_sims,
        warning=(
            "run_ablation_study.py NAO expoe --r-bar: as simulacoes da ablacao rodam "
            f"sempre no default do config ({ABLATION_RBAR_CENTS:.0e} centimos). "
            "Com --r-bar diferente disso no passo 3, gamma e k seriam calibrados "
            "num tick relativo e aplicados noutro."
        ),
    )


def step_figures(args: argparse.Namespace, gamma: str) -> Step:
    """Passo 6: figuras e tabelas do artigo."""
    commands = (
        [_py(), "plot_optimization_results.py",
         "--csv", args.grid_csv, "--outdir", args.figdir],
        [_py(), "plot_decomposition.py",
         "--gamma", gamma,
         "--ks", *[f"{k:g}" for k in args.ks],
         "--seed", str(args.base_seed),
         "--log-prefix", "log/TCC_Opt",
         "--out", os.path.join(args.figdir, "decomposition.png")],
        [_py(), "make_latex_table.py", "grid",
         "--csv", args.grid_csv,
         "--out", os.path.join(args.tabdir, "table_grid.tex")],
        [_py(), "make_latex_table.py", "ablation",
         "--csv", args.ablation_csv,
         "--out", os.path.join(args.tabdir, "table_ablation.tex")],
    )
    rb_infix = f"rb{args.r_bar:.0e}"
    return Step(
        number=6,
        title="Figuras e tabelas",
        rationale=(
            "A superficie de calibracao e a decomposicao por distancia de cotacao sao\n"
            "  as duas figuras que sustentam a tese: a primeira mostra que nenhuma\n"
            "  celula do grid e lucrativa, a segunda mostra por que — aproximar-se do\n"
            "  topo reduz a perda por acao mas amplia o volume exposto a selecao\n"
            "  adversa, e o spread realizado permanece negativo em toda a faixa\n"
            "  praticavel. As tabelas LaTeX sempre acompanham as medias das respectivas\n"
            "  meias-larguras de IC: reportar a media isolada convidaria o leitor a\n"
            "  discriminar celulas que os dados nao separam."
        ),
        commands=commands,
        outputs=(
            os.path.join(args.figdir, "*.png"),
            os.path.join(args.tabdir, "table_grid.tex"),
            os.path.join(args.tabdir, "table_ablation.tex"),
        ),
        n_sims=0,
        warning=(
            "plot_decomposition.py monta o caminho como "
            "`<log-prefix>_g<gamma>_k<k>_s<seed>`, sem o infixo de r_bar que "
            "optimization_as.py insere (`..._k<k>_" + rb_infix + "_s<seed>`). "
            "Enquanto os dois nao forem reconciliados, crie ligacoes simbolicas "
            "antes de rodar este comando, por exemplo:\n"
            "      for d in log/TCC_Opt_g*_" + rb_infix + "_s*; do "
            "ln -sfn \"$(basename $d)\" \"${d/_" + rb_infix + "/}\"; done"
        ),
    )


# --- Execucao e leitura de resultados ---------------------------------------


def measure_kill_sigma2(log_dir: str, quantile_key: str = "sigma2_p99") -> float:
    """Le o percentil de sigma^2 da rodada de referencia.

    Parameters
    ----------
    log_dir : str
        Diretorio de log da rodada de referencia.
    quantile_key : str, optional
        Chave devolvida por :func:`analysis.metrics.calculate_sigma2_stats`.

    Returns
    -------
    float
        Limiar em dolares ao quadrado, pronto para `--kill-sigma2`.

    Notes
    -----
    Importa Pandas apenas aqui: o modo de inspecao do script precisa rodar sem
    o venv do ABIDES, e um import de topo o impediria.
    """
    from analysis.metrics import calculate_sigma2_stats
    from analysis.parser import load_agent_log, parse_as_metrics

    stats = calculate_sigma2_stats(parse_as_metrics(load_agent_log(log_dir)))
    return float(stats[quantile_key])


def select_best_cell(grid_csv: str) -> Tuple[float, float, Dict[str, Any]]:
    """Escolhe a celula de maior Sharpe medio no CSV agregado do grid.

    Parameters
    ----------
    grid_csv : str
        Caminho do agregado produzido por `optimization_as.py`.

    Returns
    -------
    tuple
        ``(gamma, k, linha)`` da celula vencedora.

    Raises
    ------
    FileNotFoundError
        Se o CSV nao existir — tipicamente porque o passo 3 nao rodou.
    ValueError
        Se nenhuma linha tiver Sharpe finito.

    Notes
    -----
    Usa o modulo `csv` da biblioteca padrao, e nao Pandas, para que a selecao
    seja auditavel a olho e o modo de inspecao permaneca livre de dependencias.
    """
    if not os.path.exists(grid_csv):
        raise FileNotFoundError(
            f"Agregado do grid nao encontrado: {grid_csv!r}. Rode o passo 3."
        )

    with open(grid_csv, newline="") as handle:
        rows = [row for row in csv.DictReader(handle)]

    scored: List[Tuple[float, Dict[str, str]]] = []
    for row in rows:
        try:
            sharpe = float(row["sharpe_mean"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Linha sem 'sharpe_mean' utilizavel em {grid_csv!r}") from exc
        if math.isnan(sharpe):
            continue
        scored.append((sharpe, row))

    if not scored:
        raise ValueError(f"Nenhuma celula com Sharpe finito em {grid_csv!r}.")

    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_sharpe, best = scored[0]

    # Sobreposicao de IC: se a segunda melhor celula cobre a media da primeira,
    # a ordenacao entre as duas nao e sustentada pelos dados.
    if len(scored) > 1:
        runner_sharpe, runner = scored[1]
        try:
            runner_half = float(runner.get("sharpe_ci95", "nan"))
        except (TypeError, ValueError):
            runner_half = float("nan")
        if not math.isnan(runner_half) and runner_sharpe + runner_half >= best_sharpe:
            print(
                f"  [ATENCAO] O IC da celula (gamma={runner['gamma']}, k={runner['k']}) "
                f"cobre a media da vencedora. A ordenacao entre as duas nao e "
                f"separada pelos dados."
            )

    return float(best["gamma"]), float(best["k"]), dict(best)


def render(command: Sequence[str]) -> str:
    """Formata um comando para exibicao, encurtando o caminho do interpretador.

    Parameters
    ----------
    command : sequence of str
        Argumentos do subprocesso.

    Returns
    -------
    str
        Linha de comando legivel, com aspas apenas onde necessarias.
    """
    parts: List[str] = []
    for i, token in enumerate(command):
        display = os.path.basename(token) if i == 0 else token
        parts.append(f'"{display}"' if (" " in display or "\n" in display) else display)
    return " ".join(parts)


def show(step: Step, execute: bool) -> None:
    """Imprime o cabecalho, a justificativa e os comandos de um passo."""
    print()
    print("=" * 78)
    print(f" PASSO {step.number} — {step.title}")
    print("=" * 78)
    print(f"  PORQUE: {step.rationale}")
    if step.n_sims:
        minutes = math.ceil(step.n_sims / max(1, _WORKERS[0])) * MINUTES_PER_SIM
        print(
            f"\n  CUSTO : {step.n_sims} simulacao(oes) ~ {minutes / 60.0:.1f} h de "
            f"relogio a {MINUTES_PER_SIM:.0f} min/simulacao"
        )
    if step.warning:
        print(f"\n  ATENCAO: {step.warning}")
    print("\n  COMANDO(S):")
    for command in step.commands:
        print(f"    $ {render(command)}")
    if step.outputs:
        print("\n  PRODUZ:")
        for out in step.outputs:
            print(f"    - {out}")
    if not execute:
        print("\n  [dry-run] nada foi executado.")


def run(step: Step) -> None:
    """Executa os comandos de um passo, abortando na primeira falha.

    Parameters
    ----------
    step : Step
        Passo a executar.

    Raises
    ------
    SystemExit
        Se qualquer comando retornar codigo diferente de zero. Prosseguir apos
        uma falha produziria CSVs parciais indistinguiveis dos completos.
    """
    for command in step.commands:
        print(f"\n  >>> {render(command)}", flush=True)
        result = subprocess.run(command)
        if result.returncode != 0:
            raise SystemExit(
                f"Passo {step.number} falhou (codigo {result.returncode}): "
                f"{render(command)}"
            )


# Compartilhado com `show` apenas para a estimativa de tempo; evita ter de
# arrastar o Namespace inteiro ate a funcao de impressao.
_WORKERS: List[int] = [3]


def _check_rbar(args: argparse.Namespace) -> None:
    """Recusa combinacoes em que r_bar nao se propaga ate a ablacao.

    Parameters
    ----------
    args : argparse.Namespace
        Argumentos ja parseados.

    Raises
    ------
    SystemExit
        Em modo `--execute`, quando `--r-bar` difere do default do config e o
        usuario nao assumiu explicitamente o risco.

    Notes
    -----
    `optimization_as.py` propaga `--r-bar` para o config; `run_ablation_study.py`
    nao. Calibrar o grid num nivel de preco e ablacionar noutro troca o tick
    relativo entre os dois passos, que e justamente a grandeza que determina se
    existe premio de liquidez disputavel.
    """
    if math.isclose(args.r_bar, ABLATION_RBAR_CENTS, rel_tol=1e-9):
        return

    message = (
        f"--r-bar={args.r_bar:g} difere do default do config "
        f"({ABLATION_RBAR_CENTS:.0e}). run_ablation_study.py nao expoe --r-bar, "
        "de modo que o passo 5 rodaria num tick relativo diferente do passo 3."
    )
    print(f"\n[ATENCAO] {message}\n")
    if args.execute and not args.allow_rbar_mismatch:
        raise SystemExit(
            "Abortado. Use --allow-rbar-mismatch para assumir a inconsistencia, "
            "ou mantenha --r-bar no default."
        )


def build_parser() -> argparse.ArgumentParser:
    """Constroi o parser da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Sequencia reproduzivel do experimento A-S sobre ABIDES",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Executa de fato. Sem esta flag o script apenas imprime a sequencia.",
    )
    parser.add_argument(
        "--only", type=int, nargs="+", default=None, metavar="N",
        help="Restringe a execucao aos passos indicados (ex.: --only 3 4).",
    )

    ecology = parser.add_argument_group("ecologia da simulacao")
    ecology.add_argument(
        "--r-bar", type=float, default=1e5,
        help="Nivel do fundamental em centimos. Determina o tick relativo.",
    )
    ecology.add_argument(
        "--mm-count", type=int, default=2,
        help="Market makers incumbentes. Fixa o spread praticado pelo mercado.",
    )
    ecology.add_argument("--num-noise", type=int, default=5000)
    ecology.add_argument("--fund-vol", type=float, default=1e-8)
    ecology.add_argument(
        "--as-colocate", action="store_true", default=True,
        help="Iguala a latencia do agente A-S a do primeiro incumbente.",
    )
    ecology.add_argument(
        "--no-as-colocate", dest="as_colocate", action="store_false",
        help="Desliga a co-locacao. Confunde estrategia com geografia.",
    )
    ecology.add_argument("--ticker", type=str, default="ABM")
    ecology.add_argument("--date", type=str, default="20240101")
    ecology.add_argument("--end-time", type=str, default="11:30:00")

    design = parser.add_argument_group("desenho experimental")
    design.add_argument("--seeds", type=int, default=4, help="Sementes por celula.")
    design.add_argument("--workers", type=int, default=3, help="Simulacoes simultaneas.")
    design.add_argument("--base-seed", type=int, default=20240101)
    design.add_argument(
        "--gammas", type=float, nargs="+", default=[0.5, 1.0, 5.0],
        help="Valores de gamma do grid search.",
    )
    design.add_argument(
        "--ks", type=float, nargs="+", default=[20.0, 50.0, 100.0, 200.0],
        help="Valores de k do grid search. k < 20 coloca o agente fora da "
             "distribuicao executavel deste venue.",
    )
    design.add_argument(
        "--pov", type=float, default=0.25,
        help="Participacao de volume do agente institucional, apenas na ablacao.",
    )
    design.add_argument(
        "--eta-obi", type=float, default=0.1,
        help="Sensibilidade do preditor OBI, em dolares.",
    )

    paths = parser.add_argument_group("artefactos")
    paths.add_argument("--ref-log", type=str, default="CALIB_REF")
    paths.add_argument("--grid-csv", type=str, default="grid_recentered.csv")
    paths.add_argument("--grid-raw-csv", type=str, default="grid_recentered_raw.csv")
    paths.add_argument("--ablation-csv", type=str, default="ablation_results.csv")
    paths.add_argument("--ablation-raw-csv", type=str, default="ablation_results_raw.csv")
    paths.add_argument("--figdir", type=str, default="figuras")
    paths.add_argument("--tabdir", type=str, default="tabelas")
    paths.add_argument(
        "--allow-rbar-mismatch", action="store_true",
        help="Permite --execute com r_bar que a ablacao nao consegue reproduzir.",
    )
    return parser


def main() -> None:
    """Imprime (e opcionalmente executa) a sequencia completa do experimento."""
    args = build_parser().parse_args()
    _WORKERS[0] = args.workers
    wanted = set(args.only) if args.only else None

    def enabled(number: int) -> bool:
        return wanted is None or number in wanted

    print()
    print("#" * 78)
    print(" REPRODUCAO DO EXPERIMENTO — market making Avellaneda-Stoikov sobre ABIDES")
    print("#" * 78)
    print(f"  modo        : {'EXECUCAO' if args.execute else 'DRY-RUN (default)'}")
    print(f"  interpretador: {sys.executable}")
    print(f"  ecologia    : r_bar={args.r_bar:g}c, {args.mm_count} MM, "
          f"{args.num_noise} ruido, fund_vol={args.fund_vol:g}"
          f"{', co-locado' if args.as_colocate else ''}")
    print(f"  grid        : {len(args.gammas)}x{len(args.ks)} celulas x "
          f"{args.seeds} sementes | {args.workers} workers")
    print(f"  estressor   : POV {args.pov:g} (apenas no passo 5)")
    if not args.execute:
        print("\n  Nada sera executado. Use --execute para rodar de verdade.")

    _check_rbar(args)

    total_sims = 1 + len(args.gammas) * len(args.ks) * args.seeds + 8 * args.seeds
    total_hours = math.ceil(total_sims / max(1, args.workers)) * MINUTES_PER_SIM / 60.0
    print(f"\n  CUSTO TOTAL : {total_sims} simulacoes ~ {total_hours:.1f} h de relogio.")

    # --- Passo 1
    step1 = step_reference(args)
    if enabled(1):
        show(step1, args.execute)
        if args.execute:
            run(step1)

    # --- Passo 2
    step2 = step_diagnose(args)
    kill_sigma2 = PLACEHOLDER_KILL
    if enabled(2):
        show(step2, args.execute)
        if args.execute:
            run(step2)
    if args.execute and enabled(5):
        # Repetido em processo para capturar o valor: o passo 2 imprime, mas o
        # passo 5 precisa do numero, nao do texto. So e resolvido quando o passo
        # que o consome esta habilitado, para que `--only 3` nao exija a rodada
        # de referencia.
        measured = measure_kill_sigma2(os.path.join("log", args.ref_log))
        kill_sigma2 = f"{measured:.6e}"
        print(f"\n  >>> sigma2_p99 medido = {kill_sigma2} USD^2 -> --kill-sigma2")

    # --- Passo 3
    step3 = step_grid(args)
    if enabled(3):
        show(step3, args.execute)
        if args.execute:
            run(step3)

    # --- Passo 4
    step4 = step_select(args)
    gamma_star, k_star = PLACEHOLDER_GAMMA, PLACEHOLDER_K
    if enabled(4):
        show(step4, args.execute)
        if args.execute:
            run(step4)
    if args.execute and (enabled(5) or enabled(6)):
        gamma_value, k_value, row = select_best_cell(args.grid_csv)
        gamma_star, k_star = f"{gamma_value:g}", f"{k_value:g}"
        print(
            f"\n  >>> celula vencedora: gamma={gamma_star} k={k_star} "
            f"(sharpe_mean={row.get('sharpe_mean')}, pnl_mean={row.get('pnl_mean')})"
        )

    # --- Passo 5
    step5 = step_ablation(args, gamma_star, k_star, kill_sigma2)
    if enabled(5):
        show(step5, args.execute)
        if args.execute:
            run(step5)

    # --- Passo 6
    step6 = step_figures(args, gamma_star)
    if enabled(6):
        show(step6, args.execute)
        if args.execute:
            os.makedirs(args.figdir, exist_ok=True)
            os.makedirs(args.tabdir, exist_ok=True)
            run(step6)

    print()
    print("#" * 78)
    print(" FIM" if args.execute else " FIM (dry-run — nenhum comando foi executado)")
    print("#" * 78)
    print()


if __name__ == "__main__":
    main()
