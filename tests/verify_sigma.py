#!/usr/bin/env python3
"""Harness autocontido que demonstra o defeito de unidades em ``_estimate_sigma2``.

Contexto
--------
O modelo de Avellaneda-Stoikov (2008) pressupoe movimento Browniano
**aritmetico**, ``dS_t = sigma dW_t``. Nessa especificacao ``sigma`` carrega
unidade de preco por raiz de tempo e ``sigma^2`` unidade de preco ao quadrado.
O preco de reserva desloca-se do mid por ``q * gamma * sigma^2 * tau``, que so
tem sentido dimensional se ``sigma^2`` estiver em preco ao quadrado.

A implementacao original estimava ``sigma`` sobre **log-retornos**, grandeza
adimensional. Para um ativo cotado a USD 1.000, o log-retorno por segundo e da
ordem de 1e-5, de modo que o piso de seguranca ``min_sigma = 1e-3`` passava a
dominar a estimativa em toda a faixa de volatilidade praticavel. Com
``sigma^2`` travado em 1e-6 USD^2, o termo de inventario colapsava para uma
fracao de centesimo de centimo — varias ordens de grandeza abaixo do tick de 1
centimo. O agente degenerava num cotador de spread fixo, sem qualquer resposta
ao inventario retido, e as tres camadas de defesa operavam sobre um controle
que nunca existiu.

O que este script faz
---------------------
Reimplementa ``_estimate_sigma2`` em biblioteca padrao pura, nas duas versoes,
alimenta ambas com um passeio aleatorio calibrado no ambiente real da simulacao
e mede o deslocamento de inventario resultante, em centimos, contra o tick.

Deliberadamente **nao** depende de NumPy, Pandas ou do venv do ABIDES: e a
verificacao que qualquer leitor consegue reproduzir com o Python do sistema.

    python3 tests/verify_sigma.py

Calibracao
----------
Os parametros abaixo replicam ``config/rmsc03_as.py`` e o construtor de
``AvellanedaStoikovAgent``:

- mid ancorado em ``r_bar = 100_000`` centimos (USD 1.000);
- despertar a cada 1 s, sessao de 2 h (7.200 observacoes);
- ``vol_window = 60`` primeiras diferencas;
- ``min_sigma = 1e-3`` **dolares**;
- ``price_scale = 100`` centimos por dolar;
- desvio padrao amostral (``ddof=1``), como o ``np.std(..., ddof=1)`` do agente;
- mid quantizado em centimos inteiros, como o ``mid_cents`` do ABIDES.

Notes
-----
A quantizacao do mid em centimos inteiros e mantida de proposito: e o unico
ponto em que o tick entra na propria estimacao, e ignora-lo superestimaria a
resolucao do estimador nos regimes de volatilidade baixa.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from collections import deque
from typing import Dict, List, Sequence, Tuple

# --- Calibracao herdada do ambiente real -----------------------------------

R_BAR_CENTS: float = 1.0e5
"""Nivel do fundamental, em centimos (`--r-bar` do config, USD 1.000)."""

PRICE_SCALE: float = 100.0
"""Centimos por dolar. O agente guarda o historico de mid em dolares."""

VOL_WINDOW: int = 60
"""Numero de primeiras diferencas usadas na estimativa."""

HISTORY_MAXLEN: int = 10 * VOL_WINDOW
"""Comprimento do deque de mid do agente."""

MIN_SIGMA: float = 1.0e-3
"""Piso de seguranca da volatilidade, em dolares."""

TICK_CENTS: float = 1.0
"""Incremento minimo de preco do RMSC03."""

N_TICKS_DEFAULT: int = 7200
"""Observacoes de uma sessao de 2 h com despertar de 1 s."""

VOL_GRID_CENTS: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
"""Volatilidade do mid varrida, em centimos por segundo."""

INVENTORIES: Tuple[int, ...] = (100, 3596)
"""Inventarios avaliados: um tipico e o pico observado no estudo de ablacao."""

GAMMA: float = 1.0
"""Aversao ao risco usada no config (`--gamma`)."""

TAU: float = 0.5
"""Fracao de horizonte remanescente, no meio da sessao."""


# --- Reimplementacao do estimador ------------------------------------------


def _sample_stdev(values: Sequence[float]) -> float:
    """Desvio padrao amostral (ddof=1) de uma sequencia.

    Parameters
    ----------
    values : sequence of float
        Amostra com pelo menos dois elementos.

    Returns
    -------
    float
        Desvio padrao amostral.

    Notes
    -----
    Usa ``math.fsum`` em vez do somatorio ingenuo porque a amostra do regime de
    baixa volatilidade tem magnitude proxima do epsilon relativo do mid
    (centimos sobre 100.000), onde o cancelamento catastrofico e real.
    """
    n = len(values)
    mean = statistics.fmean(values)
    ss = math.fsum((x - mean) * (x - mean) for x in values)
    return math.sqrt(ss / (n - 1))


def _tail(history: deque, n: int) -> List[float]:
    """Extrai os ultimos ``n`` elementos de um deque sem copiar o resto.

    Parameters
    ----------
    history : collections.deque
        Historico de mid, em dolares.
    n : int
        Numero de elementos desejados a partir do fim.

    Returns
    -------
    list of float
        Ate ``n`` elementos, na ordem original.

    Notes
    -----
    A indexacao de deque a partir da extremidade e barata; converter o deque
    inteiro em lista a cada despertar seria O(600) por observacao e dominaria o
    custo do harness sem ganho de fidelidade.
    """
    size = len(history)
    start = max(0, size - n)
    return [history[i] for i in range(start, size)]


def sigma2_first_differences(history: deque) -> Tuple[float, bool]:
    """Estimador **corrigido**: variancia das primeiras diferencas do preco.

    Reproduz a versao atual de ``AvellanedaStoikovAgent._estimate_sigma2``.

    Parameters
    ----------
    history : collections.deque
        Historico de mid em **dolares**, do mais antigo ao mais recente.

    Returns
    -------
    tuple of (float, bool)
        ``sigma^2`` em dolares ao quadrado e um indicador de que o piso
        ``min_sigma`` foi acionado, isto e, de que a estimativa devolvida nao
        reflete os dados e sim a constante de seguranca.

    Notes
    -----
    Preserva a unidade de preco exigida pelo Browniano aritmetico: se o mid se
    move 5 centimos por segundo, ``sigma = 0,05`` dolares e ``sigma^2 = 2,5e-3``
    dolares ao quadrado, valor que o termo de inventario consegue converter em
    deslocamento visivel no livro.
    """
    if len(history) < 5:
        return MIN_SIGMA * MIN_SIGMA, True

    # A janela de `n` diferencas exige `n + 1` niveis de preco.
    levels = _tail(history, VOL_WINDOW + 1)
    diffs = [levels[i] - levels[i - 1] for i in range(1, len(levels))]

    if len(diffs) < 2:
        return MIN_SIGMA * MIN_SIGMA, True

    sigma = _sample_stdev(diffs)
    floored = math.isnan(sigma) or sigma < MIN_SIGMA
    if floored:
        sigma = MIN_SIGMA
    return sigma * sigma, floored


def sigma2_log_returns(history: deque) -> Tuple[float, bool]:
    """Estimador **defeituoso**: variancia dos log-retornos do preco.

    Reproduz a versao original, anterior a correcao de unidades.

    Parameters
    ----------
    history : collections.deque
        Historico de mid em **dolares**, do mais antigo ao mais recente.

    Returns
    -------
    tuple of (float, bool)
        ``sigma^2`` adimensional (tratado a jusante como se fosse dolares ao
        quadrado) e o indicador de acionamento do piso ``min_sigma``.

    Notes
    -----
    O defeito e de escala, nao de calculo: o log-retorno divide a variacao de
    preco pelo proprio nivel de preco. Com o fundamental em USD 1.000, isso
    encolhe a estimativa por um fator de mil, e o resultado passa a viver
    permanentemente abaixo do piso de 1e-3. O piso deixa de ser salvaguarda
    numerica e vira o proprio estimador.
    """
    if len(history) < 5:
        return MIN_SIGMA * MIN_SIGMA, True

    levels = _tail(history, VOL_WINDOW + 1)
    rets = [
        math.log(levels[i] / levels[i - 1])
        for i in range(1, len(levels))
        if levels[i] > 0.0 and levels[i - 1] > 0.0
    ]

    if len(rets) < 2:
        return MIN_SIGMA * MIN_SIGMA, True

    sigma = _sample_stdev(rets)
    floored = math.isnan(sigma) or sigma < MIN_SIGMA
    if floored:
        sigma = MIN_SIGMA
    return sigma * sigma, floored


# --- Simulacao e medicao ----------------------------------------------------


def inventory_shift_cents(q: int, sigma2: float, gamma: float = GAMMA, tau: float = TAU) -> float:
    """Deslocamento do preco de reserva pelo termo de inventario, em centimos.

    Parameters
    ----------
    q : int
        Inventario liquido, em acoes.
    sigma2 : float
        Variancia estimada, em dolares ao quadrado.
    gamma : float, optional
        Coeficiente de aversao ao risco absoluto.
    tau : float, optional
        Fracao de horizonte remanescente, em [0, 1].

    Returns
    -------
    float
        ``|q| * gamma * sigma^2 * tau`` convertido para centimos.

    Notes
    -----
    Este e o unico canal pelo qual o inventario influencia a cotacao no modelo.
    Quando o valor cai abaixo do tick, o arredondamento do livro o anula por
    completo e o agente cota como se nao tivesse posicao.
    """
    return abs(q) * gamma * sigma2 * tau * PRICE_SCALE


def run_regime(vol_cents: float, n_ticks: int, seed: int) -> Dict[str, float]:
    """Simula um regime de volatilidade e mede os dois estimadores.

    Parameters
    ----------
    vol_cents : float
        Desvio padrao do incremento do mid, em centimos por segundo.
    n_ticks : int
        Numero de despertares simulados.
    seed : int
        Semente do gerador, fixada para tornar o resultado reproduzivel.

    Returns
    -------
    dict
        ``floor_frac_log`` e ``floor_frac_diff`` : fracao de despertares em que
        o piso ``min_sigma`` dominou cada estimador; ``median_s2_log`` e
        ``median_s2_diff`` : mediana de ``sigma^2`` de cada versao.

    Notes
    -----
    O passeio e gerado em centimos e **arredondado para inteiro** antes de ser
    convertido em dolares, exatamente como o ``mid_cents`` do ABIDES. Sem esse
    arredondamento o regime de 0,5 centimo por segundo pareceria mais informativo
    do que e na pratica.
    """
    rng = random.Random(seed)
    history: deque = deque(maxlen=HISTORY_MAXLEN)

    level_cents = R_BAR_CENTS
    s2_log: List[float] = []
    s2_diff: List[float] = []
    floored_log = 0
    floored_diff = 0

    for _ in range(n_ticks):
        level_cents += rng.gauss(0.0, vol_cents)
        mid_cents = int(round(level_cents))
        history.append(mid_cents / PRICE_SCALE)

        value_log, hit_log = sigma2_log_returns(history)
        value_diff, hit_diff = sigma2_first_differences(history)

        s2_log.append(value_log)
        s2_diff.append(value_diff)
        floored_log += int(hit_log)
        floored_diff += int(hit_diff)

    return {
        "floor_frac_log": floored_log / n_ticks,
        "floor_frac_diff": floored_diff / n_ticks,
        "median_s2_log": statistics.median(s2_log),
        "median_s2_diff": statistics.median(s2_diff),
    }


def sweep(vol_grid: Sequence[float], n_ticks: int, seed: int) -> List[Dict[str, float]]:
    """Percorre a grade de volatilidade e consolida os resultados.

    Parameters
    ----------
    vol_grid : sequence of float
        Volatilidades do mid a testar, em centimos por segundo.
    n_ticks : int
        Despertares simulados por regime.
    seed : int
        Semente base; cada regime recebe uma semente derivada distinta, para que
        os regimes nao compartilhem a mesma realizacao do ruido.

    Returns
    -------
    list of dict
        Uma entrada por regime, com a volatilidade sob a chave ``vol_cents`` e
        as medidas devolvidas por :func:`run_regime`.
    """
    rows: List[Dict[str, float]] = []
    for i, vol in enumerate(vol_grid):
        row: Dict[str, float] = {"vol_cents": vol}
        row.update(run_regime(vol, n_ticks, seed + i))
        rows.append(row)
    return rows


# --- Apresentacao -----------------------------------------------------------


def _print_estimator_table(rows: Sequence[Dict[str, float]]) -> None:
    """Imprime a tabela de dominancia do piso e de magnitude de ``sigma^2``."""
    print("=" * 78)
    print(" 1. O piso min_sigma domina a versao (a) em toda a faixa de volatilidade")
    print("=" * 78)
    print(
        f"{'vol mid':>9} | {'piso min_sigma ativo (%)':^23} | "
        f"{'sigma^2 mediano (USD^2)':^27}"
    )
    print(
        f"{'(c/s)':>9} | {'(a) log-ret':>11} {'(b) diff':>11} | "
        f"{'(a) log-ret':>13} {'(b) diff':>13}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{r['vol_cents']:>9.1f} | "
            f"{100.0 * r['floor_frac_log']:>11.1f} {100.0 * r['floor_frac_diff']:>11.1f} | "
            f"{r['median_s2_log']:>13.4e} {r['median_s2_diff']:>13.4e}"
        )
    print()
    print(
        " Nota: a fracao residual da versao (b) sao os primeiros despertares, antes"
    )
    print(
        " de o historico atingir as 5 observacoes minimas exigidas pelo estimador.\n"
    )


def _print_shift_table(rows: Sequence[Dict[str, float]], q: int) -> None:
    """Imprime o deslocamento de inventario de cada versao, em centimos.

    Parameters
    ----------
    rows : sequence of dict
        Saida de :func:`sweep`.
    q : int
        Inventario avaliado.
    """
    print("=" * 78)
    print(
        f" 2. Deslocamento de inventario q*gamma*sigma^2*tau  "
        f"(q={q}, gamma={GAMMA:g}, tau={TAU:g})"
    )
    print("=" * 78)
    print(
        f"{'vol mid':>9} | {'(a) log-ret':>13} {'x tick':>9} | "
        f"{'(b) diff':>13} {'x tick':>9} | veredito"
    )
    print(f"{'(c/s)':>9} | {'centimos':>13} {'':>9} | {'centimos':>13} {'':>9} |")
    print("-" * 78)
    for r in rows:
        shift_a = inventory_shift_cents(q, r["median_s2_log"])
        shift_b = inventory_shift_cents(q, r["median_s2_diff"])
        verdict = "(b) atua, (a) nao" if shift_b >= TICK_CENTS > shift_a else (
            "ambas inertes" if shift_b < TICK_CENTS else "ambas atuam"
        )
        print(
            f"{r['vol_cents']:>9.1f} | "
            f"{shift_a:>13.5f} {shift_a / TICK_CENTS:>9.4f} | "
            f"{shift_b:>13.5f} {shift_b / TICK_CENTS:>9.2f} | {verdict}"
        )
    print()


def _print_conclusion(rows: Sequence[Dict[str, float]]) -> None:
    """Resume o veredito quantitativo do harness."""
    print("=" * 78)
    print(" 3. Conclusao")
    print("=" * 78)
    print(f" Tick do RMSC03: {TICK_CENTS:.0f} centimo. Abaixo dele o livro arredonda o")
    print(" deslocamento para zero e o controlo de inventario deixa de existir.\n")

    for q in INVENTORIES:
        worst_a = max(inventory_shift_cents(q, r["median_s2_log"]) for r in rows)
        best_b = min(inventory_shift_cents(q, r["median_s2_diff"]) for r in rows)
        max_b = max(inventory_shift_cents(q, r["median_s2_diff"]) for r in rows)
        print(f" q = {q}:")
        print(
            f"   (a) log-retornos : maximo em toda a faixa = {worst_a:.5f} c "
            f"({worst_a / TICK_CENTS:.4f} tick)"
        )
        print(
            f"   (b) diferencas   : de {best_b:.3f} c a {max_b:.1f} c "
            f"({best_b / TICK_CENTS:.2f} a {max_b / TICK_CENTS:.1f} ticks)"
        )
        print()

    min_floor_log = min(r["floor_frac_log"] for r in rows)
    print(
        f" Na versao (a) o piso min_sigma domina em pelo menos "
        f"{100.0 * min_floor_log:.1f}% dos despertares"
    )
    print(" de todos os regimes varridos, de 0,5 a 50 centimos por segundo.")
    print(
        " Ou seja: sob log-retornos, sigma^2 nao e estimado — e a constante "
        "min_sigma^2 = 1e-6,"
    )
    print(
        " invariante a volatilidade do mercado. O agente cota o mesmo spread com "
        "inventario"
    )
    print(" zero e com inventario de milhares de acoes.")
    print("=" * 78)


def main() -> None:
    """Executa a varredura e imprime o relatorio completo."""
    parser = argparse.ArgumentParser(
        description="Verificacao de unidades do estimador de sigma^2 do agente A-S"
    )
    parser.add_argument(
        "--ticks", type=int, default=N_TICKS_DEFAULT,
        help="Despertares por regime (default: 7200, sessao de 2 h a 1 s).",
    )
    parser.add_argument(
        "--seed", type=int, default=20240101,
        help="Semente base do passeio aleatorio.",
    )
    args = parser.parse_args()

    print()
    print("#" * 78)
    print(" Verificacao de unidades de sigma^2 — AvellanedaStoikovAgent")
    print(f" mid ancorado em {R_BAR_CENTS:.0f} centimos | vol_window={VOL_WINDOW} | "
          f"min_sigma={MIN_SIGMA:g} USD")
    print(f" {args.ticks} despertares de 1 s por regime | semente base {args.seed}")
    print("#" * 78)
    print()

    rows = sweep(VOL_GRID_CENTS, args.ticks, args.seed)

    _print_estimator_table(rows)
    for q in INVENTORIES:
        _print_shift_table(rows, q)
    _print_conclusion(rows)


if __name__ == "__main__":
    main()
