"""Diagnostico de posicionamento do agente A-S no livro de ofertas.

Cruza a serie de cotacoes emitidas pelo agente Avellaneda-Stoikov com o topo do
livro efetivamente praticado pelo mercado, respondendo a pergunta que o PnL
agregado nao responde: *o agente chegou a disputar a fila?*

O achado central do trabalho e que nao. Num mercado cujo spread esta fixado em
um tick, o meio-spread otimo da HJB, ``gamma^-1 * ln(1 + gamma/k)``, coloca o
agente uma ordem de grandeza atras do topo. Ele quase nunca esta no touch, e as
poucas execucoes que obtem vem de varredura adversa do livro, nao de provisao de
liquidez. Este modulo mede exatamente isso.

Convencao do arquivo ``ORDERBOOK_<sym>_FREQ_<freq>.bz2``
-------------------------------------------------------
O ABIDES grava o livro como matriz larga esparsa: linhas sao instantes
amostrados na frequencia ``book_freq``, colunas sao **niveis de preco em
centimos** e os valores sao **volumes com sinal**. Verificado empiricamente:

- volume **negativo** => nivel do lado **bid**;
- volume **positivo** => nivel do lado **ask**;
- volume zero => nivel vazio naquele instante.

Logo ``best_bid`` e o **maior** preco com volume negativo e ``best_ask`` e o
**menor** preco com volume positivo. Nao ha coluna de lado: o sinal e a unica
marcacao disponivel, e inverte-lo troca as duas pontas do livro sem qualquer
erro visivel a jusante.

O arquivo so existe quando a simulacao roda **sem** ``--lean-logs`` (a flag zera
``book_freq`` em ``config/rmsc03_as.py``). Rodadas de referencia destinadas a
diagnostico precisam, portanto, pagar o custo do log completo.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.parser import load_agent_log, parse_as_metrics

#: Nome do evento de submissao de ordem no log do TradingAgent do ABIDES.
EVENT_ORDER_SUBMITTED: str = "ORDER_SUBMITTED"

#: Nome do evento de execucao (fill) no log do TradingAgent do ABIDES.
EVENT_ORDER_EXECUTED: str = "ORDER_EXECUTED"

#: Padrao dos arquivos de serie do livro gravados pela ExchangeAgent.
_ORDERBOOK_GLOB: str = "ORDERBOOK_*_FREQ_*.bz2"


def find_orderbook_file(log_dir: str, symbol: Optional[str] = None) -> str:
    """Localiza o arquivo de serie do livro de ofertas dentro de um log.

    Parameters
    ----------
    log_dir : str
        Diretorio de log da simulacao (ex.: ``log/CALIB_REF``).
    symbol : str, optional
        Ticker a procurar. Quando omitido, aceita qualquer ticker desde que
        exista exatamente um arquivo de livro no diretorio.

    Returns
    -------
    str
        Caminho absoluto ou relativo do arquivo encontrado.

    Raises
    ------
    FileNotFoundError
        Se nenhum arquivo casar com o padrao. O caso tipico e a simulacao ter
        sido executada com ``--lean-logs``, que suprime a serie do livro.
    ValueError
        Se houver mais de um ticker no mesmo diretorio e ``symbol`` for omitido.
    """
    pattern = (
        os.path.join(log_dir, f"ORDERBOOK_{symbol}_FREQ_*.bz2")
        if symbol is not None
        else os.path.join(log_dir, _ORDERBOOK_GLOB)
    )
    matches = sorted(glob.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"Serie do livro nao encontrada em {log_dir!r} (padrao {pattern!r}). "
            "A simulacao provavelmente rodou com --lean-logs, que zera book_freq "
            "e suprime o ORDERBOOK_*.bz2. Diagnostico de posicionamento exige "
            "rodada sem --lean-logs."
        )
    if len(matches) > 1:
        tickers = sorted(
            {m.group(1) for m in (re.search(r"ORDERBOOK_(.+)_FREQ_", p) for p in matches) if m}
        )
        raise ValueError(
            f"Multiplos livros em {log_dir!r} (tickers {tickers}). "
            "Informe o argumento `symbol` para desambiguar."
        )
    return matches[0]


def _densify(book: pd.DataFrame) -> pd.DataFrame:
    """Converte a matriz esparsa do livro para densa, preservando o indice.

    Parameters
    ----------
    book : pd.DataFrame
        Matriz larga lida do pickle, tipicamente com todas as colunas em
        ``SparseDtype``.

    Returns
    -------
    pd.DataFrame
        Mesma matriz em dtype denso ``float64``, com colunas ordenadas em ordem
        crescente de preco.

    Notes
    -----
    A ordenacao das colunas nao e cosmetica: `_best_quotes` localiza o topo de
    cada lado por posicao no eixo das colunas, e assume monotonicidade dos
    precos. O ABIDES ja grava ordenado, mas a ordenacao explicita torna a
    invariante local em vez de herdada.
    """
    all_sparse = all(isinstance(dtype, pd.SparseDtype) for dtype in book.dtypes)
    dense = book.sparse.to_dense() if all_sparse else book
    return dense.astype(np.float64).sort_index(axis=1)


def _best_quotes(dense: pd.DataFrame) -> pd.DataFrame:
    """Extrai o topo dos dois lados do livro, de forma vetorizada.

    Parameters
    ----------
    dense : pd.DataFrame
        Livro denso com colunas em ordem crescente de preco (centimos) e
        valores em volume com sinal.

    Returns
    -------
    pd.DataFrame
        Colunas ``best_bid`` e ``best_ask``, em centimos, indexadas pelos
        mesmos instantes do livro. Instantes em que um dos lados esta vazio
        recebem ``NaN`` naquele lado.

    Notes
    -----
    Sinal negativo marca bid e positivo marca ask (ver docstring do modulo).
    Como as colunas estao ordenadas, o melhor bid e a **ultima** coluna
    negativa e o melhor ask e a **primeira** coluna positiva; ambos saem de um
    unico ``argmax`` sobre a mascara booleana, sem laco no interpretador.
    """
    prices = dense.columns.values.astype(np.float64)
    values = dense.values

    is_bid = values < 0.0
    is_ask = values > 0.0

    n_levels = values.shape[1]
    # argmax sobre a mascara invertida devolve a ultima ocorrencia do lado bid.
    idx_bid = n_levels - 1 - np.argmax(is_bid[:, ::-1], axis=1)
    idx_ask = np.argmax(is_ask, axis=1)

    has_bid = is_bid.any(axis=1)
    has_ask = is_ask.any(axis=1)

    best_bid = np.where(has_bid, prices[idx_bid], np.nan)
    best_ask = np.where(has_ask, prices[idx_ask], np.nan)

    return pd.DataFrame(
        {"best_bid": best_bid, "best_ask": best_ask}, index=dense.index
    )


def load_book_top(log_dir: str, symbol: Optional[str] = None) -> pd.DataFrame:
    """Le a serie do livro e devolve apenas o topo de cada lado.

    Parameters
    ----------
    log_dir : str
        Diretorio de log da simulacao.
    symbol : str, optional
        Ticker do arquivo de livro. Ver :func:`find_orderbook_file`.

    Returns
    -------
    pd.DataFrame
        Colunas ``best_bid`` e ``best_ask`` em centimos, ordenadas no tempo,
        ja sem os instantes em que qualquer um dos lados esteja vazio.
    """
    path = find_orderbook_file(log_dir, symbol)
    dense = _densify(pd.read_pickle(path))
    top = _best_quotes(dense).dropna(subset=["best_bid", "best_ask"])
    return top.sort_index()


def count_order_events(raw_log: pd.DataFrame) -> Tuple[int, int]:
    """Conta submissoes e execucoes no log bruto do agente.

    Parameters
    ----------
    raw_log : pd.DataFrame
        Log cru do agente (saida de :func:`analysis.parser.load_agent_log`),
        com a coluna ``EventType``.

    Returns
    -------
    tuple of int
        ``(n_orders_submitted, n_fills)``.

    Notes
    -----
    ``ORDER_EXECUTED`` conta **eventos de execucao**, nao ordens distintas: uma
    ordem varrida em varios niveis gera varios eventos. Isso e desejavel aqui,
    porque o denominador (``ORDER_SUBMITTED``) tambem conta cada envio
    individual de limite. A razao entre os dois e uma taxa de conversao de
    tentativa em negocio, nao uma probabilidade por ordem unica.

    Cuidado correlato: ordens a mercado no ABIDES retornam com ``order_id``
    diferente do submetido, porque a bolsa decompoe a ordem ao percorrer o
    livro. Qualquer contagem baseada em casar ids submetidos com ids executados
    subestimaria as execucoes; contar eventos evita o problema.
    """
    if "EventType" not in raw_log.columns:
        raise KeyError(
            "Log do agente sem coluna 'EventType'. O pipeline de posicionamento "
            "depende dela para contar ORDER_SUBMITTED e ORDER_EXECUTED."
        )
    events = raw_log["EventType"]
    n_submitted = int((events == EVENT_ORDER_SUBMITTED).sum())
    n_fills = int((events == EVENT_ORDER_EXECUTED).sum())
    return n_submitted, n_fills


def quote_positioning(
    log_dir: str,
    symbol: Optional[str] = None,
    agent_name: str = "AVELLANEDA_STOIKOV_AGENT",
) -> Dict[str, Any]:
    """Mede onde as cotacoes do agente ficam em relacao ao topo do livro.

    Parameters
    ----------
    log_dir : str
        Diretorio de log de uma simulacao executada **sem** ``--lean-logs``.
    symbol : str, optional
        Ticker do arquivo ``ORDERBOOK_<sym>_FREQ_*.bz2``. Autodetectado quando
        ha um unico livro no diretorio.
    agent_name : str, optional
        Nome do arquivo de log do agente, sem extensao.

    Returns
    -------
    dict
        ``market_spread_median`` : float
            Mediana de ``best_ask - best_bid`` do mercado, em centimos, sobre a
            amostra alinhada as cotacoes do agente.
        ``agent_spread_median`` : float
            Mediana de ``ask - bid`` do agente, em centimos.
        ``bid_gap_median`` : float
            Mediana de ``best_bid - bid_agente``. Positivo significa bid do
            agente **abaixo** do topo, isto e, atras na fila.
        ``ask_gap_median`` : float
            Mediana de ``ask_agente - best_ask``. Positivo significa ask do
            agente **acima** do topo, isto e, atras na fila.
        ``pct_time_bid_at_touch`` : float
            Percentual de cotacoes com ``bid >= best_bid``.
        ``pct_time_ask_at_touch`` : float
            Percentual de cotacoes com ``ask <= best_ask``.
        ``n_orders_submitted`` : int
            Eventos ``ORDER_SUBMITTED`` no log do agente.
        ``n_fills`` : int
            Eventos ``ORDER_EXECUTED`` no log do agente.
        ``fill_rate`` : float
            ``n_fills / n_orders_submitted``. ``nan`` se nada foi submetido.
        ``n_quotes`` : int
            Cotacoes do agente efetivamente alinhadas a um estado do livro.
        ``n_book_snapshots`` : int
            Instantes do livro com os dois lados presentes.

    Notes
    -----
    **Alinhamento temporal.** O livro e amostrado em grade fixa (1 s por
    default) e o agente cota em instantes arbitrarios, deslocados pela latencia
    par-a-par do kernel. O casamento usa ``merge_asof`` com
    ``direction="backward"``: cada cotacao e comparada ao ultimo estado do livro
    **conhecido no momento em que ela foi emitida**. Usar ``"nearest"`` ou
    ``"forward"`` compararia a cotacao com um livro que ja incorpora o efeito
    dela propria, viesando o diagnostico a favor do agente. O custo e perder as
    cotacoes anteriores ao primeiro instante amostrado do livro.

    **O agente esta dentro do livro que ele proprio mede.** ``best_bid`` e
    ``best_ask`` incluem as ordens do agente A-S. Enquanto ele cota atras do
    topo, isso e irrelevante; quando ele estivesse no touch, ``bid >= best_bid``
    seria satisfeito trivialmente pela sua propria ordem. A metrica de touch e,
    portanto, um limite **superior** da presenca efetiva no topo.

    **Unidades.** Tudo em centimos, como o resto da simulacao. O tick do RMSC03
    e 1 centimo, de modo que os valores devolvidos ja estao expressos em ticks.
    """
    raw_log = load_agent_log(log_dir, agent_name=agent_name)
    quotes = parse_as_metrics(raw_log)

    book = load_book_top(log_dir, symbol)

    # merge_asof exige as duas series ordenadas e a chave como coluna, nao indice.
    left = quotes[["bid", "ask", "mid"]].sort_index().reset_index()
    time_col = left.columns[0]
    right = book.reset_index()
    right.columns = [time_col, "best_bid", "best_ask"]

    merged = pd.merge_asof(left, right, on=time_col, direction="backward")
    merged = merged.dropna(subset=["best_bid", "best_ask"])

    if merged.empty:
        raise ValueError(
            f"Nenhuma cotacao do agente pode ser alinhada ao livro em {log_dir!r}. "
            "Verifique se o livro cobre a janela de atuacao do agente."
        )

    market_spread = merged["best_ask"] - merged["best_bid"]
    agent_spread = merged["ask"] - merged["bid"]
    bid_gap = merged["best_bid"] - merged["bid"]
    ask_gap = merged["ask"] - merged["best_ask"]

    n_submitted, n_fills = count_order_events(raw_log)

    return {
        "market_spread_median": float(market_spread.median()),
        "agent_spread_median": float(agent_spread.median()),
        "bid_gap_median": float(bid_gap.median()),
        "ask_gap_median": float(ask_gap.median()),
        "pct_time_bid_at_touch": float(
            100.0 * (merged["bid"] >= merged["best_bid"]).mean()
        ),
        "pct_time_ask_at_touch": float(
            100.0 * (merged["ask"] <= merged["best_ask"]).mean()
        ),
        "n_orders_submitted": n_submitted,
        "n_fills": n_fills,
        "fill_rate": (float(n_fills) / n_submitted) if n_submitted else float("nan"),
        "n_quotes": int(len(merged)),
        "n_book_snapshots": int(len(book)),
    }


def _main() -> None:
    """CLI de conveniencia: imprime o diagnostico de um diretorio de log."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnostico de posicionamento do agente A-S no livro"
    )
    parser.add_argument(
        "--log-dir", type=str, default="log/CALIB_REF",
        help="Diretorio de log de uma simulacao rodada sem --lean-logs.",
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Ticker do arquivo de livro. Autodetectado se omitido.",
    )
    args = parser.parse_args()

    stats = quote_positioning(args.log_dir, symbol=args.symbol)

    print("\n" + "=" * 62)
    print("  Posicionamento das cotacoes A-S vs. topo do livro")
    print("=" * 62)
    print(f"  log_dir                        : {args.log_dir}")
    print(f"  cotacoes alinhadas             : {stats['n_quotes']:>12,}")
    print(f"  instantes do livro             : {stats['n_book_snapshots']:>12,}")
    print("-" * 62)
    print(f"  spread do mercado (mediana)    : {stats['market_spread_median']:>12.2f} c")
    print(f"  spread do agente  (mediana)    : {stats['agent_spread_median']:>12.2f} c")
    print(f"  bid atras do topo (mediana)    : {stats['bid_gap_median']:>12.2f} c")
    print(f"  ask atras do topo (mediana)    : {stats['ask_gap_median']:>12.2f} c")
    print(f"  tempo com bid no topo          : {stats['pct_time_bid_at_touch']:>12.3f} %")
    print(f"  tempo com ask no topo          : {stats['pct_time_ask_at_touch']:>12.3f} %")
    print("-" * 62)
    print(f"  ordens submetidas              : {stats['n_orders_submitted']:>12,}")
    print(f"  execucoes                      : {stats['n_fills']:>12,}")
    print(f"  taxa de conversao              : {100.0 * stats['fill_rate']:>12.3f} %")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    _main()
