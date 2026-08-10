"""Series temporais de patrimonio e estoque do agente Avellaneda-Stoikov.

O CSV agregado do grid search (`optimization_as.py`) reporta apenas o estado
terminal de cada execucao: PnL final, Sharpe, inventario maximo. Isso responde
*quanto* o agente ganhou, mas nao *como* — se o resultado veio de acumulacao
regular de spread ou de uma unica posicao direcional que deu certo, e se o
controle de inventario da HJB de fato manteve o estoque em torno de zero ao
longo da sessao.

Este modulo reconstroi a trajetoria completa a partir do log de cotacoes e
oferece as tres operacoes necessarias para compara-las entre execucoes:

1. carga da serie de uma execucao (:func:`load_equity_series`);
2. reamostragem numa grade temporal comum (:func:`align_series`), sem a qual
   series de execucoes distintas nao sao sobreponiveis — os instantes de
   cotacao dependem da realizacao estocastica de cada semente;
3. agregacao entre sementes com banda de confianca (:func:`aggregate_seeds`).

Unidades
--------
O log do ABIDES esta integralmente em centimos. Este modulo entrega **tudo em
dolares**: `equity`, `pnl` e `mid` sao convertidos, e `inv` permanece em acoes.
Note que isso difere de :func:`analysis.parser.parse_as_metrics`, que converte
apenas `PnL`/`Equity` e devolve `mid` em centimos. A escolha aqui e por um
DataFrame de unidade unica, para que qualquer coluna possa ir direto ao eixo de
um grafico sem fator de correcao implicito.
"""

from __future__ import annotations

import contextlib
import io
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from analysis.parser import load_agent_log, parse_as_metrics

#: Capital inicial do agente, em centimos (US$ 100.000). Mesmo valor usado pelos
#: orquestradores; precisa coincidir com o `starting_cash` da simulacao, sob pena
#: de deslocar a curva de patrimonio por uma constante.
STARTING_CASH_CENTS: int = 10_000_000

#: Fator de conversao de centimos para dolares.
CENTS_PER_DOLLAR: float = 100.0

#: Quantil da normal padrao para um intervalo de confianca bilateral de 95%.
Z_95: float = 1.96

#: Colunas devolvidas por :func:`load_equity_series`, nesta ordem.
EQUITY_COLUMNS: Sequence[str] = ("equity", "pnl", "inv", "mid")


def load_equity_series(
    log_dir: str, starting_cash: int = STARTING_CASH_CENTS
) -> pd.DataFrame:
    """Carrega a trajetoria de patrimonio e estoque de uma execucao.

    Parameters
    ----------
    log_dir : str
        Diretorio de log da simulacao, contendo
        ``AVELLANEDA_STOIKOV_AGENT.bz2``.
    starting_cash : int, optional
        Capital inicial em centimos, propagado a
        :func:`analysis.parser.parse_as_metrics`. Deve coincidir com o valor
        usado na simulacao.

    Returns
    -------
    pd.DataFrame
        Indexado por ``EventTime`` (DatetimeIndex, ordenado), com as colunas
        ``equity`` e ``pnl`` em dolares, ``inv`` em acoes e ``mid`` em dolares.

    Raises
    ------
    FileNotFoundError
        Se o log do agente nao existir no diretorio.
    ValueError
        Se o log nao contiver eventos de cotacao (``AS_QUOTE``).

    Notes
    -----
    :func:`analysis.parser.parse_as_metrics` emite progresso em ``stdout``. Como
    este modulo e consumido por scripts que produzem saida estruturada, a
    impressao e capturada e descartada — o unico canal de erro continua sendo a
    excecao.
    """
    # O parser do projeto imprime progresso; silencia-lo aqui evita poluir a
    # saida de quem consome este modulo, sem mascarar excecoes.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        parsed = parse_as_metrics(
            load_agent_log(log_dir), starting_cash=starting_cash
        )

    series = pd.DataFrame(
        {
            "equity": parsed["Equity"].astype(float),
            "pnl": parsed["PnL"].astype(float),
            "inv": parsed["inv"].astype(float),
            "mid": parsed["mid"].astype(float) / CENTS_PER_DOLLAR,
        }
    )

    # O kernel do ABIDES processa eventos em ordem cronologica, mas o log e
    # concatenado por agente; ordenar torna a reamostragem determinista.
    return series.sort_index()


def align_series(
    series: Dict[str, pd.DataFrame], freq: str = "10S"
) -> Dict[str, pd.DataFrame]:
    """Reamostra varias series numa grade temporal comum.

    Parameters
    ----------
    series : dict of str to pd.DataFrame
        Series indexadas no tempo, tipicamente uma por execucao, com o rotulo da
        execucao como chave.
    freq : str, optional
        Frequencia da grade, na notacao de offset do pandas. O padrao de 10 s
        reduz as ~7.100 cotacoes de uma sessao de 2 h a ~715 pontos, suficiente
        para a curva e barato de sobrepor.

    Returns
    -------
    dict of str to pd.DataFrame
        As mesmas chaves, com cada DataFrame reamostrado na grade.

    Notes
    -----
    A reducao e ``last`` seguida de ``ffill``: patrimonio e estoque sao
    variaveis de **estado**, nao fluxos. O valor que vale num intervalo e o
    ultimo observado, e um intervalo sem cotacao significa estado inalterado —
    nao ausencia de dado. Media ou soma no lugar de ``last`` distorceriam o
    nivel; ``interpolate`` inventaria transicoes que nao ocorreram.

    Os instantes de cotacao dependem da realizacao estocastica de cada semente,
    de modo que series de execucoes distintas nao compartilham indice. Sem esta
    etapa, sobrepor ou agregar curvas exigiria um alinhamento implicito e
    silencioso do pandas.
    """
    aligned: Dict[str, pd.DataFrame] = {}
    for label, frame in series.items():
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(
                f"A serie '{label}' nao tem indice temporal "
                f"(recebido: {type(frame.index).__name__}); a reamostragem "
                "exige DatetimeIndex."
            )
        aligned[label] = frame.resample(freq).last().ffill()
    return aligned


def aggregate_seeds(series: Sequence[pd.DataFrame], column: str) -> pd.DataFrame:
    """Agrega uma coluna entre sementes, com banda de confianca de 95%.

    Parameters
    ----------
    series : sequence of pd.DataFrame
        Series **ja alinhadas** por :func:`align_series`, uma por semente.
    column : str
        Coluna a agregar (``equity``, ``pnl``, ``inv`` ou ``mid``).

    Returns
    -------
    pd.DataFrame
        Indexado pela grade temporal, com as colunas ``mean``, ``lo`` e ``hi``,
        onde ``lo``/``hi`` sao a media mais ou menos 1,96 erro padrao.

    Raises
    ------
    ValueError
        Se a sequencia estiver vazia.
    KeyError
        Se alguma serie nao contiver a coluna pedida.

    Notes
    -----
    A banda e o intervalo de confianca da **media** entre sementes, nao a
    dispersao das realizacoes: com 4 sementes ela e larga por construcao e serve
    para indicar quanto da curva media e ruido amostral. A dispersao bruta entre
    realizacoes deve ser mostrada plotando as sementes individualmente.

    Instantes cobertos por menos de duas sementes recebem ``lo``/``hi`` nulos
    (NaN): o erro padrao e indefinido com uma unica observacao, e preencher com
    a propria media sugeriria uma precisao inexistente. Como todas as execucoes
    do grid cobrem a mesma janela de pregao, isso so ocorre nas bordas quando as
    series tem duracoes distintas.
    """
    if len(series) == 0:
        raise ValueError("Nenhuma serie fornecida para agregacao.")

    columns: List[pd.Series] = []
    for i, frame in enumerate(series):
        if column not in frame.columns:
            raise KeyError(
                f"Coluna '{column}' ausente na serie de indice {i}; "
                f"disponiveis: {sorted(frame.columns)}."
            )
        columns.append(frame[column].astype(float))

    # Uniao dos indices: uma semente mais curta nao trunca as demais, apenas
    # deixa de contribuir para a media nos instantes que nao cobre.
    panel = pd.concat(columns, axis=1, join="outer")
    panel.columns = pd.RangeIndex(len(columns))

    count = panel.count(axis=1)
    mean = panel.mean(axis=1)
    # ddof=1 porque as sementes sao uma amostra das realizacoes possiveis do
    # mercado, nao a populacao.
    std = panel.std(axis=1, ddof=1)
    half_width = Z_95 * std / np.sqrt(count.where(count > 0))
    half_width = half_width.where(count >= 2)

    return pd.DataFrame(
        {"mean": mean, "lo": mean - half_width, "hi": mean + half_width}
    )
