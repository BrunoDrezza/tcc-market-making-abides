"""
analysis/diagnostics.py

Diagnosticos de microestrutura e de ecologia sobre logs ja gerados pelo ABIDES.

Consolida as analises que antes existiam apenas como codigo colado no terminal:
topo do livro, participacao de volume por faixa de agente, latencia de uma via,
estimacao do ``k`` de Avellaneda-Stoikov, escala do deslocamento induzido pelo
preditor OBI e efeitos marginais entre celulas do estudo de ablacao.

Todas as funcoes retornam dados (DataFrame ou dict) e nunca imprimem: a
formatacao para leitura humana e responsabilidade de ``run_diagnostics.py``.

Notes
-----
Precos e caixa no ABIDES estao em centavos; nenhuma funcao deste modulo
converte para dolares, exceto onde a unidade e explicitamente exigida pela
formula do modelo (ver :func:`estimate_k`).
"""

import contextlib
import glob
import io
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.parser import load_agent_log, parse_as_metrics

# Faixas de agent_id da ecologia default do rmsc03_as (--num-noise 5000,
# --num-value 100, --mm-count 2, 25 momentum fixos, 1 A-S, 1 POV opcional).
# Nao importamos config.rmsc03_as para deriva-las porque aquele modulo executa
# a simulacao inteira em import time; qualquer override deve vir pela CLI.
DEFAULT_AGENT_RANGES: Dict[str, Tuple[int, int]] = {
    "exchange": (0, 0),
    "noise": (1, 5000),
    "value": (5001, 5100),
    "adaptive_mm": (5101, 5102),
    "momentum": (5103, 5127),
    "as_agent": (5128, 5128),
    "pov": (5129, 5129),
}

# Pares (base, tratamento, rotulo) do desenho fatorial 2^3 de run_ablation_study.py.
DEFAULT_ABLATION_PAIRS: Tuple[Tuple[int, int, str], ...] = (
    (1, 5, "OBI isolado"),
    (1, 3, "HEDGE isolado"),
    (1, 2, "KILL isolado"),
    (1, 8, "Stack completo"),
)

_EXCHANGE_LOG = "EXCHANGE_AGENT.bz2"
_ORDERBOOK_GLOB = "ORDERBOOK_*_FREQ_1S.bz2"


def _load_exchange_events(log_dir: str, event_type: str) -> pd.DataFrame:
    """Carrega o log da bolsa e expande os dicts de um unico ``EventType``.

    Parameters
    ----------
    log_dir : str
        Diretorio do log da simulacao (ex.: ``log/CALIB_REF``).
    event_type : str
        Valor exato da coluna ``EventType`` a reter (ex.: ``'ORDER_EXECUTED'``).

    Returns
    -------
    pd.DataFrame
        Um campo do dict ``Event`` por coluna, preservando o ``DatetimeIndex``
        original do log da bolsa.

    Raises
    ------
    FileNotFoundError
        Se ``EXCHANGE_AGENT.bz2`` nao existir no diretorio.
    ValueError
        Se nenhum evento do tipo pedido existir no log.

    Notes
    -----
    A coluna ``Event`` guarda dicts Python, entao nao ha operacao vetorizada
    que a leia in-place. ``DataFrame(series.tolist())`` faz a expansao numa
    unica passagem em C, o que e uma ordem de grandeza mais rapido do que
    ``.map`` campo a campo e mantem o restante do pipeline vetorizado.
    """
    path = os.path.join(log_dir, _EXCHANGE_LOG)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Log da bolsa nao encontrado: {}. Rode a simulacao sem --lean-logs.".format(path)
        )

    raw = pd.read_pickle(path)
    events = raw.loc[raw["EventType"] == event_type, "Event"]
    if events.empty:
        raise ValueError(
            "Nenhum evento '{}' no log da bolsa em {}.".format(event_type, log_dir)
        )

    expanded = pd.DataFrame(events.tolist(), index=events.index)
    # O log da bolsa carrega uma linha AGENT_TYPE com indice NaT; qualquer
    # operacao temporal precisa dela fora do caminho.
    return expanded.loc[expanded.index.notna()]


def load_orderbook_top(log_dir: str) -> pd.DataFrame:
    """Reconstroi o topo do livro a partir do snapshot de 1 segundo.

    Parameters
    ----------
    log_dir : str
        Diretorio do log da simulacao contendo ``ORDERBOOK_<symbol>_FREQ_1S.bz2``.

    Returns
    -------
    pd.DataFrame
        Indexado pelo ``DatetimeIndex`` do snapshot, com as colunas
        ``best_bid``, ``best_ask``, ``mid`` e ``spread``, todas em centavos.
        Linhas sem os dois lados do livro, ou com ``spread <= 0``, sao
        descartadas.

    Raises
    ------
    FileNotFoundError
        Se nenhum arquivo de orderbook casar com o padrao no diretorio.

    Notes
    -----
    O arquivo e uma matriz larga: cada coluna e um nivel de preco (em centavos)
    e cada celula e o volume naquele nivel.

    **Convencao de sinal, verificada empiricamente em 22 amostras:** volume
    **negativo** identifica o lado comprador (bid) e volume **positivo** o lado
    vendedor (ask). Logo ``best_bid`` e o **maior** preco com volume negativo e
    ``best_ask`` e o **menor** preco com volume positivo. Inverter essa
    convencao produz um spread negativo em praticamente toda a sessao, que e
    justamente o filtro de sanidade aplicado no final.

    O snapshot e gravado em formato esparso; ele e densificado aqui porque a
    varredura por linha precisa do retangulo completo.
    """
    matches = sorted(glob.glob(os.path.join(log_dir, _ORDERBOOK_GLOB)))
    if not matches:
        raise FileNotFoundError(
            "Nenhum {} em {}. O log foi gerado com --lean-logs?".format(
                _ORDERBOOK_GLOB, log_dir
            )
        )

    frame = pd.read_pickle(matches[0])
    try:
        dense = frame.sparse.to_dense()
    except AttributeError:
        # O acessor .sparse so existe quando todas as colunas sao esparsas;
        # se o snapshot ja veio denso, seguimos com ele como esta.
        dense = frame
    dense = dense.astype("float64")

    volumes = dense.to_numpy()
    prices = dense.columns.to_numpy(dtype="float64")

    # Varredura vetorizada: mascaramos os niveis do lado errado com +/-inf para
    # que min/max ao longo do eixo de precos devolva o topo de cada lado sem
    # nenhum laco Python sobre as 7200 linhas.
    price_grid = np.broadcast_to(prices, volumes.shape)
    best_bid = np.where(volumes < 0.0, price_grid, -np.inf).max(axis=1)
    best_ask = np.where(volumes > 0.0, price_grid, np.inf).min(axis=1)

    top = pd.DataFrame(
        {"best_bid": best_bid, "best_ask": best_ask}, index=dense.index
    )
    top = top.replace([np.inf, -np.inf], np.nan).dropna()
    top["mid"] = (top["best_bid"] + top["best_ask"]) / 2.0
    top["spread"] = top["best_ask"] - top["best_bid"]

    # Spread nao-positivo denuncia livro cruzado ou snapshot de um lado so:
    # nada disso e observacao valida de topo de livro.
    return top.loc[top["spread"] > 0.0]


def volume_share_by_agent(
    log_dir: str, agent_ranges: Dict[str, Tuple[int, int]]
) -> pd.DataFrame:
    """Agrega o volume executado por faixa de ``agent_id``.

    Parameters
    ----------
    log_dir : str
        Diretorio do log da simulacao.
    agent_ranges : dict of str to tuple of int
        Mapa ``nome do grupo -> (id_inicial, id_final)``, ambos inclusivos.
        Ver :data:`DEFAULT_AGENT_RANGES` para a ecologia padrao do rmsc03_as.

    Returns
    -------
    pd.DataFrame
        Colunas ``grupo``, ``execucoes``, ``acoes`` e ``pct_volume``, ordenada
        por ``acoes`` em ordem decrescente. ``pct_volume`` e a fracao do volume
        executado **total** do log, nao apenas o coberto pelas faixas.

    Notes
    -----
    Somente o log da bolsa registra a atividade de todos os agentes: os logs
    individuais so existem para quem foi instanciado com ``log_orders=True``.
    Por isso a agregacao parte de ``EXCHANGE_AGENT.bz2`` e nao da uniao dos
    arquivos por agente.

    Cada preenchimento gera um evento ``ORDER_EXECUTED`` por lado, entao a soma
    de ``acoes`` sobre todos os grupos conta o volume duas vezes em relacao ao
    volume nocional negociado. As participacoes relativas nao sao afetadas.
    """
    executed = _load_exchange_events(log_dir, "ORDER_EXECUTED")
    agent_id = executed["agent_id"].to_numpy()
    quantity = executed["quantity"].to_numpy()

    total_volume = float(quantity.sum())

    rows: List[Dict[str, object]] = []
    for group, (low, high) in agent_ranges.items():
        mask = (agent_id >= low) & (agent_id <= high)
        shares = float(quantity[mask].sum())
        rows.append(
            {
                "grupo": group,
                "execucoes": int(mask.sum()),
                "acoes": shares,
                "pct_volume": 100.0 * shares / total_volume if total_volume else np.nan,
            }
        )

    result = pd.DataFrame(rows, columns=["grupo", "execucoes", "acoes", "pct_volume"])
    return result.sort_values("acoes", ascending=False).reset_index(drop=True)


def latency_by_agent(log_dir: str) -> pd.DataFrame:
    """Mede a latencia de uma via (agente -> bolsa) por agente.

    Parameters
    ----------
    log_dir : str
        Diretorio do log da simulacao.

    Returns
    -------
    pd.DataFrame
        Uma linha por ``agent_id``, com as colunas ``agent_id``, ``n``,
        ``mediana_ms``, ``min_ms`` e ``max_ms``, ordenada por ``mediana_ms``.

    Notes
    -----
    A latencia e calculada sobre eventos ``LIMIT_ORDER`` como o instante em que
    a bolsa registrou a ordem (indice do log da bolsa) menos ``time_placed``
    (instante em que o agente a emitiu). E portanto **uma via**, nao round-trip.

    No ABIDES a latencia par-a-par vem de posicoes sorteadas sobre uma reta e e
    deterministica por par de agentes, entao mediana, minimo e maximo coincidem
    para a esmagadora maioria dos agentes; divergencia entre eles indica
    reordenacao na fila de mensagens do kernel.
    """
    limits = _load_exchange_events(log_dir, "LIMIT_ORDER")

    placed = pd.to_datetime(limits["time_placed"])
    # Diferenca em ns convertida para ms de uma vez, sem passar por objetos
    # Timedelta do Python.
    latency_ms = (limits.index.to_numpy() - placed.to_numpy()) / np.timedelta64(1, "ms")

    frame = pd.DataFrame(
        {"agent_id": limits["agent_id"].to_numpy(), "latencia_ms": latency_ms}
    )
    agg = frame.groupby("agent_id")["latencia_ms"].agg(["count", "median", "min", "max"])
    agg.columns = ["n", "mediana_ms", "min_ms", "max_ms"]
    agg = agg.reset_index()
    return agg.sort_values("mediana_ms").reset_index(drop=True)


def estimate_k(
    log_dir: str, max_dist_cents: float = 20.0, bin_width: float = 1.0
) -> Dict[str, float]:
    """Estima o ``k`` de Avellaneda-Stoikov pela decaida da intensidade de execucao.

    Parameters
    ----------
    log_dir : str
        Diretorio do log da simulacao (precisa ter livro e log da bolsa).
    max_dist_cents : float, optional
        Distancia maxima ao mid, em centavos, considerada no ajuste. Default 20.
    bin_width : float, optional
        Largura do bin do histograma, em centavos. Default 1.

    Returns
    -------
    dict of str to float
        ``k_hat``, ``A_hat``, ``r_squared``, ``n_trades``, ``n_bins`` e os
        percentis ``p50_cents``, ``p90_cents`` e ``p99_cents`` da distancia dos
        negocios ao mid.

    Notes
    -----
    Ajusta ``lambda(delta) = A * exp(-k * delta)`` por minimos quadrados sobre
    ``log(contagem) ~ delta``, usando o centro de cada bin como ``delta``.

    ``delta`` entra no ajuste em **dolares**, nao em centavos, para que ``k_hat``
    saia na mesma unidade do ``k`` usado na formula do spread otimo do agente
    (``(2/gamma) * ln(1 + gamma/k)``, com precos em dolares). Os percentis
    reportados, ao contrario, ficam em centavos por serem lidos contra o tick.

    O ajuste e grosseiro quando a distribuicao e muito concentrada: no RMSC03 o
    negocio mediano ocorre a meio centavo do mid e a cauda alimenta poucos bins.
    Use ``r_squared`` para julgar quanto peso dar a estimativa; bins vazios sao
    descartados porque ``log(0)`` nao e definido, o que enviesa o ajuste para
    cima na cauda.

    O mid usado e o do snapshot de 1s imediatamente anterior a cada negocio
    (``ffill``), entao negocios anteriores ao primeiro snapshot sao descartados.
    """
    book = load_orderbook_top(log_dir)
    executed = _load_exchange_events(log_dir, "ORDER_EXECUTED")

    trades = pd.DataFrame(
        {"fill_price": pd.to_numeric(executed["fill_price"], errors="coerce")},
        index=executed.index,
    ).dropna()
    trades = trades.sort_index()

    # reindex+ffill casa cada negocio com o ultimo mid observado; equivale a um
    # merge_asof backward, mas sem materializar a chave de juncao.
    trades["mid"] = book["mid"].reindex(trades.index, method="ffill").to_numpy()
    trades = trades.dropna(subset=["mid"])

    distance = (trades["fill_price"] - trades["mid"]).abs()

    edges = np.arange(0.0, max_dist_cents + bin_width, bin_width)
    counts, edges = np.histogram(distance.to_numpy(), bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0

    occupied = counts > 0
    n_bins = int(occupied.sum())
    if n_bins < 2:
        raise ValueError(
            "Apenas {} bin(s) ocupado(s) ate {} cents: sem graus de liberdade "
            "para ajustar k.".format(n_bins, max_dist_cents)
        )

    delta_dollars = centers[occupied] / 100.0
    log_counts = np.log(counts[occupied].astype("float64"))

    design = np.vstack([delta_dollars, np.ones_like(delta_dollars)]).T
    coefficients, _, _, _ = np.linalg.lstsq(design, log_counts, rcond=None)
    slope, intercept = float(coefficients[0]), float(coefficients[1])

    fitted = design.dot(coefficients)
    ss_res = float(((log_counts - fitted) ** 2).sum())
    ss_tot = float(((log_counts - log_counts.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    return {
        "k_hat": -slope,
        "A_hat": float(np.exp(intercept)),
        "r_squared": r_squared,
        "n_trades": float(len(trades)),
        "n_bins": float(n_bins),
        "p50_cents": float(distance.quantile(0.50)),
        "p90_cents": float(distance.quantile(0.90)),
        "p99_cents": float(distance.quantile(0.99)),
    }


def obi_displacement_scale(
    log_dir: str, eta: float, market_spread_cents: float = 1.0
) -> Dict[str, float]:
    """Dimensiona o deslocamento do preco de reserva causado pelo preditor OBI.

    Parameters
    ----------
    log_dir : str
        Diretorio do log da simulacao, contendo ``AVELLANEDA_STOIKOV_AGENT.bz2``.
    eta : float
        Coeficiente ``eta_obi`` usado na simulacao (``--eta-obi``, default 0.5
        no rmsc03_as).
    market_spread_cents : float, optional
        Spread de mercado de referencia, em centavos, para a razao de escala.
        Default 1.0, que e a mediana medida no RMSC03.

    Returns
    -------
    dict of str to float
        ``obi_abs_median``, ``obi_abs_p90``, ``obi_abs_max``,
        ``displacement_median_cents``, ``displacement_p90_cents``,
        ``displacement_max_cents``, ``agent_half_spread_cents``,
        ``ratio_displacement_half_spread``, ``ratio_displacement_market_spread``
        e ``n_quotes``.

    Notes
    -----
    ``eta`` tem unidade de DOLARES, nao de centavos, e o deslocamento precisa
    ser convertido antes de ser comparado ao tick. A cadeia no agente e:
    ``_update_quotes`` calcula ``mid = mid_cents / price_scale``, entregando o
    mid em dolares a ``_avellaneda_stoikov``; ali ``r_t = mid - term_1 +
    eta_obi * obi`` soma o termo do preditor a uma grandeza em dolares; e so
    em ``_quotes_to_cents`` o resultado volta para centavos, multiplicado por
    ``price_scale``. Portanto ``eta * |OBI|`` sai em dolares e exige o fator 100.

    Confundir as duas unidades subestima o deslocamento em duas ordens de
    grandeza e faz um preditor que desloca a cotacao vinte ticks parecer inerte.
    E precisamente o modo de falha que este trabalho documenta, e ele reapareceu
    na primeira versao desta propria funcao.

    As duas razoes respondem a pergunta pratica: o sinal chega a mover a cotacao
    o suficiente para mudar a fila em que o agente esta? Um deslocamento muito
    abaixo do tick e economicamente inerte, por mais significante que o OBI seja
    estatisticamente.

    Num log rodado **sem** ``--use-obi`` o agente registra ``obi=0`` em todas as
    cotacoes e todas as estatisticas saem zeradas; isso e log ausente de sinal,
    nao sinal nulo.
    """
    # parse_as_metrics escreve progresso em stdout; este modulo nao imprime,
    # entao a saida do parser e capturada e descartada.
    with contextlib.redirect_stdout(io.StringIO()):
        parsed = parse_as_metrics(load_agent_log(log_dir))

    obi_abs = parsed["obi"].abs()
    # eta esta em dolares; o fator 100 leva o deslocamento a centavos, que e a
    # unidade em que o tick, o spread do mercado e as cotacoes sao expressos.
    displacement = eta * obi_abs * 100.0
    half_spread = (parsed["ask"] - parsed["bid"]) / 2.0

    half_spread_median = float(half_spread.median())
    displacement_median = float(displacement.median())

    return {
        "n_quotes": float(len(parsed)),
        "obi_abs_median": float(obi_abs.median()),
        "obi_abs_p90": float(obi_abs.quantile(0.90)),
        "obi_abs_max": float(obi_abs.max()),
        "displacement_median_cents": displacement_median,
        "displacement_p90_cents": float(displacement.quantile(0.90)),
        "displacement_max_cents": float(displacement.max()),
        "agent_half_spread_cents": half_spread_median,
        "ratio_displacement_half_spread": (
            displacement_median / half_spread_median
            if half_spread_median > 0.0
            else float("nan")
        ),
        "ratio_displacement_market_spread": (
            displacement_median / market_spread_cents
            if market_spread_cents > 0.0
            else float("nan")
        ),
    }


def marginal_effects(
    df: pd.DataFrame,
    pairs: Sequence[Tuple[int, int, str]],
    value_col: str = "pnl_mean",
    err_col: str = "pnl_ci95",
    id_col: str = "test_id",
) -> pd.DataFrame:
    """Compara celulas do estudo de ablacao duas a duas.

    Parameters
    ----------
    df : pd.DataFrame
        CSV agregado da ablacao (``ablation_results.csv``), com uma linha por
        celula do desenho fatorial.
    pairs : sequence of tuple
        Trincas ``(id_base, id_tratamento, rotulo)``. O efeito e sempre medido
        do base para o tratamento.
    value_col : str, optional
        Coluna com a media da metrica. Default ``'pnl_mean'``.
    err_col : str, optional
        Coluna com o semi-intervalo de confianca de 95%. Default ``'pnl_ci95'``.
    id_col : str, optional
        Coluna identificadora da celula. Default ``'test_id'``.

    Returns
    -------
    pd.DataFrame
        Colunas ``rotulo``, ``id_base``, ``id_trat``, ``valor_base``,
        ``valor_trat``, ``diff``, ``ic95_combinado`` e ``veredito``. O veredito
        e ``'separado'`` quando ``|diff| > ic95_combinado`` e
        ``'nao separado'`` caso contrario.

    Raises
    ------
    KeyError
        Se alguma coluna pedida ou algum id de celula nao existir em ``df``.

    Notes
    -----
    O IC combinado e ``sqrt(err_base^2 + err_trat^2)``, o que **pressupoe
    independencia entre as celulas**. As celulas compartilham as mesmas
    sementes, entao seus erros sao positivamente correlacionados e a soma
    quadratica superestima a incerteza da diferenca: o teste e conservador e
    um veredito ``'separado'`` e mais forte do que parece, enquanto
    ``'nao separado'`` nao prova ausencia de efeito.

    A regra ``|diff| > IC`` e uma leitura de sobreposicao de barras de erro, nao
    um teste de hipotese formal com controle de erro tipo I sobre multiplas
    comparacoes.
    """
    for column in (id_col, value_col, err_col):
        if column not in df.columns:
            raise KeyError(
                "Coluna '{}' ausente no CSV; disponiveis: {}".format(
                    column, list(df.columns)
                )
            )

    indexed = df.set_index(id_col)

    base_ids = [pair[0] for pair in pairs]
    treat_ids = [pair[1] for pair in pairs]
    labels = [pair[2] for pair in pairs]

    missing = sorted(set(base_ids + treat_ids) - set(indexed.index))
    if missing:
        raise KeyError(
            "Celulas ausentes na coluna '{}': {}".format(id_col, missing)
        )

    base_value = indexed.loc[base_ids, value_col].to_numpy(dtype="float64")
    treat_value = indexed.loc[treat_ids, value_col].to_numpy(dtype="float64")
    base_err = indexed.loc[base_ids, err_col].to_numpy(dtype="float64")
    treat_err = indexed.loc[treat_ids, err_col].to_numpy(dtype="float64")

    diff = treat_value - base_value
    combined_ci = np.sqrt(base_err**2 + treat_err**2)
    verdict = np.where(np.abs(diff) > combined_ci, "separado", "nao separado")

    return pd.DataFrame(
        {
            "rotulo": labels,
            "id_base": base_ids,
            "id_trat": treat_ids,
            "valor_base": base_value,
            "valor_trat": treat_value,
            "diff": diff,
            "ic95_combinado": combined_ci,
            "veredito": verdict,
        }
    )
