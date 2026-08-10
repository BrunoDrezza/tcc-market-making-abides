"""
run_diagnostics.py

CLI para os diagnosticos de microestrutura em ``analysis/diagnostics.py``.

Cada subcomando roda uma analise sobre logs ja gravados e imprime uma tabela
legivel. Nenhum subcomando dispara simulacao.

Examples
--------
::

    python run_diagnostics.py book     --log-dir log/CALIB_REF
    python run_diagnostics.py volume   --log-dir log/CALIB_REF
    python run_diagnostics.py latency  --log-dir log/CALIB_REF --top 10
    python run_diagnostics.py kfit     --log-dir log/CALIB_REF
    python run_diagnostics.py obi      --log-dir log/CALIB_REF --eta 0.5
    python run_diagnostics.py marginal --csv ablation_results.csv
"""

import argparse
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from analysis.diagnostics import (
    DEFAULT_ABLATION_PAIRS,
    DEFAULT_AGENT_RANGES,
    estimate_k,
    latency_by_agent,
    load_orderbook_top,
    marginal_effects,
    obi_displacement_scale,
    volume_share_by_agent,
)

_RULE_WIDTH = 78


def _header(title: str) -> None:
    """Imprime um cabecalho de secao."""
    print("")
    print("=" * _RULE_WIDTH)
    print("  {}".format(title))
    print("=" * _RULE_WIDTH)


def _print_mapping(mapping: Dict[str, float]) -> None:
    """Imprime um dict de metricas alinhado em duas colunas."""
    for key, value in mapping.items():
        if isinstance(value, float) and value != 0.0 and abs(value) < 1e-3:
            print("  {:<34s}: {:>16.6e}".format(key, value))
        else:
            print("  {:<34s}: {:>16.4f}".format(key, value))


def parse_agent_ranges(spec: str) -> Dict[str, Tuple[int, int]]:
    """Converte ``'nome:lo-hi,nome:lo-hi'`` no mapa de faixas de agent_id.

    Parameters
    ----------
    spec : str
        Especificacao textual das faixas, inclusivas em ambos os extremos.

    Returns
    -------
    dict of str to tuple of int
        Mapa pronto para :func:`analysis.diagnostics.volume_share_by_agent`.

    Raises
    ------
    argparse.ArgumentTypeError
        Se algum item nao seguir o formato ``nome:inicio-fim`` com inteiros.
    """
    ranges: Dict[str, Tuple[int, int]] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            name, bounds = item.split(":")
            low_text, high_text = bounds.split("-")
            ranges[name.strip()] = (int(low_text), int(high_text))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Faixa invalida '{}'; use nome:inicio-fim.".format(item)
            )
    if not ranges:
        raise argparse.ArgumentTypeError("Nenhuma faixa valida em '{}'.".format(spec))
    return ranges


def parse_pairs(spec: str) -> List[Tuple[int, int, str]]:
    """Converte ``'base:trat:rotulo,...'`` nas trincas de efeito marginal.

    Parameters
    ----------
    spec : str
        Especificacao textual dos pares a comparar.

    Returns
    -------
    list of tuple
        Trincas ``(id_base, id_tratamento, rotulo)``.

    Raises
    ------
    argparse.ArgumentTypeError
        Se algum item nao seguir o formato ``base:trat:rotulo`` com ids inteiros.
    """
    pairs: List[Tuple[int, int, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "Par invalido '{}'; use base:trat:rotulo.".format(item)
            )
        try:
            pairs.append((int(parts[0]), int(parts[1]), parts[2].strip()))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Ids nao inteiros em '{}'.".format(item)
            )
    if not pairs:
        raise argparse.ArgumentTypeError("Nenhum par valido em '{}'.".format(spec))
    return pairs


def cmd_book(args: argparse.Namespace) -> None:
    """Resume o topo do livro reconstruido do snapshot de 1s."""
    top = load_orderbook_top(args.log_dir)

    _header("Topo do livro — {}".format(args.log_dir))
    print("  snapshots validos              : {}".format(len(top)))
    print("  janela                         : {} -> {}".format(top.index[0], top.index[-1]))
    print("")
    print("  Spread (cents)")
    print("    mediana                      : {:.4f}".format(top["spread"].median()))
    print("    media                        : {:.4f}".format(top["spread"].mean()))
    print("    p90 / p99                    : {:.4f} / {:.4f}".format(
        top["spread"].quantile(0.90), top["spread"].quantile(0.99)
    ))
    print("    min / max                    : {:.4f} / {:.4f}".format(
        top["spread"].min(), top["spread"].max()
    ))
    print("")
    print("  Mid (cents)")
    print("    mediana                      : {:.2f}".format(top["mid"].median()))
    print("    min / max                    : {:.2f} / {:.2f}".format(
        top["mid"].min(), top["mid"].max()
    ))
    print("")
    print("  Primeiras e ultimas observacoes:")
    preview = pd.concat([top.head(3), top.tail(3)])
    print(preview.to_string())
    print("")


def cmd_volume(args: argparse.Namespace) -> None:
    """Imprime a participacao de volume executado por faixa de agente."""
    ranges = args.ranges if args.ranges else DEFAULT_AGENT_RANGES
    table = volume_share_by_agent(args.log_dir, ranges)

    _header("Participacao de volume por grupo — {}".format(args.log_dir))
    if not args.ranges:
        print("  (faixas default do rmsc03_as; use --ranges para sobrescrever)")
    print("")
    formatted = table.copy()
    formatted["acoes"] = formatted["acoes"].map("{:,.0f}".format)
    formatted["execucoes"] = formatted["execucoes"].map("{:,d}".format)
    formatted["pct_volume"] = formatted["pct_volume"].map("{:.3f}%".format)
    print(formatted.to_string(index=False))
    print("")
    # pct_volume tem o volume executado total como denominador, entao a soma da
    # coluna mede a cobertura das faixas — com override ela pode nao fechar 100%.
    print("  Volume coberto pelas faixas (dois lados): {:,.0f} acoes".format(
        table["acoes"].sum()
    ))
    print("  Cobertura do volume executado total     : {:.3f}%".format(
        table["pct_volume"].sum()
    ))
    print("")


def cmd_latency(args: argparse.Namespace) -> None:
    """Imprime a latencia de uma via por agente."""
    table = latency_by_agent(args.log_dir)

    _header("Latencia de uma via (agente -> bolsa) — {}".format(args.log_dir))
    print("  agentes com ordens limitadas   : {}".format(len(table)))
    print("  mediana entre agentes          : {:.4f} ms".format(table["mediana_ms"].median()))
    print("")

    if args.agents:
        selected = table.loc[table["agent_id"].isin(args.agents)]
        print("  Agentes selecionados:")
        print(selected.to_string(index=False))
    else:
        print("  {} agentes mais rapidos:".format(args.top))
        print(table.head(args.top).to_string(index=False))
        print("")
        print("  {} agentes mais lentos:".format(args.top))
        print(table.tail(args.top).to_string(index=False))
    print("")


def cmd_kfit(args: argparse.Namespace) -> None:
    """Imprime a estimativa de k do modelo de Avellaneda-Stoikov."""
    result = estimate_k(
        args.log_dir, max_dist_cents=args.max_dist, bin_width=args.bin_width
    )

    _header("Ajuste de k (A-S) — {}".format(args.log_dir))
    print("  bins de {:.2f} cent ate {:.2f} cents".format(args.bin_width, args.max_dist))
    print("")
    _print_mapping(result)
    print("")
    print("  k_hat esta em 1/dolar (delta convertido de cents para dolares).")
    if result["r_squared"] < 0.90:
        print("  ATENCAO: r_squared < 0.90 — distribuicao mal descrita por exponencial.")
    print("")


def cmd_obi(args: argparse.Namespace) -> None:
    """Imprime a escala do deslocamento induzido pelo preditor OBI."""
    result = obi_displacement_scale(
        args.log_dir, eta=args.eta, market_spread_cents=args.market_spread
    )

    _header("Escala do deslocamento por OBI — {}".format(args.log_dir))
    print("  eta_obi = {:.4f} | spread de mercado de referencia = {:.2f} cent".format(
        args.eta, args.market_spread
    ))
    print("")
    _print_mapping(result)
    print("")
    if result["obi_abs_max"] == 0.0:
        print("  ATENCAO: OBI identicamente zero — log rodado sem --use-obi.")
    print("")


def cmd_marginal(args: argparse.Namespace) -> None:
    """Imprime os efeitos marginais entre celulas da ablacao."""
    frame = pd.read_csv(args.csv)
    pairs: Sequence[Tuple[int, int, str]] = (
        args.pairs if args.pairs else list(DEFAULT_ABLATION_PAIRS)
    )
    table = marginal_effects(
        frame,
        pairs,
        value_col=args.value_col,
        err_col=args.err_col,
        id_col=args.id_col,
    )

    _header("Efeitos marginais — {} ({})".format(args.csv, args.value_col))
    if not args.pairs:
        print("  (pares default do desenho 2^3; use --pairs para sobrescrever)")
    print("")
    print(table.to_string(index=False))
    print("")
    print("  IC combinado = sqrt(err_base^2 + err_trat^2); assume independencia")
    print("  entre celulas, que compartilham sementes — logo e conservador.")
    print("")


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser de argumentos com todos os subcomandos.

    Returns
    -------
    argparse.ArgumentParser
        Parser configurado com os subcomandos ``book``, ``volume``,
        ``latency``, ``kfit``, ``obi`` e ``marginal``.
    """
    parser = argparse.ArgumentParser(
        description="Diagnosticos de microestrutura sobre logs do ABIDES."
    )
    subparsers = parser.add_subparsers(dest="command")

    book = subparsers.add_parser("book", help="Topo do livro e distribuicao do spread.")
    book.add_argument("--log-dir", required=True, help="Diretorio do log da simulacao.")
    book.set_defaults(func=cmd_book)

    volume = subparsers.add_parser("volume", help="Participacao de volume por grupo.")
    volume.add_argument("--log-dir", required=True, help="Diretorio do log da simulacao.")
    volume.add_argument(
        "--ranges",
        type=parse_agent_ranges,
        default=None,
        help="Override das faixas: 'nome:inicio-fim,nome:inicio-fim'.",
    )
    volume.set_defaults(func=cmd_volume)

    latency = subparsers.add_parser("latency", help="Latencia de uma via por agente.")
    latency.add_argument("--log-dir", required=True, help="Diretorio do log da simulacao.")
    latency.add_argument(
        "--agents",
        type=int,
        nargs="+",
        default=None,
        help="Lista de agent_id a exibir (default: extremos da distribuicao).",
    )
    latency.add_argument(
        "--top", type=int, default=5, help="Quantos agentes por extremo (default: 5)."
    )
    latency.set_defaults(func=cmd_latency)

    kfit = subparsers.add_parser("kfit", help="Estimativa de k do modelo A-S.")
    kfit.add_argument("--log-dir", required=True, help="Diretorio do log da simulacao.")
    kfit.add_argument(
        "--max-dist", type=float, default=20.0, help="Distancia maxima em cents (20)."
    )
    kfit.add_argument(
        "--bin-width", type=float, default=1.0, help="Largura do bin em cents (1)."
    )
    kfit.set_defaults(func=cmd_kfit)

    obi = subparsers.add_parser("obi", help="Escala do deslocamento por OBI.")
    obi.add_argument("--log-dir", required=True, help="Diretorio do log da simulacao.")
    obi.add_argument(
        "--eta", type=float, default=0.5, help="Coeficiente eta_obi da simulacao (0.5)."
    )
    obi.add_argument(
        "--market-spread",
        type=float,
        default=1.0,
        help="Spread de mercado de referencia em cents (1.0).",
    )
    obi.set_defaults(func=cmd_obi)

    marginal = subparsers.add_parser("marginal", help="Efeitos marginais da ablacao.")
    marginal.add_argument("--csv", required=True, help="CSV agregado da ablacao.")
    marginal.add_argument(
        "--pairs",
        type=parse_pairs,
        default=None,
        help="Override dos pares: 'base:trat:rotulo,...'.",
    )
    marginal.add_argument("--value-col", default="pnl_mean", help="Coluna da media.")
    marginal.add_argument("--err-col", default="pnl_ci95", help="Coluna do semi-IC 95%%.")
    marginal.add_argument("--id-col", default="test_id", help="Coluna identificadora.")
    marginal.set_defaults(func=cmd_marginal)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Ponto de entrada da CLI.

    Parameters
    ----------
    argv : sequence of str, optional
        Argumentos de linha de comando. Default ``sys.argv[1:]``.

    Returns
    -------
    int
        Codigo de saida: 0 em sucesso, 1 em erro de dado ou arquivo ausente,
        2 quando nenhum subcomando foi informado.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 2

    # Erros de dado (log ausente, coluna faltando, amostra insuficiente) sao
    # esperados no uso interativo e viram mensagem curta; qualquer outra
    # excecao sobe com traceback, que e o que se quer para um bug real.
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print("ERRO: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
