"""Curvas de patrimonio e estoque do agente Avellaneda-Stoikov.

O mapa de calor do grid search resume cada execucao a um numero. Estas figuras
mostram o caminho ate ele, que e onde a politica se revela:

**Figura A — curvas por parametrizacao.** Para um `gamma` fixo, uma curva por
valor de `k`, cada uma sendo a media entre sementes com banda de +/- 1,96 erro
padrao. Le-se ao mesmo tempo o acumulo de resultado (painel superior) e o
controle de inventario que o produziu (painel inferior). Um `k` que enriquece
acumulando estoque direcional e visualmente distinto de um que enriquece
capturando spread com estoque em torno de zero, ainda que ambos terminem no
mesmo PnL.

**Figura B — dispersao entre sementes.** Para uma unica celula `(gamma, k)`, uma
curva por semente, sem agregacao. Com 4 sementes, a banda da Figura A e larga o
bastante para que a media sozinha sugira mais precisao do que os dados
sustentam; esta figura mostra a variacao bruta entre realizacoes do mercado.

Uso:
    python plot_equity_curves.py --gamma 0.5 --cell-k 100
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.equity import (  # noqa: E402
    STARTING_CASH_CENTS,
    aggregate_seeds,
    align_series,
    load_equity_series,
)

#: Capital inicial em dolares, usado como linha de referencia do painel superior.
STARTING_CASH_USD: float = STARTING_CASH_CENTS / 100.0

#: Paleta das curvas. Cores distinguiveis tambem em impressao monocromatica pela
#: ordem de luminancia, ja que a versao final do TCC pode ser impressa em P&B.
PALETTE: Tuple[str, ...] = ("#1565C0", "#2E7D32", "#EF6C00", "#C62828", "#6A1B9A")


def build_log_dir(
    log_prefix: str, gamma: float, k: float, r_bar: Optional[float], seed: int
) -> str:
    """Monta o caminho do log de uma celula do grid.

    Parameters
    ----------
    log_prefix : str
        Prefixo dos diretorios, tipicamente ``log/TCC_Opt``.
    gamma, k : float
        Coordenadas da celula na grade de calibracao.
    r_bar : float or None
        Nivel de preco fundamental em centimos. Se informado, entra no nome como
        ``_rb<r_bar em notacao .0e>``; se ``None``, o infixo e omitido.
    seed : int
        Semente estocastica da execucao.

    Returns
    -------
    str
        Caminho do diretorio, no formato gravado por ``optimization_as.py``:
        ``log/TCC_Opt_g0.5_k100.0_rb1e+04_s20240101``.

    Notes
    -----
    O nome depende da representacao decimal de `gamma` e `k` como float
    (``0.5`` -> ``g0.5``, ``100.0`` -> ``k100.0``), porque o orquestrador
    interpola os valores da CLI diretamente. Passar ``100`` como inteiro
    produziria ``k100`` e nao casaria com nenhum diretorio existente — por isso
    os argumentos sao lidos como ``float``.
    """
    infix = "" if r_bar is None else f"_rb{r_bar:.0e}"
    return f"{log_prefix}_g{gamma}_k{k}{infix}_s{seed}"


def load_cell(
    log_prefix: str,
    gamma: float,
    k: float,
    r_bar: Optional[float],
    seeds: Sequence[int],
    freq: str,
) -> Dict[int, pd.DataFrame]:
    """Carrega e alinha as series de todas as sementes de uma celula.

    Parameters
    ----------
    log_prefix : str
        Prefixo dos diretorios de log.
    gamma, k : float
        Coordenadas da celula.
    r_bar : float or None
        Nivel de preco fundamental, em centimos.
    seeds : sequence of int
        Sementes a carregar.
    freq : str
        Frequencia da grade temporal comum.

    Returns
    -------
    dict of int to pd.DataFrame
        Series alinhadas, indexadas pela semente. Sementes cujo log esta ausente
        ou ilegivel sao omitidas, com aviso em ``stdout``.

    Notes
    -----
    Uma execucao faltante nao interrompe a figura: o grid e caro (~45 min por
    celula-semente) e e comum inspecionar resultados parciais enquanto o
    restante ainda roda.
    """
    raw: Dict[str, pd.DataFrame] = {}
    present: List[int] = []

    for seed in seeds:
        log_dir = build_log_dir(log_prefix, gamma, k, r_bar, seed)
        if not os.path.isdir(log_dir):
            print(f"  log ausente, ignorado: {log_dir}")
            continue
        try:
            raw[str(seed)] = load_equity_series(log_dir)
        except (FileNotFoundError, ValueError, KeyError) as err:
            print(f"  log ilegivel, ignorado: {log_dir} ({err.__class__.__name__}: {err})")
            continue
        present.append(seed)

    aligned = align_series(raw, freq=freq)
    return {seed: aligned[str(seed)] for seed in present}


#: Rotulos por idioma. O paper do ICAIF e em ingles e o TCC em portugues; a
#: figura e a mesma, logo o texto tem de sair do gerador e nao da edicao manual.
LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "equity": "Equity (USD)", "inv": "Inventory (shares)",
        "xlabel": "Simulated session time", "cash": "Starting cash (US$ {v})",
        "price": "asset at US$ {v}, ",
        "title_param": "Equity and inventory over the session ($\\gamma$ = {g})",
        "sub_param": "{p}calm market (no institutional aggressor), "
                     "mean of {n} seeds, ±1.96 SE band",
        "title_disp": "Dispersion across seeds ($\\gamma$ = {g}, $k$ = {k})",
        "sub_disp": "{p}calm market (no institutional aggressor), "
                    "{n} market realisations, no aggregation",
        "k": "$k$ = {k}", "seed": "seed {s}",
    },
    "pt": {
        "equity": "Patrimônio (USD)", "inv": "Estoque (ações)",
        "xlabel": "Hora do pregão simulado", "cash": "Capital inicial (US$ {v})",
        "price": "ativo a US$ {v}, ",
        "title_param": "Patrimônio e estoque ao longo do pregão ($\\gamma$ = {g})",
        "sub_param": "{p}mercado calmo (sem agressor institucional) — "
                     "média de {n} sementes, banda de ±1,96 EP",
        "title_disp": "Dispersão entre sementes ($\\gamma$ = {g}, $k$ = {k})",
        "sub_disp": "{p}mercado calmo (sem agressor institucional) — "
                    "{n} realizações do mercado, sem agregação",
        "k": "$k$ = {k}", "seed": "semente {s}",
    },
}


def _num(value: float, decimals: int, lang: str) -> str:
    """Formata um numero na convencao de milhar e decimal do idioma.

    Parameters
    ----------
    value : float
        Valor a formatar.
    decimals : int
        Casas decimais.
    lang : {'en', 'pt'}
        Idioma: ``'pt'`` usa ponto no milhar e virgula no decimal; ``'en'``
        mantem a convencao inversa, ja nativa do format spec.

    Returns
    -------
    str
        O numero formatado, e.g. ``100.192,03`` em pt e ``100,192.03`` em en.
    """
    out = f"{value:,.{decimals}f}"
    if lang == "en":
        return out
    # Troca via marcador intermediario porque a substituicao direta de ',' por
    # '.' colidiria com o separador decimal ja presente.
    return out.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _setup_panels(
    title: str, subtitle: str, lab: Dict[str, str]
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Cria a figura de dois paineis empilhados com eixo x compartilhado.

    Parameters
    ----------
    title : str
        Titulo principal, em negrito.
    subtitle : str
        Linha de contexto do cenario (nivel de preco, ecologia, agregacao).

    Returns
    -------
    tuple
        A figura, o eixo do patrimonio e o eixo do estoque.

    Notes
    -----
    Nada e desenhado aqui: linhas de referencia e formatacao do eixo temporal
    ficam em :func:`_finalise_panels`, que roda **depois** das curvas. Ver a
    justificativa la — desenhar antes dos dados corrompe os limites do eixo x.

    Cuidado com o par ``$...$`` do mathtext em `title` e `subtitle`: o cifrao de
    ``US$`` conta como delimitador e desemparelha a expressao, fazendo o
    matplotlib imprimir a marcacao literalmente. Em textos que citam valores em
    dolar, use caracteres Unicode (``±``, ``γ``) em vez de mathtext.
    """
    plt.rcParams.update({"font.size": 11, "font.family": "serif"})
    fig, (ax_eq, ax_inv) = plt.subplots(
        2, 1, figsize=(10, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0]},
    )

    ax_eq.set_ylabel(lab["equity"], fontweight="bold")
    ax_eq.grid(alpha=0.3)

    ax_inv.set_ylabel(lab["inv"], fontweight="bold")
    ax_inv.set_xlabel(lab["xlabel"], fontweight="bold")
    ax_inv.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)
    ax_eq.set_title(subtitle, fontsize=10, pad=8)
    return fig, ax_eq, ax_inv


def _finalise_panels(
    ax_eq: plt.Axes, ax_inv: plt.Axes, start: pd.Timestamp, end: pd.Timestamp,
    lab: Dict[str, str], lang: str,
) -> None:
    """Aplica linhas de referencia, limites e formato de hora, apos as curvas.

    Parameters
    ----------
    ax_eq, ax_inv : plt.Axes
        Paineis de patrimonio e de estoque, ja com as curvas desenhadas.
    start, end : pd.Timestamp
        Extremos da janela de pregao efetivamente coberta pelos dados.
    lab : dict of str to str
        Rotulos do idioma corrente, extraidos de :data:`LABELS`.
    lang : {'en', 'pt'}
        Idioma, usado na formatacao numerica da linha de capital inicial.

    Notes
    -----
    A ordem importa. Se `axhline` for chamado antes de qualquer dado temporal, o
    eixo x ainda nao tem unidade registrada; quando o conversor de datas do
    matplotlib entra em cena, ele **semeia** ``dataLim.intervalx`` com a janela
    padrao 2000-2010, que depois so e estendida pelos dados reais. O resultado e
    um eixo de 24 anos com a sessao de 2 h comprimida num unico pixel. Plotar
    primeiro, decorar depois, e fixar `set_xlim` explicitamente elimina o
    problema em qualquer ordem de chamada.

    Locator e formatter sao aplicados aos **dois** eixos: com ``sharex``, os
    objetos ``Ticker`` sao compartilhados, mas a flag ``isDefault_majloc`` e por
    eixo — configurar so um deles permite que o conversor de datas sobrescreva a
    configuracao ao registrar a unidade no outro.
    """
    # O capital inicial e a referencia que separa lucro de prejuizo; sem ela a
    # curva de patrimonio e ilegivel, porque a variacao (~1e2) e tres ordens de
    # grandeza menor que o nivel (1e5).
    ax_eq.axhline(
        STARTING_CASH_USD, color="black", lw=0.9, ls="--",
        label=lab["cash"].format(v=_num(STARTING_CASH_USD, 0, lang)),
    )
    ax_inv.axhline(0, color="black", lw=0.9)

    for ax in (ax_eq, ax_inv):
        ax.set_xlim(start, end)
        # byminute fixa as marcas nos quartos de hora; `interval=15` sozinho
        # ancora no inicio da serie e produz rotulos como 09:31, 09:46.
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=(0, 15, 30, 45)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


def _save(fig: plt.Figure, outfile: str) -> None:
    """Grava a figura no padrao editorial do trabalho e a fecha."""
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  gravado: {outfile}")


def plot_parametrisation(
    cells: Dict[float, Dict[int, pd.DataFrame]],
    gamma: float,
    r_bar: Optional[float],
    outfile: str,
    lang: str = "pt",
) -> None:
    """Figura A: curva media por valor de `k`, com banda de confianca.

    Parameters
    ----------
    cells : dict of float to (dict of int to pd.DataFrame)
        Series alinhadas, agrupadas por `k` e depois por semente.
    gamma : float
        Coeficiente de aversao ao risco comum a todas as curvas.
    r_bar : float or None
        Nivel de preco fundamental em centimos, usado no subtitulo.
    outfile : str
        Caminho do PNG de saida.
    lang : {'pt', 'en'}, optional
        Idioma dos rotulos.

    Notes
    -----
    A banda e o intervalo de confianca da media entre sementes. Curvas cujas
    bandas se sobrepoem em todo o pregao nao estao separadas pelos dados,
    qualquer que seja a distancia entre as medias no instante final.
    """
    lab = LABELS[lang]
    price = "" if r_bar is None else lab["price"].format(v=_num(r_bar / 100.0, 2, lang))
    n_seeds = max((len(s) for s in cells.values()), default=0)
    fig, ax_eq, ax_inv = _setup_panels(
        lab["title_param"].format(g=gamma),
        lab["sub_param"].format(p=price, n=n_seeds),
        lab,
    )

    span: List[pd.Timestamp] = []
    for i, k in enumerate(sorted(cells)):
        seeds = cells[k]
        if not seeds:
            continue
        color = PALETTE[i % len(PALETTE)]
        series = list(seeds.values())

        equity = aggregate_seeds(series, "equity")
        ax_eq.plot(equity.index, equity["mean"], color=color, lw=1.6,
                   label=lab["k"].format(k=f"{k:.0f}"))
        ax_eq.fill_between(equity.index, equity["lo"], equity["hi"], color=color, alpha=0.15)

        # O estoque oscila em alta frequencia e as quatro curvas se sobrepoem;
        # traco mais fino e banda mais clara evitam que o painel vire mancha.
        inventory = aggregate_seeds(series, "inv")
        ax_inv.plot(inventory.index, inventory["mean"], color=color, lw=1.0,
                    label=lab["k"].format(k=f"{k:.0f}"))
        ax_inv.fill_between(
            inventory.index, inventory["lo"], inventory["hi"], color=color, alpha=0.10
        )
        span.extend([equity.index[0], equity.index[-1]])

    _finalise_panels(ax_eq, ax_inv, min(span), max(span), lab, lang)
    ax_eq.legend(fontsize=9, loc="upper left", ncol=2)
    _save(fig, outfile)


def plot_seed_dispersion(
    seeds: Dict[int, pd.DataFrame],
    gamma: float,
    k: float,
    r_bar: Optional[float],
    outfile: str,
    lang: str = "pt",
) -> None:
    """Figura B: uma curva por semente, sem agregacao.

    Parameters
    ----------
    seeds : dict of int to pd.DataFrame
        Series alinhadas da celula, indexadas pela semente.
    gamma, k : float
        Coordenadas da celula.
    r_bar : float or None
        Nivel de preco fundamental em centimos, usado no subtitulo.
    outfile : str
        Caminho do PNG de saida.
    lang : {'pt', 'en'}, optional
        Idioma dos rotulos.
    """
    lab = LABELS[lang]
    price = "" if r_bar is None else lab["price"].format(v=_num(r_bar / 100.0, 2, lang))
    fig, ax_eq, ax_inv = _setup_panels(
        lab["title_disp"].format(g=gamma, k=f"{k:.0f}"),
        lab["sub_disp"].format(p=price, n=len(seeds)),
        lab,
    )

    span: List[pd.Timestamp] = []
    for i, seed in enumerate(sorted(seeds)):
        frame = seeds[seed]
        color = PALETTE[i % len(PALETTE)]
        ax_eq.plot(frame.index, frame["equity"], color=color, lw=1.3,
                   label=lab["seed"].format(s=seed))
        ax_inv.plot(frame.index, frame["inv"], color=color, lw=1.3,
                    label=lab["seed"].format(s=seed))
        span.extend([frame.index[0], frame.index[-1]])

    _finalise_panels(ax_eq, ax_inv, min(span), max(span), lab, lang)
    ax_eq.legend(fontsize=9, loc="upper left", ncol=2)
    _save(fig, outfile)


def _report_cell(k: float, seeds: Dict[int, pd.DataFrame]) -> None:
    """Imprime o estado terminal de uma celula, para conferencia com o CSV."""
    if not seeds:
        print(f"  k={k:<6.0f} sem execucoes disponiveis")
        return
    finals = [float(f["equity"].iloc[-1]) for f in seeds.values()]
    invs = [float(f["inv"].iloc[-1]) for f in seeds.values()]
    mean_equity = sum(finals) / len(finals)
    print(
        f"  k={k:<6.0f} n={len(finals)}  patrimonio final medio="
        f"{mean_equity:>11,.2f} USD  (PnL {mean_equity - STARTING_CASH_USD:+8.2f})  "
        f"estoque final medio={sum(invs) / len(invs):+8.1f}"
    )


def main() -> None:
    """Carrega as celulas pedidas e emite as duas figuras."""
    parser = argparse.ArgumentParser(
        description="Curvas de patrimonio e estoque do agente A-S"
    )
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--ks", type=float, nargs="+", default=[20.0, 50.0, 100.0, 200.0])
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[20240101, 20240102, 20240103, 20240104],
    )
    parser.add_argument(
        "--r-bar", type=float, default=1e4,
        help="Nivel de preco fundamental em centimos. optimization_as.py grava o "
             "infixo _rb<r_bar> no nome do diretorio; informe o mesmo valor do "
             "grid. Use --r-bar 0 para omitir o infixo (logs antigos).",
    )
    parser.add_argument("--log-prefix", type=str, default="log/TCC_Opt")
    parser.add_argument(
        "--cell-k", type=float, default=100.0,
        help="Valor de k da celula usada na figura de dispersao entre sementes.",
    )
    parser.add_argument("--outdir", type=str, default="../tcc-latex/figuras")
    parser.add_argument("--freq", type=str, default="10S")
    parser.add_argument(
        "--lang", type=str, default="pt", choices=("en", "pt"),
        help="Idioma dos rotulos. O paper do ICAIF usa en; o TCC, pt.",
    )
    args = parser.parse_args()

    r_bar: Optional[float] = None if args.r_bar == 0 else args.r_bar

    print(f"Curvas de patrimonio: gamma={args.gamma}, k={args.ks}, "
          f"sementes={args.seeds}, grade={args.freq}")

    cells: Dict[float, Dict[int, pd.DataFrame]] = {}
    for k in args.ks:
        cells[k] = load_cell(args.log_prefix, args.gamma, k, r_bar, args.seeds, args.freq)
        _report_cell(k, cells[k])

    if not any(cells.values()):
        raise SystemExit(
            "Nenhuma execucao encontrada. Verifique --log-prefix, --r-bar e --seeds."
        )

    tag = f"g{args.gamma}"
    plot_parametrisation(
        cells, args.gamma, r_bar,
        os.path.join(args.outdir, f"equity_curves_{tag}.png"), args.lang,
    )

    cell = cells.get(args.cell_k)
    if cell is None:
        cell = load_cell(
            args.log_prefix, args.gamma, args.cell_k, r_bar, args.seeds, args.freq
        )
    if not cell:
        print(f"  celula (gamma={args.gamma}, k={args.cell_k}) sem execucoes; "
              "figura de dispersao nao gerada.")
        return

    plot_seed_dispersion(
        cell, args.gamma, args.cell_k, r_bar,
        os.path.join(args.outdir, f"equity_dispersion_{tag}_k{args.cell_k:.0f}.png"),
        args.lang,
    )


if __name__ == "__main__":
    main()
