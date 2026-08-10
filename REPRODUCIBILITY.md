# Reprodução dos resultados

Este documento é o caminho mínimo entre um clone limpo do repositório e as tabelas
e figuras do trabalho. A sequência inteira está codificada em
`reproduce_experiments.py`, que por padrão **não executa nada**: imprime os comandos
exatos, com os parâmetros resolvidos, e a justificativa de cada passo.

```bash
python reproduce_experiments.py            # inspeção (default)
python reproduce_experiments.py --execute  # execução completa (~5 h)
```

---

## 1. Ambiente

O ABIDES é código de 2019. Os pins originais (`numpy==1.16.3`, `pandas==0.25.1`) não
compilam em Python 3.9 — `numpy.distutils` daquela versão quebra contra `setuptools`
moderno e não há wheel `cp39`. `requirements.txt` é a relaxação mínima com wheels
disponíveis, verificada contra as APIs que o ABIDES realmente usa.

```bash
uv venv --python 3.9 ABIDES_Enviroment
uv pip install --python ABIDES_Enviroment/bin/python -r requirements.txt
```

Todos os comandos abaixo assumem o venv **ativado**, ou o interpretador explícito
`./ABIDES_Enviroment/bin/python`.

> Os orquestradores (`optimization_as.py`, `run_ablation_study.py`, `rodar_pov.py`)
> disparam subprocessos com `sys.executable`, isto é, **herdam o interpretador
> ativo**. Rodá-los com o `python3` do sistema faz todas as células falharem em
> silêncio e o CSV sair com `Status=CRASHED`.

Uma exceção deliberada: `tests/verify_sigma.py` e o modo de inspeção de
`reproduce_experiments.py` rodam com **qualquer** Python 3, sem NumPy nem Pandas.

---

## 2. Sequência de experimentos

| # | Comando | Produz | Custo |
|---|---|---|---|
| 1 | `python abides.py -c rmsc03_as -t ABM -d 20240101 -s 20240101 -l CALIB_REF --end-time 11:30:00 --gamma 0.5 --k 20 --mm-count 2 --num-noise 5000 --fund-vol 1e-8 --r-bar 1e5 --as-colocate` | `log/CALIB_REF/` | ~12 min |
| 2 | `python run_analysis.py --log_dir log/CALIB_REF`<br>`python -m analysis.positioning --log-dir log/CALIB_REF` | **σ²_p99** para o passo 7, e o diagnóstico de posicionamento | < 1 min |
| 3 | `python probe_ecology.py --preset mm`<br>`python probe_ecology.py --preset liquidity`<br>`python probe_ecology.py --preset rbar` | `probe_ecology_results.csv`, `probe_ecology_liquidity.csv`, `probe_ecology_rbar.csv` | ~2 h |
| 4 | `python optimization_as.py --gammas 0.5 1.0 5.0 --ks 20 50 100 200 --seeds 4 --workers 3 --base-seed 20240101 --mm-count 2 --r-bar 1e5 --as-colocate --lean-logs --out grid_infeasible_region.csv --raw-out grid_infeasible_region_raw.csv` | grade no regime de US$ 1.000 | ~3,2 h |
| 5 | `python optimization_as.py --gammas 0.5 1.0 5.0 --ks 20 50 100 200 --seeds 4 --workers 3 --base-seed 20240101 --mm-count 2 --r-bar 1e4 --as-colocate --lean-logs --out grid_rbar1e4.csv --raw-out grid_rbar1e4_raw.csv` | `grid_rbar1e4.csv`, `grid_rbar1e4_raw.csv` | ~3,2 h |
| 6 | `python sweep_eta.py --r-bar 1e4 --out eta_sweep_rb1e4.csv --raw-out eta_sweep_rb1e4_raw.csv` | `eta_sweep_rb1e4.csv` → **η calibrado ao venue** | ~1,5 h |
| 7 | `python run_ablation_study.py --gamma 0.5 --k 100 --kill-sigma2 σ²_p99 --pov 0.25 --eta-obi 0.02 --seeds 20 --workers 3 --base-seed 20240101 --mm-count 2 --num-noise 5000 --fund-vol 1e-8 --r-bar 1e4 --end-time 11:30:00 --as-colocate --lean-logs --out ablation_rb1e4_eta002.csv --raw-out ablation_rb1e4_eta002_raw.csv` | `ablation_rb1e4_eta002.csv` (agregado), `_raw.csv` (por semente), `.manifest.json` | ~8 h |
| 8 | `python plot_decomposition_regimes.py --gamma 0.5 --seed 20240101`<br>`python plot_equity_curves.py --gamma 0.5 --cell-k 100 --lang en`<br>`python make_latex_table.py ablation --csv ablation_rb1e4_eta002.csv` | figuras e tabelas do artigo | ~2 min |

Opcionais, registrados nos CSV mas fora do caminho crítico:

| Comando | Produz |
|---|---|
| `python optimization_as.py --gammas 0.1 0.01 --ks 100 --seeds 4 --r-bar 1e4 ... --out m2_gamma_zero.csv` | `m2_gamma_zero.csv` — ablação do termo de estoque por γ→0 |
| `python plot_decomposition.py --gamma 0.5 --ks 20 50 100 200 --seed 20240101 --horizon H` | `decomp_agg.csv` — decomposição por horizonte |

### Por que a ordem é essa

Dois valores atravessam a sequência e **precisam ser medidos, não arbitrados**:

- **`--kill-sigma2` sai do passo 2, não de uma constante.** A escala de σ² depende de
  `r_bar`, de `fund_vol` e da densidade de agentes; não existe valor universal. Um
  limiar acima da cauda empírica nunca dispara, e as quatro células com `KILL=True`
  viram duplicatas exatas das células com `KILL=False` — a ablação mede oito cenários
  e reporta quatro. `run_ablation_study.py` torna o argumento obrigatório justamente
  para proibir a constante arbitrária.
- **`(γ*, k*)` sai do passo 4, por Sharpe médio entre sementes.** PnL de uma única
  realização premia a célula que teve sorte na trajetória do fundamental.

E a presença do estressor é o que separa os passos 3 e 5:

- **O grid roda SEM o agente institucional** (`-e -p` ausentes). Calibrar sob choque
  confundiria o efeito próprio dos parâmetros de controlo ótimo com a reação ao
  estressor, e a ablação não teria mais o que atribuir às camadas.
- **A ablação roda COM o agente institucional** (`--pov 0.25`). As três camadas são
  mecanismos de defesa: em calmaria mediríamos oito variantes de um agente que nunca
  precisou se defender.

O passo 1 é o único que roda **sem** `--lean-logs`, porque a flag zera `book_freq` no
config e suprime o `ORDERBOOK_*.bz2` — insumo exclusivo do diagnóstico de
posicionamento. Os passos 3 e 5 usam `--lean-logs`: o pipeline de risco-retorno lê
apenas o log do agente, e manter o livro em três processos simultâneos estoura a
memória.

---

## 3. Custo de tempo

Medido numa máquina de 4 núcleos, com 3 workers:

- **~12 min de relógio por simulação** de 2 h de pregão (09:30–11:30, 5000+ agentes,
  ~1,2 M mensagens). Com 3 simulações concorrentes, a vazão é de ~4 min por simulação.
- Grid de 12 células × 4 sementes = 48 simulações → **~3,2 h**.
- Ablação de 8 cenários × 4 sementes = 32 simulações → **~2,2 h**.
- Sequência completa (81 simulações) → **~5,4 h**.

O tempo cresce de forma superlinear com a duração da sessão: 5 min de pregão levam
~30 s. Encurtar `--end-time` acelera muito, mas comprime a janela do agente POV, que
opera de `mkt_open+30min` a `mkt_close-30min`; abaixo de 1 h de sessão o estressor
deixa de operar e a ablação perde sentido.

Os dois orquestradores gravam checkpoint por linha no CSV `_raw` e retomam de onde
pararam. Interromper e reexecutar o mesmo comando não repete trabalho concluído.

---

## 4. Convenções que não são óbvias

### 4.1 σ² está em unidade de preço, sobre primeiras diferenças

O modelo de Avellaneda-Stoikov pressupõe movimento Browniano **aritmético**,
`dS = σ dW`, no qual σ carrega unidade de preço por raiz de tempo. O estimador
consistente é o desvio padrão das **primeiras diferenças** do mid, **não** dos
log-retornos.

A versão original usava log-retornos. Para um ativo cotado a USD 1.000 isso encolhe a
estimativa por um fator de mil, e o piso `min_sigma = 1e-3` passa a dominar em 100%
dos despertares e em toda a faixa de volatilidade praticável. Com σ² travado em
1e-6 USD², o deslocamento de inventário `q·γ·σ²·τ` cai para 0,005 centésimo com
q = 100 — cerca de 1/200 do tick. O agente degenera num cotador de spread fixo, sem
qualquer resposta ao inventário, e as três camadas de defesa passam a operar sobre um
controle que nunca existiu.

A demonstração é autocontida e roda sem o venv:

```bash
python3 tests/verify_sigma.py
```

Qualquer alteração em `_estimate_sigma2` precisa preservar essa unidade.

### 4.2 τ é normalizado em [0, 1], não é `T − t` em segundos

`_remaining_horizon_fraction` devolve `(horizon_end − t) / (horizon_end − mkt_open)`,
uma fração que decresce de 1,0 a 0,0 ao longo da sessão. Isso muda a escala de γ em
relação ao artigo original: γ absorve a duração do pregão. Comparar o γ calibrado aqui
com o de um trabalho que use `T − t` em segundos exige reescalar pelo horizonte.

### 4.3 O livro de ofertas marca o lado pelo **sinal do volume**

`ORDERBOOK_<sym>_FREQ_1S.bz2` é uma matriz larga esparsa: linhas são instantes,
colunas são **níveis de preço em centavos**, valores são **volumes com sinal**.

- volume **negativo** ⇒ nível do lado **bid**;
- volume **positivo** ⇒ nível do lado **ask**;
- zero ⇒ nível vazio.

Logo `best_bid` é o **maior** preço com volume negativo e `best_ask` é o **menor**
preço com volume positivo. Não existe coluna de lado: o sinal é a única marcação, e
invertê-lo troca as duas pontas do livro sem produzir nenhum erro visível a jusante.
`analysis/positioning.py` implementa e documenta essa convenção.

### 4.4 `-s` precisa ser passado explicitamente

Sem `-s`, `config/rmsc03_as.py` deriva a semente do timestamp de execução. Cada
cenário roda então sobre uma realização diferente do oráculo e qualquer comparação
entre células passa a comparar ruído com ruído. É exatamente a diferença entre
`ablation_rb1e4_eta002.csv` e `ablation_results_no_seed.csv`. Os dois orquestradores já
propagam `-s` por construção; execuções manuais não.

Pelo mesmo motivo, calibração e ablação precisam rodar na **mesma ecologia**
(`--mm-count`, `--num-noise`, `--fund-vol`, `--r-bar`, `--as-colocate`): aplicar
parâmetros ajustados num mercado a um mercado de spread distinto invalida a
comparação.

### 4.5 Ordens a mercado voltam com `order_id` diferente do submetido

A Exchange do ABIDES decompõe uma ordem a mercado ao percorrer o livro, e os eventos
`ORDER_EXECUTED` retornam com identificadores novos. A ordem original **nunca é
removida** de `self.orders`, de modo que não dá para inferir quantidade em voo a
partir dessa estrutura. `_liquidate_inventory` contorna isso com guarda por inventário
e purga manual, e qualquer contagem de execuções deve contar **eventos**, não casar
identificadores — é o que `analysis/positioning.py` faz.

Correlato: o kernel desperta o agente mais de uma vez por ciclo durante a janela de
liquidação, em instantes separados pelo `computation_delay` (10 µs). Rotinas que
emitam ordens nessa janela precisam ser idempotentes.

### 4.6 Duas armadilhas de propagação de flags

- **`run_ablation_study.py` não expõe `--r-bar`.** As simulações da ablação rodam
  sempre no default do config (1e5 centavos). Se o grid rodar com `--r-bar` diferente,
  γ e k serão calibrados num tick relativo e aplicados noutro.
  `reproduce_experiments.py` aborta em `--execute` nesse caso, salvo
  `--allow-rbar-mismatch`.
- **`plot_decomposition.py` monta o caminho como `<log-prefix>_g<γ>_k<k>_s<seed>`,**
  sem o infixo de `r_bar` que `optimization_as.py` insere
  (`..._k<k>_rb1e+05_s<seed>`). Enquanto os dois não forem reconciliados, crie
  ligações simbólicas antes do passo 6:

  ```bash
  for d in log/TCC_Opt_g*_rb1e+05_s*; do ln -sfn "$(basename $d)" "${d/_rb1e+05/}"; done
  ```

### 4.7 Unidades e telemetria

Todo preço e caixa da simulação está em **centavos** (`starting_cash = 10_000_000` =
US$ 100 k). A conversão para dólar acontece só em `analysis/parser.py`. O canal entre
a simulação e toda a camada de análise é uma única linha de log por cotação:

```
inv=<int> mid=<int> bid=<int> ask=<int> obi=<float> cash=<int> sigma2=<float>
```

extraída por regex nomeada em `analysis/parser.py`. Mudar a ordem, os nomes ou o
separador quebra simultaneamente `run_analysis.py`, `run_ablation_study.py`,
`optimization_as.py`, `analysis/positioning.py` e os scripts de plot.

---

## 5. Números de referência

Diagnóstico de posicionamento sobre `log/CALIB_REF` (γ = 0,5, k = 20, 2 MM
incumbentes, sem estressor), reproduzível com
`python -m analysis.positioning --log-dir log/CALIB_REF`:

| Métrica | Valor |
|---|---|
| spread do mercado (mediana) | 1,00 centavo |
| spread do agente (mediana) | 20,00 centavos |
| bid atrás do topo (mediana) | 9,00 centavos |
| ask atrás do topo (mediana) | 9,00 centavos |
| tempo com bid no topo | 0,014 % |
| tempo com ask no topo | 0,140 % |
| ordens submetidas | 5 854 |
| execuções | 64 |
| taxa de conversão | 1,09 % |

É o achado central: num venue cujo spread está fixado em um tick, o meio-spread ótimo
da HJB coloca o agente uma ordem de grandeza atrás do topo. Ele praticamente nunca
disputa a fila, e as poucas execuções que obtém vêm de varredura adversa do livro —
não de provisão de liquidez.
