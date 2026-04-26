import numpy as np
import pandas as pd
import math
from typing import Dict, Union
from agent.TradingAgent import TradingAgent


class AvellanedaStoikovAgent(TradingAgent):
    """
    Agente Formador de Mercado baseado no modelo de controle estocástico de Avellaneda & Stoikov (2008).

    Esta implementação estende o arcabouço original incorporando três camadas de defesa empírica
    para atuar em microestruturas de Limit Order Book (LOB) de alta frequência sob estresse direcional:
    1. Preditor de Curto Prazo (Order Book Imbalance - OBI).
    2. Gestão de Risco Transversal (Hedge Sintético no SPY).
    3. Desarme por Entropia (Kill Switch baseado em variância extrema).

    References:
        Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book.
        Quantitative Finance, 8(3), 217-224.
    """

    def __init__(
        self,
        id,
        name,
        type,
        symbol,
        starting_cash,
        random_state=None,
        log_orders=True,
        order_size=100,
        wake_up_freq="1s",
        gamma=0.1,
        k=1.5,
        vol_window=60,
        min_sigma=1e-4,
        max_inventory=5000,
        # horizon_end=None,
        mkt_open=None,
        mkt_close=None,
        use_ofi=True,
        use_hedge=False,
        use_kill_switch=False,
        eta_ofi=0.5,
    ):
        """
        Inicializa o agente Avellaneda-Stoikov com os parâmetros de controle ótimo e flags de ablação.

        Args:
            id (int): Identificador único do agente no kernel do ABIDES.
            name (str): Nome de registro do agente.
            type (str): Tipo de classe do agente (usado para métricas e logs).
            symbol (str): Ticker do ativo provido (ex: 'AAPL').
            starting_cash (int): Caixa inicial em centavos.
            random_state (np.random.RandomState, optional): Semente estocástica local.
            log_orders (bool, optional): Flag para registro de envio/cancelamento de ordens.
            order_size (int, optional): Tamanho do lote padrão por cotação (bid/ask).
            wake_up_freq (str, optional): Frequência do loop de decisão (ex: '1s', '10ms').
            gamma (float, optional): Coeficiente de aversão ao risco absoluto (CARA). Define a penalização do inventário.
            k (float, optional): Sensibilidade da taxa de execução de ordens em relação à distância do mid-price.
            vol_window (int, optional): Janela contínua para estimação da variância (sigma^2).
            min_sigma (float, optional): Piso de segurança para a volatilidade, evitando divisões por zero.
            max_inventory (int, optional): Limite de posição máxima (risk limit). Ao atingir, suspende ordens de agravamento.
            horizon_end (pd.Timestamp, optional): Tempo final (T) do pregão para o cálculo de (T-t).
            mkt_open (pd.Timestamp, optional): Horário de abertura da simulação.
            mkt_close (pd.Timestamp, optional): Horário de fechamento da simulação.

            # --- Flags do Estudo de Ablação ---
            use_obi (bool, optional): Se True, ativa a Camada 1: Desloca o Preço de Reserva (r_t) usando o Order Book Imbalance.
            use_hedge (bool, optional): Se True, ativa a Camada 2: Simula trava de risco no SPY caso o inventário cruze o limite crítico.
            use_kill_switch (bool, optional): Se True, ativa a Camada 3: Cancela todas as ordens se a variância cruzar o limite máximo.
            eta_obi (float, optional): Coeficiente de sensibilidade do sinal de drift (OBI).
        """
        super().__init__(
            id,
            name,
            type,
            starting_cash=starting_cash,
            log_orders=log_orders,
            random_state=random_state,
        )
        self.log_events = True
        self.symbol = symbol
        self.order_size = order_size
        self.wake_up_freq = wake_up_freq
        self.gamma = gamma
        self.k = k
        self.vol_window = vol_window
        self.min_sigma = min_sigma
        self.max_inventory = max_inventory
        self.mkt_open = mkt_open
        self.horizon_end = mkt_close
        self.price_scale = 100.0
        self.tick_size = 1
        self.mid_history = pd.Series(dtype="float64")
        # Onde estava self.last_quotes = {"mid": None}
        self.last_quotes: Dict[str, Union[int, None]] = {"mid": None}
        self.use_obi = use_ofi  # Renomeado
        self.use_hedge = use_hedge
        self.use_kill_switch = use_kill_switch
        self.eta_obi = eta_ofi  # Renomeado
        self.obi_proxy = 0.0  # Renomeado

        self.is_hedged = False  # Estado inicial do derivativo
        self.hedge_threshold = 150  # Inventário crítico antes da trava no SPY
        self.hedge_cost_cents = 2  # Custo financeiro simulado do spread (SPY)
        self.kill_switch_sigma2 = 50.0  # Tolerância máxima de variância (entropia)

    def getWakeFrequency(self):
        """
        Retorna a frequência de despertar do agente no formato de Timedelta do Pandas.

        Returns:
            pd.Timedelta: O intervalo de tempo entre cada ciclo de decisão do agente.
        """
        return pd.Timedelta(self.wake_up_freq)

    def kernelStarting(self, startTime):
        """
        Gatilho de inicialização disparado pelo Kernel do ABIDES antes do início do loop principal.

        Configura o primeiro despertar do agente. Se houver um horário de abertura de mercado
        definido (mkt_open), o agente agenda seu primeiro 'wakeup' para esse instante.

        Args:
            startTime (pd.Timestamp): O tempo global inicial da simulação.
        """
        super().kernelStarting(startTime)
        if self.mkt_open is not None:
            self.setWakeup(self.mkt_open)
        else:
            self.setWakeup(startTime + pd.Timedelta(self.wake_up_freq))

    def wakeup(self, currentTime):
        """
        Ponto de entrada do relógio de simulação (Kernel Event Loop).

        Desperta o agente de acordo com a frequência (wake_up_freq). Durante o horário regular de mercado,
        dispara a requisição de top-of-book (Spread) para a Exchange, o que iniciará o processo de precificação.

        Args:
            currentTime (pd.Timestamp): O instante de tempo atual fornecido pelo Global Virtual Time do ABIDES.

        Note:
            Possui proteção contra quebra estrutural no ambiente (try/except) para garantir que um erro de
            cálculo em um nanosegundo específico não trave a simulação inteira.
        """
        try:
            super().wakeup(currentTime)

            # Loga apenas na virada da hora para não poluir
            if currentTime.minute == 0 and currentTime.second == 0:
                self.logEvent("HEARTBEAT", f"Wakeup at {currentTime}")

            if self.mkt_open is not None and currentTime < self.mkt_open:
                self.setWakeup(currentTime + self.getWakeFrequency())
                return

            if self.horizon_end is not None and currentTime >= self.horizon_end:
                self.cancelAllOrders()
                return

            # Mercado Aberto: Solicita o spread e agenda o próximo segundo
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.getCurrentSpread(self.symbol, depth=1)

        except Exception as e:
            self.logEvent("CRASH_WAKEUP", str(e))

    def receiveMessage(self, currentTime, msg):
        """
        Processador de mensagens de rede do agente (Callback).

        Responde a eventos recebidos via rede simulada (com latência contabilizada). O principal gatilho
        ocorre quando a Exchange responde com os dados do 'QUERY_SPREAD', iniciando a recalibração de cotações.

        Args:
            currentTime (pd.Timestamp): O instante de chegada da mensagem após o atraso de rede (latency).
            msg (Message): Objeto de mensagem serializado pelo ABIDES.
        """
        try:
            super().receiveMessage(currentTime, msg)
            if msg.body["msg"] == "QUERY_SPREAD":
                self._update_quotes(currentTime)
        except Exception as e:
            self.logEvent("CRASH_RECEIVE", str(e))

    def cancelAllOrders(self):
        """
        Sub-rotina de segurança: Varre o dicionário de ordens ativas e envia mensagens de cancelamento
        para o Exchange Agent. Utilizada no fechamento de mercado e durante a ativação do Kill Switch.
        """
        for order in list(self.orders.values()):
            self.cancelOrder(order)

    def _compute_mid_cents(self):
        """
        Processa o estado atual do Limit Order Book e calcula o sinal de microestrutura (OBI).

        Camada 1 da Ablação: Se `use_obi` estiver ativo, calcula o Order Book Imbalance empírico,
        que captura a pressão direcional da liquidez no topo do livro.

        Math:
            OBI = (V_bid - V_ask) / (V_bid + V_ask)

        Returns:
            int: O preço médio (mid-price) do ativo em centavos.
        """
        res = self.getKnownBidAsk(self.symbol)

        # Prevenção de quebra caso o livro retorne apenas (bid, ask) ou (None, None)
        if len(res) == 4:
            bid, bid_vol, ask, ask_vol = res
        else:
            bid, ask = res[:2]
            bid_vol, ask_vol = 0, 0

        # --- LAYER 1: CÁLCULO DO OBI PROXY (Order Book Imbalance) ---
        if self.use_obi and bid_vol and ask_vol and (bid_vol + ask_vol) > 0:
            self.obi_proxy = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        else:
            self.obi_proxy = 0.0

        if bid is not None and ask is not None:
            mid = int(round((bid + ask) / 2))
            self.last_quotes["mid"] = mid
            return mid
        elif self.last_quotes.get("mid") is not None:
            return self.last_quotes["mid"]
        else:
            self.last_quotes["mid"] = 100000
            return 100000

    def _update_mid_history(self, currentTime, mid_cents):
        """
        Atualiza a série temporal em memória com o logaritmo dos preços para cálculo de volatilidade.

        Args:
            currentTime (pd.Timestamp): O instante de tempo atual.
            mid_cents (int): O mid-price atual em centavos.
        """
        mid_dollars = mid_cents / self.price_scale
        self.mid_history = pd.concat(
            [self.mid_history, pd.Series([mid_dollars], index=[currentTime])]
        )
        max_len = 10 * self.vol_window
        if len(self.mid_history) > max_len:
            self.mid_history = self.mid_history.iloc[-max_len:]

    def _estimate_sigma2(self):
        """
        Estima a variância instantânea (sigma^2) do ativo com base em retornos logarítmicos.

        Esta função foi otimizada para microestrutura usando arrays C-contíguos do NumPy
        para evitar o overhead de indexação do Pandas em simulações de alta frequência.

        Returns:
            float: A variância (sigma^2) estrita e limitada a um piso (min_sigma^2) para evitar
                   indeterminações matemáticas na equação HJB.
        """
        series = self.mid_history.dropna()
        if len(series) < 5:
            return self.min_sigma**2

        # 1. Extração direta para matriz C-contígua (zero overhead de índice do Pandas)
        vals = series.to_numpy(dtype=np.float64)

        # 2. Vetorização pura. A flag '# type: ignore' silencia a alucinação do Pylance
        # garantindo que não sacrificamos milissegundos por causa do linter.
        log_vals = np.log(vals)  # type: ignore

        # 3. Diferença vetorizada (equivalente a .diff().dropna() do Pandas, mas muito mais rápido)
        log_returns = np.diff(log_vals)

        # 4. MICRO-OTIMIZAÇÃO: Fatiamos apenas a última janela necessária em vez de
        # processar a série histórica inteira.
        window = log_returns[-self.vol_window :]

        if len(window) < 2:
            return self.min_sigma**2

        # Cálculo estrito em C
        sigma = float(np.std(window, ddof=1))

        import math

        if math.isnan(sigma) or sigma < self.min_sigma:
            sigma = self.min_sigma

        return sigma * sigma

    def _remaining_horizon_fraction(self, currentTime):
        """
        Calcula a fração de tempo remanescente até o final do pregão (tau).

        No modelo original, tau = T - t. Aqui, normalizamos a duração total para um espaço [0, 1].

        Args:
            currentTime (pd.Timestamp): Instante atual.

        Returns:
            float: Fração decrescente de tempo remanescente [1.0 -> 0.0].
        """
        if self.mkt_open is None or self.horizon_end is None:
            return 0.0
        total = (self.horizon_end - self.mkt_open).total_seconds()
        remaining = (self.horizon_end - currentTime).total_seconds()
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, remaining / total))

    def _avellaneda_stoikov(self, mid, q_t, sigma2, tau, obi):
        """
        Motor de Precificação Estocástica (Hamilton-Jacobi-Bellman).

        Calcula o preço de reserva (r_t) e o meio-spread ótimo (delta) integrando a
        solução clássica de Avellaneda & Stoikov com preditores informacionais de curtíssimo prazo.

        Math:
            r_t = S_t - (q_t * gamma * sigma^2 * tau) + (eta * OBI)
            delta = (gamma * sigma^2 * tau / 2) + (1 / gamma) * ln(1 + gamma/k)

        Args:
            mid (float): Mid-price atual do mercado.
            q_t (int): Inventário atual do agente.
            sigma2 (float): Variância estocástica atual.
            tau (float): Fração de tempo remanescente.
            obi (float): Order Book Imbalance (Sinal de Drift).

        Returns:
            tuple(float, float): Preço de reserva (r_t) e o spread ótimo contínuo (delta).
        """
        gamma = max(self.gamma, 1e-12)
        k = max(self.k, 1e-12)

        # --- MATEMÁTICA: R_t Clássico vs R_t com Preditor OBI ---
        if self.use_obi:
            r_t = mid - (q_t * gamma * sigma2 * tau) + (self.eta_obi * obi)
        else:
            r_t = mid - (q_t * gamma * sigma2 * tau)

        delta = (gamma * sigma2 * tau / 2.0) + ((1.0 / gamma) * math.log1p(gamma / k))
        return r_t, max(delta, self.tick_size / self.price_scale)

    def _quotes_to_cents(self, r_t, delta, q_t):
        """
        Discretiza as cotações teóricas do espaço contínuo para o grid de mercado (Tick Size).

        Força limites de risco: Se o agente atingir a restrição máxima de inventário (Guéant et al., 2013),
        as cotações do lado agravante são deslocadas (skewed) passivamente para evitar execução indesejada.

        Args:
            r_t (float): Preço de reserva.
            delta (float): Meio-spread ótimo.
            q_t (int): Inventário atual.

        Returns:
            tuple(int, int): Preços de Bid e Ask determinísticos em centavos.
        """
        bid = int(math.floor((r_t - delta) * self.price_scale))
        ask = int(math.ceil((r_t + delta) * self.price_scale))

        if ask <= bid:
            ask = bid + self.tick_size
        if q_t >= self.max_inventory:
            bid = min(bid, ask - self.tick_size)
        if q_t <= -self.max_inventory:
            ask = max(ask, bid + self.tick_size)

        return bid, ask

    def _update_quotes(self, currentTime):
        """
        Orquestrador Central da Estratégia de Market Making.

        Esta função condensa todo o pipeline de tomada de decisão. Ela invoca a extração de métricas,
        avalia a elegibilidade do ambiente (Kill Switch), aplica hedges transversais (Camada 2),
        gera os preços ótimos via HJB e comanda o roteador a enviar as ordens.

        Args:
            currentTime (pd.Timestamp): O instante global atual da tomada de decisão.
        """
        try:
            # Força o tipo int ou faz fallback direto caso algo bizarro aconteça
            mid_cents = self._compute_mid_cents()
            if mid_cents is None:
                mid_cents = 100000

            self._update_mid_history(currentTime, mid_cents)
            sigma2 = self._estimate_sigma2()

            # --- LAYER 3: KILL SWITCH (Entropia Extrema) ---
            if self.use_kill_switch and sigma2 > self.kill_switch_sigma2:
                self.cancelAllOrders()
                self.logEvent(
                    "KILL_SWITCH",
                    f"Volatilidade extrema: {sigma2:.6f}. Ordens suspensas.",
                )
                return

            q_t = self.getHoldings(self.symbol)

            # --- LAYER 2: HEDGE SINTÉTICO NO SPY ---
            if self.use_hedge:
                if abs(q_t) >= self.hedge_threshold and not self.is_hedged:
                    self.is_hedged = True
                    cost = abs(q_t) * self.hedge_cost_cents
                    self.cash -= cost
                    self.logEvent(
                        "HEDGE_ENTER",
                        f"Inventário={q_t}. SPY Hedge ON. Custo=-{cost}c.",
                    )
                elif abs(q_t) < self.hedge_threshold and self.is_hedged:
                    self.is_hedged = False
                    cost = self.hedge_threshold * self.hedge_cost_cents
                    self.cash -= cost
                    self.logEvent(
                        "HEDGE_EXIT",
                        f"Inventário={q_t}. SPY Hedge OFF. Custo=-{cost}c.",
                    )

            tau = self._remaining_horizon_fraction(currentTime)
            mid = mid_cents / self.price_scale

            # Usa o obi_proxy atualizado
            r_t, delta = self._avellaneda_stoikov(mid, q_t, sigma2, tau, self.obi_proxy)
            bid_cents, ask_cents = self._quotes_to_cents(r_t, delta, q_t)

            self._reprice_quotes(bid_cents, ask_cents)

            log_str = f"inv={q_t} mid={mid_cents} bid={bid_cents} ask={ask_cents} obi={self.obi_proxy:.2f}"
            self.logEvent("AS_QUOTE", log_str)

        except Exception as e:
            self.logEvent("CRASH_UPDATE", str(e))

    def _reprice_quotes(self, bid_cents, ask_cents):
        """
        Mecanismo de Reconciliação do Livro do Agente (Order Router).

        Varre as ordens abertas enviadas pelo agente e as compara com os novos limites ótimos
        (bid_cents, ask_cents). Cancela passivamente as ordens obsoletas e insere liquidez nos
        novos níveis calculados pelo controle estocástico, sujeitando-se às travas de inventário.

        Args:
            bid_cents (int): Novo limite desejado de compra (Bid).
            ask_cents (int): Novo limite desejado de venda (Ask).
        """
        open_orders = list(self.orders.values())

        # Correção agressiva anti-crash nas ordens
        bid_orders = [
            o
            for o in open_orders
            if getattr(o, "is_buy_order", False) and o.symbol == self.symbol
        ]
        ask_orders = [
            o
            for o in open_orders
            if not getattr(o, "is_buy_order", True) and o.symbol == self.symbol
        ]

        for o in bid_orders:
            if o.limit_price != bid_cents:
                self.cancelOrder(o)

        for o in ask_orders:
            if o.limit_price != ask_cents:
                self.cancelOrder(o)

        has_bid = any(o.limit_price == bid_cents for o in bid_orders)
        has_ask = any(o.limit_price == ask_cents for o in ask_orders)

        inv = self.getHoldings(self.symbol)

        if not has_bid and inv < self.max_inventory:
            self.placeLimitOrder(self.symbol, self.order_size, True, bid_cents)

        if not has_ask and inv > -self.max_inventory:
            self.placeLimitOrder(self.symbol, self.order_size, False, ask_cents)