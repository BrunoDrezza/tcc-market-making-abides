import numpy as np
import pandas as pd

from agent.TradingAgent import TradingAgent
from util.util import log_print


class AvellanedaStoikovAgent(TradingAgent):
    """
    Market Making agent based on the Avellaneda-Stoikov (2008) stochastic
    optimal control model.  Computes optimal bid/ask quotes as a function
    of inventory, volatility, and remaining horizon.
    """

    def __init__(self, id, name, type, symbol, starting_cash,
                 random_state=None, log_orders=False,
                 order_size=100, wake_up_freq='1s',
                 gamma=0.1, k=1.5, vol_window=60,
                 min_sigma=1e-4, max_inventory=5000,
                 horizon_end=None):

        super().__init__(id, name, type,
                         starting_cash=starting_cash,
                         log_orders=log_orders,
                         random_state=random_state)

        # --- Symbol & order parameters ---
        self.symbol = symbol
        self.order_size = order_size
        self.wake_up_freq = wake_up_freq

        # --- Avellaneda-Stoikov model parameters ---
        self.gamma = gamma          # risk-aversion coefficient
        self.k = k                  # order-arrival intensity parameter
        self.vol_window = vol_window  # rolling window (in ticks) for σ estimation
        self.min_sigma = min_sigma  # floor for estimated volatility
        self.max_inventory = max_inventory  # absolute inventory limit
        self.horizon_end = horizon_end      # pd.Timestamp for session end (T)

        # --- Internal state ---
        self.price_scale = 100.0          # dollars → cents multiplier
        self.tick_size = 1                # minimum price increment (cents)
        self.state = "AWAITING_WAKEUP"
        self.requote_delay = pd.Timedelta("100ns")
        self.mid_history = pd.Series(dtype="float64")
        self.last_quotes = {"bid": None, "ask": None}

    # ------------------------------------------------------------------
    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------
    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)
        if not can_trade:
            return

        if self.horizon_end is None:
            self.horizon_end = self.mkt_close

        if currentTime >= self.horizon_end:
            self.cancelAllOrders()
            return

        self.getCurrentSpread(self.symbol, depth=1)
        self.state = "AWAITING_SPREAD"

    # ------------------------------------------------------------------
    #  Message routing
    # ------------------------------------------------------------------
    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        mtype = msg.body["msg"]

        if mtype == "QUERY_SPREAD" and self.state == "AWAITING_SPREAD":
            self._update_quotes(currentTime)

        elif mtype in ("ORDER_EXECUTED", "ORDER_CANCELLED"):
            if currentTime < self.horizon_end:
                self.setWakeup(currentTime + self.requote_delay)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def cancelAllOrders(self):
        for order in list(self.orders.values()):
            self.cancelOrder(order)

    # ------------------------------------------------------------------
    #  Data preparation & estimators
    # ------------------------------------------------------------------
    def _compute_mid_cents(self):
        bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
        if bid is None or ask is None:
            return self.last_trade.get(self.symbol, None)
        return int(round((bid + ask) / 2))

    def _update_mid_history(self, currentTime, mid_cents):
        mid_dollars = mid_cents / self.price_scale
        self.mid_history = pd.concat(
            [self.mid_history, pd.Series([mid_dollars], index=[currentTime])]
        )
        max_len = 10 * self.vol_window
        if len(self.mid_history) > max_len:
            self.mid_history = self.mid_history.iloc[-max_len:]

    def _estimate_sigma2(self):
        series = self.mid_history.dropna()
        if len(series) < 5:
            return self.min_sigma ** 2
        log_returns = np.log(series).diff().dropna()
        sigma = log_returns.rolling(self.vol_window).std(ddof=1).iloc[-1]
        if np.isnan(sigma) or sigma < self.min_sigma:
            sigma = self.min_sigma
        return sigma * sigma

    def _remaining_horizon_fraction(self, currentTime):
        total = (self.horizon_end - self.mkt_open).total_seconds()
        remaining = (self.horizon_end - currentTime).total_seconds()
        if total <= 0:
            return 0.0
        tau = max(0.0, min(1.0, remaining / total))
        return tau

    # ------------------------------------------------------------------
    #  Avellaneda-Stoikov core math
    # ------------------------------------------------------------------
    def _avellaneda_stoikov(self, mid, q_t, sigma2, tau):
        gamma = max(self.gamma, 1e-12)
        k = max(self.k, 1e-12)

        # Reservation price
        r_t = mid - (q_t * gamma * sigma2 * tau)

        # Optimal half-spread (asymptotic expansion)
        delta = (gamma * sigma2 * tau / 2.0) + ((1.0 / gamma) * np.log1p(gamma / k))

        return r_t, max(delta, self.tick_size / self.price_scale)

    def _quotes_to_cents(self, r_t, delta, q_t):
        bid = int(np.floor((r_t - delta) * self.price_scale))
        ask = int(np.ceil((r_t + delta) * self.price_scale))

        # Sanity: ask must be strictly above bid
        if ask <= bid:
            ask = bid + self.tick_size

        # Inventory skew guardrails
        if q_t >= self.max_inventory:
            bid = min(bid, ask - self.tick_size)
        if q_t <= -self.max_inventory:
            ask = max(ask, bid + self.tick_size)

        return bid, ask

    # ------------------------------------------------------------------
    #  Quote update orchestrator
    # ------------------------------------------------------------------
    def _update_quotes(self, currentTime):
        mid_cents = self._compute_mid_cents()
        if mid_cents is None:
            self.state = "AWAITING_WAKEUP"
            self.setWakeup(currentTime + self.getWakeFrequency())
            return

        self._update_mid_history(currentTime, mid_cents)

        sigma2 = self._estimate_sigma2()
        q_t = self.getHoldings(self.symbol)
        tau = self._remaining_horizon_fraction(currentTime)

        mid = mid_cents / self.price_scale
        r_t, delta = self._avellaneda_stoikov(mid, q_t, sigma2, tau)
        bid_cents, ask_cents = self._quotes_to_cents(r_t, delta, q_t)

        self._reprice_quotes(bid_cents, ask_cents)
        self.state = "AWAITING_WAKEUP"
        self.setWakeup(currentTime + self.getWakeFrequency())
