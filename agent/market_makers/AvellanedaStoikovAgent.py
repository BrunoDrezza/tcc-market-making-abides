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
            # (A-S math will be injected here in a later step)
            self.state = "AWAITING_WAKEUP"
            self.setWakeup(currentTime + self.getWakeFrequency())

        elif mtype in ("ORDER_EXECUTED", "ORDER_CANCELLED"):
            if currentTime < self.horizon_end:
                self.setWakeup(currentTime + self.requote_delay)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def cancelAllOrders(self):
        for order in list(self.orders.values()):
            self.cancelOrder(order)
