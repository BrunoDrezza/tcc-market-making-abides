import numpy as np
import pandas as pd
import traceback

from agent.TradingAgent import TradingAgent
from util.util import log_print

class AvellanedaStoikovAgent(TradingAgent):
    def __init__(self, id, name, type, symbol, starting_cash,
                 random_state=None, log_orders=True,
                 order_size=100, wake_up_freq='1s',
                 gamma=0.1, k=1.5, vol_window=60,
                 min_sigma=1e-4, max_inventory=5000,
                 horizon_end=None,
                 mkt_open=None, mkt_close=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
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
        self.last_quotes = {"mid": None}

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        if self.mkt_open is not None:
            self.setWakeup(self.mkt_open)
        else:
            self.setWakeup(startTime + pd.Timedelta(self.wake_up_freq))

    def wakeup(self, currentTime):
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
        try:
            super().receiveMessage(currentTime, msg)
            if msg.body["msg"] == "QUERY_SPREAD":
                self._update_quotes(currentTime)
        except Exception as e:
            self.logEvent("CRASH_RECEIVE", str(e))

    def cancelAllOrders(self):
        for order in list(self.orders.values()):
            self.cancelOrder(order)

    def _compute_mid_cents(self):
        bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
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
        mid_dollars = mid_cents / self.price_scale
        self.mid_history = pd.concat([self.mid_history, pd.Series([mid_dollars], index=[currentTime])])
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
        if self.mkt_open is None or self.horizon_end is None:
            return 0.0
        total = (self.horizon_end - self.mkt_open).total_seconds()
        remaining = (self.horizon_end - currentTime).total_seconds()
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, remaining / total))

    def _avellaneda_stoikov(self, mid, q_t, sigma2, tau):
        gamma = max(self.gamma, 1e-12)
        k = max(self.k, 1e-12)
        r_t = mid - (q_t * gamma * sigma2 * tau)
        delta = (gamma * sigma2 * tau / 2.0) + ((1.0 / gamma) * np.log1p(gamma / k))
        return r_t, max(delta, self.tick_size / self.price_scale)

    def _quotes_to_cents(self, r_t, delta, q_t):
        bid = int(np.floor((r_t - delta) * self.price_scale))
        ask = int(np.ceil((r_t + delta) * self.price_scale))
        if ask <= bid:
            ask = bid + self.tick_size
        if q_t >= self.max_inventory:
            bid = min(bid, ask - self.tick_size)
        if q_t <= -self.max_inventory:
            ask = max(ask, bid + self.tick_size)
        return bid, ask

    def _update_quotes(self, currentTime):
        try:
            mid_cents = self._compute_mid_cents()
            self._update_mid_history(currentTime, mid_cents)

            sigma2 = self._estimate_sigma2()
            q_t = self.getHoldings(self.symbol)
            tau = self._remaining_horizon_fraction(currentTime)

            mid = mid_cents / self.price_scale
            r_t, delta = self._avellaneda_stoikov(mid, q_t, sigma2, tau)
            bid_cents, ask_cents = self._quotes_to_cents(r_t, delta, q_t)

            self._reprice_quotes(bid_cents, ask_cents)
            
            log_str = f"inv={q_t} mid={mid_cents} bid={bid_cents} ask={ask_cents}"
            self.logEvent("AS_QUOTE", log_str)
        except Exception as e:
            self.logEvent("CRASH_UPDATE", str(e))

    def _reprice_quotes(self, bid_cents, ask_cents):
        open_orders = list(self.orders.values())
        
        # Correção agressiva anti-crash nas ordens
        bid_orders = [o for o in open_orders if getattr(o, 'is_buy_order', False) and o.symbol == self.symbol]
        ask_orders = [o for o in open_orders if not getattr(o, 'is_buy_order', True) and o.symbol == self.symbol]

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