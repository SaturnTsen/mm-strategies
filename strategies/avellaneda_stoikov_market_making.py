from __future__ import annotations

from collections import deque
from decimal import Decimal
from math import ceil
from math import floor
from math import log
from math import sqrt
from math import tanh

from nautilus_trader.config import PositiveFloat
from nautilus_trader.config import PositiveInt
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.message import Event
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import OrderBookDepth10
from nautilus_trader.model.data import OrderBookDeltas
from nautilus_trader.model.enums import BookType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.events import OrderCanceled
from nautilus_trader.model.events import OrderDenied
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import LimitOrder
from nautilus_trader.trading.strategy import Strategy


class AvellanedaStoikovMarketMakerConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    gamma: PositiveFloat = 0.10
    kappa: PositiveFloat = 1.50
    tau: PositiveFloat = 1.0
    eta: PositiveFloat = 1.0
    inventory_limit: PositiveFloat = 5.0
    base_order_size: PositiveFloat = 5.0
    sigma_window: PositiveInt = 512
    alpha_window: PositiveInt = 32
    trend_weight: float = 0.10
    trend_inventory_weight: float = 0.0
    trend_size_weight: float = 0.0
    trend_gate: bool = False
    trend_gate_threshold: float = 0.0
    min_spread_abs: PositiveFloat = 1.0
    min_spread_bps: PositiveFloat = 5.0
    quote_interval_ms: int = 0
    book_data_type: str = "deltas"
    close_positions_on_stop: bool = False


class AvellanedaStoikovMarketMaker(Strategy):
    def __init__(self, config: AvellanedaStoikovMarketMakerConfig) -> None:
        if config.sigma_window < 2:
            raise ValueError("sigma_window must be at least 2")
        if config.alpha_window > config.sigma_window:
            raise ValueError("alpha_window must be less than or equal to sigma_window")
        if config.trend_inventory_weight < 0.0:
            raise ValueError("trend_inventory_weight must be non-negative")
        if config.trend_size_weight < 0.0:
            raise ValueError("trend_size_weight must be non-negative")
        if config.trend_gate_threshold < 0.0:
            raise ValueError("trend_gate_threshold must be non-negative")
        if config.quote_interval_ms < 0:
            raise ValueError("quote_interval_ms must be non-negative")
        if config.book_data_type not in ("deltas", "depth10"):
            raise ValueError("book_data_type must be deltas or depth10")
        super().__init__(config)

        self.instrument: Instrument | None = None
        self.book: OrderBook | None = None
        self.buy_order: LimitOrder | None = None
        self.sell_order: LimitOrder | None = None
        self._pending_quote: tuple[float, float, float, float] | None = None
        self.quote_records: list[dict[str, float | int]] = []
        self._last_mid: float | None = None
        self._last_quote_ts: int | None = None
        self._returns: deque[float] = deque(maxlen=config.sigma_window)

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            raise RuntimeError(f"Could not find instrument for {self.config.instrument_id}")

        self.book = OrderBook(
            instrument_id=self.config.instrument_id,
            book_type=BookType.L2_MBP,
        )
        if self.config.book_data_type == "depth10":
            self.subscribe_order_book_depth(self.config.instrument_id, book_type=BookType.L2_MBP, depth=10)
        else:
            self.subscribe_order_book_deltas(self.config.instrument_id)

    def on_order_book_deltas(self, deltas: OrderBookDeltas) -> None:
        if self.book is None:
            raise RuntimeError("Order book is not initialized")
        if self.instrument is None:
            raise RuntimeError("Instrument is not initialized")

        self.book.apply_deltas(deltas)
        bid = self.book.best_bid_price()
        ask = self.book.best_ask_price()
        if bid is None or ask is None:
            return

        bid_f = bid.as_double()
        ask_f = ask.as_double()
        if bid_f <= 0.0 or ask_f <= bid_f:
            raise ValueError(f"Invalid top of book: bid={bid_f} ask={ask_f}")

        self._update_quotes(deltas.ts_event, bid_f, ask_f)

    def on_order_book_depth(self, depth: OrderBookDepth10) -> None:
        if self.instrument is None:
            raise RuntimeError("Instrument is not initialized")

        bid = depth.bids[0]
        ask = depth.asks[0]
        bid_f = bid.price.as_double()
        ask_f = ask.price.as_double()
        if bid_f <= 0.0 or ask_f <= bid_f:
            raise ValueError(f"Invalid depth top of book: bid={bid_f} ask={ask_f}")

        self._update_quotes(depth.ts_event, bid_f, ask_f)

    def _update_quotes(self, ts_event: int, bid_f: float, ask_f: float) -> None:
        if self.instrument is None:
            raise RuntimeError("Instrument is not initialized")

        mid = (bid_f + ask_f) / 2.0
        if self._last_mid is not None:
            self._returns.append(mid - self._last_mid)
        self._last_mid = mid

        if len(self._returns) < self.config.sigma_window:
            return
        if self._last_quote_ts is not None:
            interval_ns = self.config.quote_interval_ms * 1_000_000
            if interval_ns > 0 and ts_event - self._last_quote_ts < interval_ns:
                return
        self._last_quote_ts = ts_event

        returns = list(self._returns)
        mean_return = sum(returns) / len(returns)
        sigma = sqrt(sum((value - mean_return) ** 2 for value in returns) / len(returns))
        alpha_returns = returns[-self.config.alpha_window :]
        alpha = sum(alpha_returns) / len(alpha_returns)
        inventory = float(self.portfolio.net_position(self.config.instrument_id))
        trend_signal = alpha / sigma if sigma > 0.0 else 0.0
        target_q = tanh(self.config.trend_inventory_weight * trend_signal)
        q = inventory / self.config.inventory_limit - target_q
        reservation = mid - q * self.config.gamma * sigma * self.config.tau + self.config.trend_weight * alpha
        optimal_spread = self.config.gamma * sigma * self.config.tau + (2.0 / self.config.gamma) * log(1.0 + self.config.gamma / self.config.kappa)
        min_spread = max(self.config.min_spread_abs, mid * self.config.min_spread_bps / 10_000.0)
        bid_quote = floor(min(reservation - optimal_spread / 2.0, mid - min_spread / 2.0))
        ask_quote = ceil(max(reservation + optimal_spread / 2.0, mid + min_spread / 2.0))
        if bid_quote >= ask_quote:
            raise ValueError(f"Crossed strategy quotes: bid_quote={bid_quote} ask_quote={ask_quote}")

        trend_size_skew = tanh(self.config.trend_size_weight * trend_signal)
        bid_size = self.config.base_order_size * max(
            0.0,
            1.0 - self.config.eta * q + trend_size_skew,
        )
        ask_size = self.config.base_order_size * max(
            0.0,
            1.0 + self.config.eta * q - trend_size_skew,
        )
        if self.config.trend_gate and abs(trend_signal) >= self.config.trend_gate_threshold:
            if trend_signal < 0.0:
                bid_size = 0.0
            elif trend_signal > 0.0:
                ask_size = 0.0
        self.quote_records.append(
            {
                "ts_event": ts_event,
                "bid": bid_f,
                "ask": ask_f,
                "mid": mid,
                "bid_quote": bid_quote,
                "ask_quote": ask_quote,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "inventory": inventory,
                "sigma": sigma,
                "alpha": alpha,
            },
        )
        self._replace_quotes(bid_quote, ask_quote, bid_size, ask_size)

    def _replace_quotes(self, bid_quote: float, ask_quote: float, bid_size: float, ask_size: float) -> None:
        if self.instrument is None:
            raise RuntimeError("Instrument is not initialized")

        self._pending_quote = (bid_quote, ask_quote, bid_size, ask_size)
        canceling = False
        if self.buy_order is not None and self.buy_order.is_open:
            canceling = True
            if not self.buy_order.is_pending_cancel:
                self.cancel_order(self.buy_order)
        if self.sell_order is not None and self.sell_order.is_open:
            canceling = True
            if not self.sell_order.is_pending_cancel:
                self.cancel_order(self.sell_order)
        if canceling:
            return

        self._submit_pending_quote()

    def _submit_pending_quote(self) -> None:
        if self.instrument is None:
            raise RuntimeError("Instrument is not initialized")
        if self._pending_quote is None:
            return

        bid_quote, ask_quote, bid_size, ask_size = self._pending_quote
        self._pending_quote = None

        min_qty = self.instrument.min_quantity.as_double()
        if bid_size >= min_qty:
            self.buy_order = self.order_factory.limit(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=self.instrument.make_qty(Decimal(str(bid_size))),
                price=self.instrument.make_price(bid_quote),
                time_in_force=TimeInForce.GTC,
                post_only=True,
            )
            self.submit_order(self.buy_order)
        else:
            self.buy_order = None

        if ask_size >= min_qty:
            self.sell_order = self.order_factory.limit(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.SELL,
                quantity=self.instrument.make_qty(Decimal(str(ask_size))),
                price=self.instrument.make_price(ask_quote),
                time_in_force=TimeInForce.GTC,
                post_only=True,
            )
            self.submit_order(self.sell_order)
        else:
            self.sell_order = None

    def _submit_pending_quote_if_flat(self) -> None:
        buy_live = self.buy_order is not None and self.buy_order.is_open
        sell_live = self.sell_order is not None and self.sell_order.is_open
        if not buy_live and not sell_live:
            self._submit_pending_quote()

    def on_order_canceled(self, event: OrderCanceled) -> None:
        if self.buy_order is not None and event.client_order_id == self.buy_order.client_order_id:
            self.buy_order = None
        if self.sell_order is not None and event.client_order_id == self.sell_order.client_order_id:
            self.sell_order = None
        self._submit_pending_quote_if_flat()

    def on_order_denied(self, event: OrderDenied) -> None:
        if self.buy_order is not None and event.client_order_id == self.buy_order.client_order_id:
            self.buy_order = None
        if self.sell_order is not None and event.client_order_id == self.sell_order.client_order_id:
            self.sell_order = None

    def on_event(self, event: Event) -> None:
        if isinstance(event, OrderFilled):
            if (
                self.buy_order is not None
                and event.client_order_id == self.buy_order.client_order_id
                and self.buy_order.is_closed
            ):
                self.buy_order = None
            if (
                self.sell_order is not None
                and event.client_order_id == self.sell_order.client_order_id
                and self.sell_order.is_closed
            ):
                self.sell_order = None

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        if self.config.close_positions_on_stop:
            self.close_all_positions(self.config.instrument_id)
        if self.config.book_data_type == "depth10":
            self.unsubscribe_order_book_depth(self.config.instrument_id)
        else:
            self.unsubscribe_order_book_deltas(self.config.instrument_id)

    def on_reset(self) -> None:
        self.buy_order = None
        self.sell_order = None
        self._pending_quote = None
        self.quote_records.clear()
        self._last_mid = None
        self._last_quote_ts = None
        self._returns.clear()
