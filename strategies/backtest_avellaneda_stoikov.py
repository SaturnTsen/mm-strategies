#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from avellaneda_stoikov_market_making import AvellanedaStoikovMarketMaker
from avellaneda_stoikov_market_making import AvellanedaStoikovMarketMakerConfig
from data_utils import load_depths
from data_utils import parse_catalog_file_segment
from data_utils import segment_from_key
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.core.nautilus_pyo3 import BacktestEngine # type: ignore
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.data import BookOrder
from nautilus_trader.model.data import OrderBookDepth10
from nautilus_trader.model import Money
from nautilus_trader.model import TraderId
from nautilus_trader.model import Venue
from nautilus_trader.model.currencies import BTC
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import BookType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.core.nautilus_pyo3 import ParquetDataCatalog


DEFAULT_CATALOG_PATH = Path("catalog")
DEFAULT_INSTRUMENT = "BTCUSDT.BINANCE"
DEFAULT_REPORT_DIR = Path("reports/avellaneda_stoikov")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Avellaneda-Stoikov market making with Nautilus BacktestEngine.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--instrument", default=DEFAULT_INSTRUMENT)
    parser.add_argument("--segment", default="latest", help="Segment key, latest, or all.")
    parser.add_argument("--book-data-type", default="depth10", choices=["depth10", "deltas"])
    parser.add_argument("--max-rows", type=int, default=None, help="Max OrderBookDeltas batches from the segment start.")
    parser.add_argument("--max-minutes", type=float, default=None, help="Max minutes from the segment start.")
    parser.add_argument("--min-duration-minutes", type=float, default=45.0)
    parser.add_argument("--min-5m-fills", type=int, default=5)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--trader-id", default="BACKTESTER-AVS-001")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--kappa", type=float, default=1.50)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--inventory-limit", type=float, default=5.0)
    parser.add_argument("--base-order-size", type=float, default=5.0)
    parser.add_argument("--sigma-window", type=int, default=512)
    parser.add_argument("--alpha-window", type=int, default=32)
    parser.add_argument("--trend-weight", type=float, default=0.10)
    parser.add_argument("--trend-inventory-weight", type=float, default=0.0)
    parser.add_argument("--trend-size-weight", type=float, default=0.0)
    parser.add_argument("--min-spread-abs", type=float, default=1.0)
    parser.add_argument("--min-spread-bps", type=float, default=5.0)
    parser.add_argument("--quote-interval-ms", type=int, default=0)
    parser.add_argument("--starting-cash", type=float, default=1_000_000.0)
    parser.add_argument("--starting-base", type=float, default=5.0)
    parser.add_argument("--maker-fee", type=str, default="0.001")
    parser.add_argument("--taker-fee", type=str, default="0.001")
    parser.add_argument("--account-type", default="CASH", choices=["CASH", "MARGIN"])
    parser.add_argument("--allow-cash-borrowing", action="store_true")
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    if args.max_minutes is not None and args.max_minutes <= 0.0:
        raise ValueError("--max-minutes must be positive")
    if args.min_duration_minutes <= 0.0:
        raise ValueError("--min-duration-minutes must be positive")
    if args.min_5m_fills <= 0:
        raise ValueError("--min-5m-fills must be positive")
    return args


def resolve_segment(catalog: Path, instrument: str, requested: str) -> str | None:
    if requested == "all":
        return None
    if requested != "latest":
        return requested

    directory = catalog / "data" / "order_book_deltas" / instrument
    if not directory.exists():
        raise FileNotFoundError(f"No order_book_deltas directory: {directory}")
    segments = [parse_catalog_file_segment(path) for path in directory.glob("*.parquet")]
    if not segments:
        raise FileNotFoundError(f"No order_book_deltas parquet files under {directory}")
    return max(segments, key=lambda segment: segment.end).key


def load_catalog_data(
    catalog_path: Path,
    instrument: str,
    segment_key: str | None,
    max_rows: int | None,
    max_minutes: float | None,
    book_data_type: str,
) -> tuple[list[object], list[object], pd.Timestamp, pd.Timestamp]:
    catalog = ParquetDataCatalog(str(catalog_path))
    if book_data_type == "depth10":
        book_data = load_depth10_data(catalog_path, instrument, segment_key)
    else:
        query = {}
        if segment_key is not None:
            segment = segment_from_key(segment_key)
            query["start"] = segment.start
            query["end"] = segment.end
        book_data = catalog.order_book_deltas([instrument], batched=True, **query)

    if not book_data:
        raise ValueError(f"No {book_data_type} book data loaded for {instrument} segment={segment_key or 'all'}")
    if max_minutes is not None:
        cutoff_ns = book_data[0].ts_event + int(max_minutes * 60.0 * 1_000_000_000)
        book_data = [batch for batch in book_data if batch.ts_event <= cutoff_ns]
        if not book_data:
            raise ValueError(f"No book data loaded within --max-minutes={max_minutes}")
    if max_rows is not None:
        book_data = book_data[:max_rows]

    start = pd.Timestamp(book_data[0].ts_event, unit="ns", tz="UTC")
    end = pd.Timestamp(book_data[-1].ts_event, unit="ns", tz="UTC")
    trades = catalog.trade_ticks([instrument], start=start, end=end)
    return book_data, trades, start, end


def load_depth10_data(catalog_path: Path, instrument: str, segment_key: str | None) -> list[OrderBookDepth10]:
    instrument_id = InstrumentId.from_str(instrument)
    depths = load_depths(catalog_path, instrument, segment_key)
    depths = depths.sort_values(["ts_init", "_cache_file_index", "_file_index", "_row"], kind="stable")
    depths = depths.groupby("ts_event", sort=False).tail(1).reset_index(drop=True)
    depths = depths[
        (depths["bid_price_0_f"] > 0.0)
        & (depths["ask_price_0_f"] > 0.0)
        & (depths["bid_price_0_f"] < depths["ask_price_0_f"])
    ].reset_index(drop=True)
    if depths.empty:
        raise ValueError(f"No valid depth10 top-of-book rows for {instrument} segment={segment_key or 'all'}")

    empty_bid = BookOrder(OrderSide.NO_ORDER_SIDE, Price(0, precision=8), Quantity(0, precision=8), 0)
    empty_ask = BookOrder(OrderSide.NO_ORDER_SIDE, Price(0, precision=8), Quantity(0, precision=8), 0)
    data: list[OrderBookDepth10] = []
    for sequence, row in enumerate(depths.itertuples(index=False)):
        bids = []
        asks = []
        bid_counts = []
        ask_counts = []
        for level in range(10):
            bid_price = getattr(row, f"bid_price_{level}_f")
            bid_size = getattr(row, f"bid_size_{level}_f")
            bid_count = int(getattr(row, f"bid_count_{level}"))
            if bid_count > 0 and pd.notna(bid_price) and pd.notna(bid_size) and bid_price > 0.0 and bid_size > 0.0:
                bids.append(BookOrder(OrderSide.BUY, Price(float(bid_price), precision=8), Quantity(float(bid_size), precision=8), 0))
                bid_counts.append(bid_count)
            else:
                bids.append(empty_bid)
                bid_counts.append(0)

            ask_price = getattr(row, f"ask_price_{level}_f")
            ask_size = getattr(row, f"ask_size_{level}_f")
            ask_count = int(getattr(row, f"ask_count_{level}"))
            if ask_count > 0 and pd.notna(ask_price) and pd.notna(ask_size) and ask_price > 0.0 and ask_size > 0.0:
                asks.append(BookOrder(OrderSide.SELL, Price(float(ask_price), precision=8), Quantity(float(ask_size), precision=8), 0))
                ask_counts.append(ask_count)
            else:
                asks.append(empty_ask)
                ask_counts.append(0)

        data.append(
            OrderBookDepth10(
                instrument_id=instrument_id,
                bids=bids,
                asks=asks,
                bid_counts=bid_counts,
                ask_counts=ask_counts,
                flags=0,
                sequence=sequence,
                ts_event=int(row.ts_event),
                ts_init=int(row.ts_init),
            ),
        )
    return data


def build_instrument(instrument: str):
    if instrument != DEFAULT_INSTRUMENT:
        raise ValueError(f"Only {DEFAULT_INSTRUMENT} is wired in this backtest")
    return CurrencyPair(
        instrument_id=InstrumentId(
            symbol=Symbol("BTCUSDT"),
            venue=Venue("BINANCE"),
        ),
        raw_symbol=Symbol("BTCUSDT"),
        base_currency=BTC,
        quote_currency=USDT,
        price_precision=8,
        size_precision=8,
        price_increment=Price(0.01, precision=8),
        size_increment=Quantity(0.00000001, precision=8),
        lot_size=None,
        max_quantity=Quantity(9000, precision=8),
        min_quantity=Quantity(0.00000001, precision=8),
        max_notional=None,
        min_notional=Money(10.00000000, USDT),
        max_price=Price(1000000, precision=8),
        min_price=Price(0.00000001, precision=8),
        margin_init=Decimal("0"),
        margin_maint=Decimal("0"),
        maker_fee=Decimal("0.001"),
        taker_fee=Decimal("0.001"),
        ts_event=0,
        ts_init=0,
    )


def build_instrument_with_fees(instrument: str, maker_fee: Decimal, taker_fee: Decimal):
    result = build_instrument(instrument)
    if maker_fee <= Decimal("0") or taker_fee <= Decimal("0"):
        raise ValueError("Fees must be positive")
    return CurrencyPair(
        instrument_id=result.id,
        raw_symbol=result.raw_symbol,
        base_currency=result.base_currency,
        quote_currency=result.quote_currency,
        price_precision=result.price_precision,
        size_precision=result.size_precision,
        price_increment=result.price_increment,
        size_increment=result.size_increment,
        lot_size=result.lot_size,
        max_quantity=result.max_quantity,
        min_quantity=result.min_quantity,
        max_notional=result.max_notional,
        min_notional=result.min_notional,
        max_price=result.max_price,
        min_price=result.min_price,
        margin_init=result.margin_init,
        margin_maint=result.margin_maint,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        ts_event=0,
        ts_init=0,
    )


def file_token(value: float | int | str) -> str:
    return str(value).replace(".", "p").replace("-", "m").replace("/", "")


def fill_metrics(fills_report: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int]:
    window = pd.Timedelta(minutes=5)
    window_count = int((end - start) / window)
    if window_count <= 0:
        return len(fills_report), 0

    bins = pd.DatetimeIndex([start + window * step for step in range(window_count + 1)])
    if fills_report.empty:
        return 0, 0

    fill_times = pd.to_datetime(fills_report["ts_last"], utc=True)
    counts = fill_times.groupby(
        pd.cut(fill_times, bins, right=False),
        observed=False,
    ).size()
    if counts.empty:
        return len(fills_report), 0
    return len(fills_report), int(counts.min())


def money_as_float(value) -> float:
    if value is None:
        return 0.0
    return value.as_double()


def main() -> None:
    args = parse_args()
    catalog_path = args.catalog.expanduser()
    segment_key = resolve_segment(catalog_path, args.instrument, args.segment)
    instrument = build_instrument_with_fees(args.instrument, Decimal(args.maker_fee), Decimal(args.taker_fee))
    book_data, trades, start, end = load_catalog_data(
        catalog_path,
        args.instrument,
        segment_key,
        args.max_rows,
        args.max_minutes,
        args.book_data_type,
    )
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId(args.trader_id),
            logging=LoggingConfig(log_level=args.log_level),
        ),
    )
    venue = Venue("BINANCE")
    account_type = AccountType[args.account_type]
    starting_balances = (
        [Money(args.starting_cash, USDT)]
        if account_type == AccountType.MARGIN
        else [Money(args.starting_cash, USDT), Money(args.starting_base, BTC)]
    )
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=account_type,
        base_currency=USDT if account_type == AccountType.MARGIN else None,
        starting_balances=starting_balances,
        book_type=BookType.L2_MBP,
        allow_cash_borrowing=args.allow_cash_borrowing,
    )
    engine.add_instrument(instrument)
    engine.add_data(book_data)
    if trades:
        engine.add_data(trades)

    strategy = AvellanedaStoikovMarketMaker(
        config=AvellanedaStoikovMarketMakerConfig(
            instrument_id=instrument.id,
            gamma=args.gamma,
            kappa=args.kappa,
            tau=args.tau,
            eta=args.eta,
            inventory_limit=args.inventory_limit,
            base_order_size=args.base_order_size,
            sigma_window=args.sigma_window,
            alpha_window=args.alpha_window,
            trend_weight=args.trend_weight,
            trend_inventory_weight=args.trend_inventory_weight,
            trend_size_weight=args.trend_size_weight,
            min_spread_abs=args.min_spread_abs,
            min_spread_bps=args.min_spread_bps,
            quote_interval_ms=args.quote_interval_ms,
            book_data_type=args.book_data_type,
        ),
    )
    engine.add_strategy(strategy)
    engine.run()

    account_report = engine.trader.generate_account_report(venue)
    fills_report = engine.trader.generate_order_fills_report()
    positions_report = engine.trader.generate_positions_report()
    portfolio_pnl = (
        money_as_float(strategy.portfolio.realized_pnl(instrument.id))
        + money_as_float(strategy.portfolio.unrealized_pnl(instrument.id))
    )
    fills_count, min_5m_fills = fill_metrics(fills_report, start, end)
    duration_minutes = (end - start).total_seconds() / 60.0
    passes_duration = duration_minutes >= args.min_duration_minutes
    passes_fill_rate = min_5m_fills >= args.min_5m_fills
    passes_pnl = portfolio_pnl > 0.0

    args.report_dir.mkdir(parents=True, exist_ok=True)
    params = (
        f"g{file_token(args.gamma)}"
        f"_k{file_token(args.kappa)}"
        f"_tau{file_token(args.tau)}"
        f"_eta{file_token(args.eta)}"
        f"_L{file_token(args.inventory_limit)}"
        f"_x{file_token(args.base_order_size)}"
        f"_sw{args.sigma_window}"
        f"_aw{args.alpha_window}"
        f"_tiw{file_token(args.trend_inventory_weight)}"
        f"_tsw{file_token(args.trend_size_weight)}"
        f"_qms{args.quote_interval_ms}"
        f"_mbps{file_token(args.min_spread_bps)}"
        f"_mf{file_token(args.maker_fee)}"
        f"_{args.book_data_type}"
        f"_{args.account_type.lower()}"
    )
    label = f"{args.instrument}_{segment_key or 'all'}_{params}".replace("/", "").replace(":", "")
    account_path = args.report_dir / f"{label}_account.csv"
    fills_path = args.report_dir / f"{label}_fills.csv"
    positions_path = args.report_dir / f"{label}_positions.csv"
    account_report.to_csv(account_path)
    fills_report.to_csv(fills_path)
    positions_report.to_csv(positions_path)
    metrics_path = args.report_dir / f"{label}_metrics.csv"
    pd.Series(
        {
            "instrument": args.instrument,
            "segment": segment_key or "all",
            "start": start,
            "end": end,
            "portfolio_pnl": portfolio_pnl,
            "fills": fills_count,
            "min_5m_fills": min_5m_fills,
            "duration_minutes": duration_minutes,
            "passes_duration": passes_duration,
            "passes_fill_rate": passes_fill_rate,
            "passes_pnl": passes_pnl,
            "passes_all": passes_duration and passes_fill_rate and passes_pnl,
            "maker_fee": args.maker_fee,
            "taker_fee": args.taker_fee,
            "book_data_type": args.book_data_type,
            "trend_inventory_weight": args.trend_inventory_weight,
            "trend_size_weight": args.trend_size_weight,
            "account_type": args.account_type,
            "allow_cash_borrowing": args.allow_cash_borrowing,
        },
    ).to_frame("value").to_csv(metrics_path)

    print(f"instrument={args.instrument}")
    print(f"segment={segment_key or 'all'}")
    print(f"start={start}")
    print(f"end={end}")
    print(f"book_data_type={args.book_data_type}")
    print(f"book_events={len(book_data)}")
    print(f"trade_ticks={len(trades)}")
    print(f"fills={fills_count}")
    print(f"min_5m_fills={min_5m_fills}")
    print(f"duration_minutes={duration_minutes:.6f}")
    print(f"portfolio_pnl={portfolio_pnl:.8f}")
    print(f"passes_duration={passes_duration}")
    print(f"passes_fill_rate={passes_fill_rate}")
    print(f"passes_pnl={passes_pnl}")
    print(f"passes_all={passes_duration and passes_fill_rate and passes_pnl}")
    print(f"positions={len(positions_report)}")
    print(f"account_path={account_path}")
    print(f"fills_path={fills_path}")
    print(f"positions_path={positions_path}")
    print(f"metrics_path={metrics_path}")


if __name__ == "__main__":
    main()
