#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from threading import Timer
from typing import Any

from nautilus_trader.adapters.binance import (
    BINANCE,
    BinanceAccountType,
    BinanceDataClientConfig,
    BinanceLiveDataClientFactory,
)
from nautilus_trader.adapters.bybit import (
    BYBIT,
    BybitDataClientConfig,
    BybitLiveDataClientFactory,
    BybitProductType,
)
from nautilus_trader.adapters.hyperliquid import (
    HYPERLIQUID,
    HyperliquidDataClientConfig,
    HyperliquidLiveDataClientFactory,
)
from nautilus_trader.adapters.hyperliquid.enums import HyperliquidProductType
from nautilus_trader.adapters.kraken import (
    KRAKEN,
    KrakenDataClientConfig,
    KrakenLiveDataClientFactory,
)
from nautilus_trader.adapters.okx import (
    OKX,
    OKXDataClientConfig,
    OKXLiveDataClientFactory,
)
from nautilus_trader.common import Environment
from nautilus_trader.common.config import CUSTOM_ENCODINGS
from nautilus_trader.config import (
    ImportableActorConfig,
    InstrumentProviderConfig,
    LoggingConfig,
    StreamingConfig,
    TradingNodeConfig,
)
from nautilus_trader.core.nautilus_pyo3.okx import OKXInstrumentType  # type: ignore
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import OrderBookDeltas, QuoteTick, TradeTick
from nautilus_trader.model.identifiers import InstrumentId, TraderId
from nautilus_trader.persistence.writer import RotationMode


DEFAULT_CATALOG_PATH = Path("catalog")
SUPPORTED_VENUE_INSTRUMENTS = {
    KRAKEN: ["BTC/USD.KRAKEN"],
    BINANCE: ["BTCUSDT.BINANCE"],
    OKX: ["BTC-USDT.OKX"],
    BYBIT: ["BTCUSDT-SPOT.BYBIT"],
    HYPERLIQUID: ["BTC-USD-PERP.HYPERLIQUID"],
}
DEFAULT_VENUE_INSTRUMENTS = {
    KRAKEN: SUPPORTED_VENUE_INSTRUMENTS[KRAKEN],
    BINANCE: SUPPORTED_VENUE_INSTRUMENTS[BINANCE],
    OKX: SUPPORTED_VENUE_INSTRUMENTS[OKX],
    BYBIT: SUPPORTED_VENUE_INSTRUMENTS[BYBIT],
    HYPERLIQUID: SUPPORTED_VENUE_INSTRUMENTS[HYPERLIQUID],
}
STREAMING_DATA_TYPES = [
    QuoteTick,
    TradeTick,
    OrderBookDeltas,
]
CUSTOM_ENCODINGS[type(OKXInstrumentType.SPOT)] = lambda value: value.value
CUSTOM_ENCODINGS[type(BybitProductType.SPOT)] = lambda value: value.value


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        record_main(parse_record_args(sys.argv[2:]))
    else:
        record_main(parse_record_args(sys.argv[1:]))


def record_main(args: argparse.Namespace) -> None:
    venue_instruments = parse_venue_instruments(args.instrument)
    catalog_path = args.catalog.expanduser()
    catalog_path.mkdir(parents=True, exist_ok=True)

    if args.build_only:
        node = build_recording_node(args, catalog_path, venue_instruments)
        node.build()
        print(f"Built multi-venue data recorder. Catalog: {catalog_path}")
        return

    remaining_minutes = args.run_minutes
    session = 1
    while remaining_minutes > 0.0:
        session_minutes = min(args.instance_minutes, remaining_minutes)
        node = build_recording_node(args, catalog_path, venue_instruments)
        node.build()
        print(f"Starting recording session {session}: duration={session_minutes:g} minutes instance={node.instance_id}")
        stop_timer = Timer(session_minutes * 60.0, stop_node, args=(node, session_minutes))
        stop_timer.start()

        try:
            node.run(raise_exception=True)
        finally:
            stop_timer.cancel()

        remaining_minutes -= session_minutes
        session += 1


def build_recording_node(
    args: argparse.Namespace,
    catalog_path: Path,
    venue_instruments: dict[str, list[str]],
) -> TradingNode:
    node_config = TradingNodeConfig(
        trader_id=TraderId(args.trader_id),
        environment=Environment.LIVE,
        streaming=StreamingConfig(
            catalog_path=str(catalog_path),
            flush_interval_ms=args.flush_interval_ms,
            include_types=STREAMING_DATA_TYPES,
            rotation_mode=RotationMode.SIZE,
            max_file_size=args.max_file_size,
            replace_existing=False,
        ),
        logging=LoggingConfig(log_level=args.log_level),
        data_clients=build_data_clients(args, venue_instruments),
        actors=[
            build_data_tester_config(args, venue, venue_values)
            for venue, venue_values in venue_instruments.items()
        ],
    )

    node = TradingNode(config=node_config)
    for venue, factory in {
        KRAKEN: KrakenLiveDataClientFactory,
        BINANCE: BinanceLiveDataClientFactory,
        OKX: OKXLiveDataClientFactory,
        BYBIT: BybitLiveDataClientFactory,
        HYPERLIQUID: HyperliquidLiveDataClientFactory,
    }.items():
        if venue in venue_instruments:
            node.add_data_client_factory(venue, factory)
    return node


def stop_node(node: TradingNode, run_minutes: float) -> None:
    print(f"Run time limit reached: {run_minutes:g} minutes. Stopping node...")
    node.kernel.loop.call_soon_threadsafe(
        lambda: node.kernel.loop.create_task(node.stop_async()),
    )


def parse_venue_instruments(values: list[str] | None) -> dict[str, list[str]]:
    if values is None:
        return {venue: list(instruments) for venue, instruments in DEFAULT_VENUE_INSTRUMENTS.items()}

    venue_instruments: dict[str, list[str]] = {}
    for value in values:
        instrument_id = InstrumentId.from_str(value)
        venue = instrument_id.venue.value
        if venue not in SUPPORTED_VENUE_INSTRUMENTS:
            raise ValueError(f"Unsupported venue for recorder: {venue}")
        venue_instruments.setdefault(venue, []).append(value)

    return venue_instruments


def build_data_clients(
    args: argparse.Namespace,
    venue_instruments: dict[str, list[str]],
) -> dict[str, Any]:
    data_clients: dict[str, Any] = {}

    if KRAKEN in venue_instruments:
        data_clients[KRAKEN] = KrakenDataClientConfig(
            instrument_provider=InstrumentProviderConfig(load_all=True),
            http_timeout_secs=args.http_timeout_secs,
            proxy_url=args.proxy_url,
        )

    if BINANCE in venue_instruments:
        data_clients[BINANCE] = BinanceDataClientConfig(
            account_type=BinanceAccountType.SPOT,
            instrument_provider=InstrumentProviderConfig(
                load_ids=frozenset(InstrumentId.from_str(value) for value in venue_instruments[BINANCE]),
            ),
            proxy_url=args.proxy_url,
        )

    if OKX in venue_instruments:
        missing = [
            name
            for name in ("OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE")
            if not os.getenv(name)
        ]
        if missing:
            raise RuntimeError(f"OKX data client requires environment variables: {', '.join(missing)}")

        data_clients[OKX] = OKXDataClientConfig(
            instrument_types=(OKXInstrumentType.SPOT,),
            instrument_provider=InstrumentProviderConfig(
                load_ids=frozenset(InstrumentId.from_str(value) for value in venue_instruments[OKX]),
            ),
            http_timeout_secs=args.http_timeout_secs,
            proxy_url=args.proxy_url,
        )

    if BYBIT in venue_instruments:
        data_clients[BYBIT] = BybitDataClientConfig(
            product_types=(BybitProductType.SPOT,),
            instrument_provider=InstrumentProviderConfig(
                load_ids=frozenset(InstrumentId.from_str(value) for value in venue_instruments[BYBIT]),
            ),
            proxy_url=args.proxy_url,
        )

    if HYPERLIQUID in venue_instruments:
        product_types, filters = build_hyperliquid_discovery(venue_instruments[HYPERLIQUID])
        data_clients[HYPERLIQUID] = HyperliquidDataClientConfig(
            product_types=product_types,
            instrument_provider=InstrumentProviderConfig(
                load_all=True,
                filters=filters,
            ),
            http_timeout_secs=args.http_timeout_secs,
            proxy_url=args.proxy_url,
        )

    return data_clients


def build_hyperliquid_discovery(
    instruments: list[str],
) -> tuple[tuple[HyperliquidProductType, ...], dict[str, tuple[str, ...]]]:
    product_types: set[HyperliquidProductType] = set()
    market_types: set[str] = set()
    bases: set[str] = set()

    for value in instruments:
        symbol = InstrumentId.from_str(value).symbol.value
        bases.add(symbol.split("-")[0].upper())

        if symbol.endswith("-SPOT"):
            product_types.add(HyperliquidProductType.SPOT)
            market_types.add("spot")
        elif symbol.endswith("-PERP"):
            product_types.add(HyperliquidProductType.PERP)
            market_types.add("perp")
        elif ":" in symbol:
            product_types.add(HyperliquidProductType.PERP_HIP3)
            market_types.add("perp_hip3")
        else:
            raise ValueError(f"Unsupported Hyperliquid instrument format: {value}")

    return (
        tuple(sorted(product_types, key=lambda product_type: product_type.value)),
        {
            "market_types": tuple(sorted(market_types)),
            "bases": tuple(sorted(bases)),
        },
    )


def build_data_tester_config(
    args: argparse.Namespace,
    venue: str,
    instruments: list[str],
) -> ImportableActorConfig:
    return ImportableActorConfig(
        actor_path="nautilus_trader.test_kit.strategies.tester_data:DataTester",
        config_path="nautilus_trader.test_kit.strategies.tester_data:DataTesterConfig",
        config={
            "component_id": f"DataTester-{venue}",
            "instrument_ids": instruments,
            "client_id": venue,
            "subscribe_quotes": venue != BYBIT,
            "subscribe_trades": True,
            "subscribe_book_deltas": True,
            "manage_book": True,
            "book_interval_ms": args.book_interval_ms,
            "log_data": args.log_data,
        },
    )


def parse_record_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record multi-venue BTC spot market data to a Nautilus catalog.",
    )
    parser.add_argument("--trader-id", default="RECORDER-001")
    parser.add_argument("-i", "--instrument", action="append", default=None)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--flush-interval-ms", type=int, default=1000)
    parser.add_argument("--max-file-size", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--book-interval-ms", type=int, default=10)
    parser.add_argument("--http-timeout-secs", type=int, default=10)
    parser.add_argument("--proxy-url", default=None)
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-minutes", type=float, default=30.0)
    parser.add_argument("--instance-minutes", type=float, default=60.0)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args(argv)
    if args.run_minutes <= 0:
        raise ValueError("--run-minutes must be positive")
    if args.instance_minutes <= 0:
        raise ValueError("--instance-minutes must be positive")
    return args
if __name__ == "__main__":
    main()
