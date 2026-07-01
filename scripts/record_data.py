#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path
from threading import Timer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nautilus_trader.adapters.binance import (
    BinanceDataClientConfig,
    BinanceDataClientFactory,
    BinanceProductType,
    BinanceSpotMarketDataMode,
)
from nautilus_trader.adapters.bybit import (
    BybitDataClientConfig,
    BybitDataClientFactory,
    BybitProductType,
)
from nautilus_trader.adapters.hyperliquid import (
    HyperliquidDataClientConfig,
    HyperliquidDataClientFactory,
)
from nautilus_trader.adapters.kraken import KrakenDataClientConfig, KrakenDataClientFactory
from nautilus_trader.adapters.okx import (
    OKX,
    OKXDataClientConfig,
    OKXDataClientFactory,
    OKXInstrumentType,
)
from nautilus_trader.common import Environment, LoggerConfig, LogLevel
from nautilus_trader.live import LiveNode, RotationConfig, StreamingConfig
from nautilus_trader.model import ActorId, ClientId, InstrumentId, TraderId
from nautilus_trader.testkit import DataTesterConfig


KRAKEN = "KRAKEN"
BINANCE = "BINANCE"
BYBIT = "BYBIT"
HYPERLIQUID = "HYPERLIQUID"
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

def main() -> None:
    args = parse_record_args(sys.argv[1:])
    venue_instruments = parse_venue_instruments(args.instrument)
    catalog_path = args.catalog.expanduser()
    catalog_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Recording multi-venue market data to catalog: {catalog_path}")

    if args.build_only:
        node = build_recording_node(args, catalog_path, venue_instruments)
        print(f"Built multi-venue data recorder. Catalog: {catalog_path}")
        return

    remaining_minutes = args.run_minutes
    session = 1
    while remaining_minutes > 0.0:
        session_minutes = min(args.instance_minutes, remaining_minutes)
        node = build_recording_node(args, catalog_path, venue_instruments)
        print(f"Starting recording session {session}: duration={session_minutes:g} minutes instance={node.instance_id}")
        stop_timer = Timer(session_minutes * 60.0, lambda: os.kill(os.getpid(), signal.SIGTERM))
        stop_timer.start()

        try:
            node.run()
        finally:
            stop_timer.cancel()

        remaining_minutes -= session_minutes
        session += 1


def build_recording_node(
    args: argparse.Namespace,
    catalog_path: Path,
    venue_instruments: dict[str, list[str]],
) -> LiveNode:
    builder = LiveNode.builder(
        name="recorder",
        trader_id=TraderId(args.trader_id),
        environment=Environment.LIVE,
    ).with_logging(
        LoggerConfig(stdout_level=LogLevel.from_str(args.log_level)),
    ).with_streaming_config(
        StreamingConfig(
            catalog_path=str(catalog_path),
            flush_interval_ms=args.flush_interval_ms,
            replace_existing=False,
            rotation_config=RotationConfig.size(args.max_file_size),
        ),
    )

    if KRAKEN in venue_instruments:
        builder.add_data_client(
            KRAKEN,
            KrakenDataClientFactory(),
            KrakenDataClientConfig(
                timeout_secs=args.http_timeout_secs,
                proxy_url=args.proxy_url,
            ),
        )

    if BINANCE in venue_instruments:
        builder.add_data_client(
            BINANCE,
            BinanceDataClientFactory(),
            BinanceDataClientConfig(
                product_type=BinanceProductType.SPOT,
                spot_market_data_mode=BinanceSpotMarketDataMode.Json,
            ),
        )

    if OKX in venue_instruments:
        missing = [
            name
            for name in ("OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE")
            if not os.getenv(name)
        ]
        if missing:
            raise RuntimeError(f"OKX data client requires environment variables: {', '.join(missing)}")

        builder.add_data_client(
            OKX,
            OKXDataClientFactory(),
            OKXDataClientConfig(
                instrument_types=(OKXInstrumentType.SPOT,),
                http_timeout_secs=args.http_timeout_secs,
                proxy_url=args.proxy_url,
            ),
        )

    if BYBIT in venue_instruments:
        builder.add_data_client(
            BYBIT,
            BybitDataClientFactory(),
            BybitDataClientConfig(
                product_types=(BybitProductType.SPOT,),
                http_timeout_secs=args.http_timeout_secs,
                proxy_url=args.proxy_url,
            ),
        )

    if HYPERLIQUID in venue_instruments:
        for value in venue_instruments[HYPERLIQUID]:
            symbol = InstrumentId.from_str(value).symbol.value
            if ":" in symbol:
                raise ValueError(f"Hyperliquid HIP3 instruments are not exposed by the pyo3 adapter: {value}")
            if not symbol.endswith("-SPOT") and not symbol.endswith("-PERP"):
                raise ValueError(f"Unsupported Hyperliquid instrument format: {value}")

        builder.add_data_client(
            HYPERLIQUID,
            HyperliquidDataClientFactory(),
            HyperliquidDataClientConfig(
                http_timeout_secs=args.http_timeout_secs,
                proxy_url=args.proxy_url,
            ),
        )

    node = builder.build()
    for venue, venue_values in venue_instruments.items():
        node.add_builtin_actor(
            "DataTester",
            DataTesterConfig(
                actor_id=ActorId(f"DataTester-{venue}"),
                instrument_ids=[InstrumentId.from_str(value) for value in venue_values],
                client_id=ClientId(venue),
                subscribe_quotes=venue != BYBIT,
                subscribe_trades=True,
                subscribe_book_deltas=True,
                manage_book=True,
                book_interval_ms=args.book_interval_ms,
                log_data=args.log_data,
            ),
        )
    return node

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


if __name__ == "__main__":
    main()
