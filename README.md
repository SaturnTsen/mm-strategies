# MM-Strategies

## Python Multi-Venue Recording

Record BTC spot venues plus Hyperliquid BTC perp into a Nautilus catalog. The
script stops after 30 minutes by default and then converts the live stream files
to parquet.

Default instruments:

- `BTC/USD.KRAKEN`
- `BTCUSDT.BINANCE`
- `BTC-USDT.OKX`
- `BTCUSDT-SPOT.BYBIT`
- `BTC-USD-PERP.HYPERLIQUID`

OKX requires API credentials even for the current data client factory:

```bash
export OKX_API_KEY=...
export OKX_API_SECRET=...
export OKX_API_PASSPHRASE=...
```

Run the default 30 minute recording:

```bash
conda run -n nt-hl python scripts/record_data.py
```

Run without OKX credentials:

```bash
conda run -n nt-hl python scripts/record_data.py \
  --instrument BTC/USD.KRAKEN \
  --instrument BTCUSDT.BINANCE \
  --instrument BTCUSDT-SPOT.BYBIT \
  --instrument BTC-USD-PERP.HYPERLIQUID
```

Set the runtime limit:

```bash
conda run -n nt-hl python scripts/record_data.py --run-minutes 10
```

Build the node configuration without running:

```bash
conda run -n nt-hl python scripts/record_data.py --build-only
```

Useful options:

- `--catalog catalog`: output catalog path.
- `--run-minutes 30`: runtime limit before automatic shutdown.
- `--no-convert-to-parquet`: keep live stream files without converting.
- `--proxy-url http://127.0.0.1:7890`: proxy for supported adapters.
- `--log-level DEBUG`: increase Nautilus logging.

## Rust Hyperliquid Recording

```bash
cargo run --bin live-recorder -- \
  --instrument BTC-USD-PERP.HYPERLIQUID \
  --catalog data/catalog \
  --instance-id hl-live \
  --snapshot-interval-ms 250 \
  --duration-secs 300
```

The recorder uses `LiveNode + DataActor` only for subscriptions:

- L2 `OrderBookDeltas`
- `QuoteTick`
- `TradeTick`
- managed top-10 book snapshots

Persistence is handled by Nautilus `FeatherWriter`, subscribed to the Nautilus
message bus. It writes live stream files under `data/catalog/live/hl-live/`.
There is no project-local JSONL staging layer.
