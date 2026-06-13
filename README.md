# Hyperliquid Market Making

Rust-first market data capture for Hyperliquid through the Nautilus adapter, with
Python used only for Nautilus catalog writing and preview charts.

## Live Recording

```bash
cargo run --manifest-path mm-strategies/Cargo.toml --bin live-recorder -- \
  --instrument BTC-USD-PERP.HYPERLIQUID \
  --out-dir mm-strategies/data/recordings \
  --snapshot-interval-ms 250 \
  --duration-secs 300
```

The live recorder uses `LiveNode + DataActor` and subscribes to:

- L2 `OrderBookDeltas`
- `QuoteTick`
- `TradeTick`
- managed top-10 book snapshots at the configured interval

It writes JSONL staging files:

- `quotes.jsonl`
- `trades.jsonl`
- `book_deltas.jsonl`
- `depth10.jsonl`

## Historical Ingest

The first version expects local decompressed JSONL. If the source is `.gz` or
`.lz4`, decompress it before running this command.

```bash
cargo run --manifest-path mm-strategies/Cargo.toml --bin historical-ingest -- \
  --source tardis \
  --instrument BTC-USD-PERP.HYPERLIQUID \
  --input /path/to/historical-jsonl \
  --out-dir mm-strategies/data/recordings
```

Supported source modes:

- `tardis`
- `hyperliquid-archive`

Historical snapshot-only data is suitable for book shape previews, not precise
queue simulation.

## Write Nautilus Catalog

Use the project Python environment:

```bash
conda run -n nt-hl python nautilus-hyperliquid-py/scripts/catalog_writer.py \
  --recording-dir mm-strategies/data/recordings \
  --catalog mm-strategies/data/catalog \
  --instrument BTC-USD-PERP.HYPERLIQUID
```

## Preview Book State

```bash
conda run -n nt-hl python mm-strategies/scripts/preview_book.py \
  --catalog mm-strategies/data/catalog \
  --instrument BTC-USD-PERP.HYPERLIQUID \
  --out-dir mm-strategies/reports/book_preview
```

Outputs:

- `depth10_preview.csv`
- `top_of_book.png`
- `imbalance.png`
- `depth_heatmap.png`
