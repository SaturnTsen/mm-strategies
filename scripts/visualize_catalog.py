#!/usr/bin/env python
import argparse
import html
import os
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq


FIXED_SCALAR = 10**16
ACTION_ADD = 1
ACTION_UPDATE = 2
ACTION_DELETE = 3
ACTION_CLEAR = 4
SIDE_BUY = 1
SIDE_SELL = 2
DATA_TYPES = ("order_book_deltas", "quote_tick", "trade_tick")


def read_catalog_type(catalog: Path, data_type: str, instrument: str) -> pd.DataFrame:
    directory = catalog / "data" / data_type / instrument
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {directory}")

    frames = []
    for file_index, path in enumerate(files):
        table = pq.read_table(path)
        frame = table.to_pandas()
        frame["_file_index"] = file_index
        frame["_row"] = range(len(frame))
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def discover_instruments(catalog: Path) -> list[str]:
    data_root = catalog / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"Catalog data directory does not exist: {data_root}")

    instrument_sets = []
    for data_type in DATA_TYPES:
        directory = data_root / data_type
        if not directory.exists():
            raise FileNotFoundError(f"Catalog data type directory does not exist: {directory}")
        instrument_sets.append({path.name for path in directory.iterdir() if path.is_dir()})

    instruments = sorted(set.intersection(*instrument_sets))
    if not instruments:
        raise FileNotFoundError(f"No instruments with all required data types: {', '.join(DATA_TYPES)}")
    return instruments


def fixed_to_float(value: bytes, signed: bool) -> float:
    return int.from_bytes(value, "little", signed=signed) / FIXED_SCALAR


def decode_fixed_column(series: pd.Series, signed: bool) -> pd.Series:
    return series.map(lambda value: fixed_to_float(value, signed=signed))


def add_datetime(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["dt"] = pd.to_datetime(frame["ts_event"], unit="ns", utc=True)
    return frame


def load_deltas(catalog: Path, instrument: str) -> pd.DataFrame:
    deltas = read_catalog_type(catalog, "order_book_deltas", instrument)
    deltas["price_f"] = decode_fixed_column(deltas["price"], signed=True)
    deltas["size_f"] = decode_fixed_column(deltas["size"], signed=False)
    deltas = add_datetime(deltas)
    return deltas.sort_values(["ts_init", "_file_index", "_row"], kind="stable").reset_index(drop=True)


def load_quotes(catalog: Path, instrument: str) -> pd.DataFrame:
    quotes = read_catalog_type(catalog, "quote_tick", instrument)
    quotes["bid_price_f"] = decode_fixed_column(quotes["bid_price"], signed=True)
    quotes["ask_price_f"] = decode_fixed_column(quotes["ask_price"], signed=True)
    quotes["bid_size_f"] = decode_fixed_column(quotes["bid_size"], signed=False)
    quotes["ask_size_f"] = decode_fixed_column(quotes["ask_size"], signed=False)
    quotes["mid_f"] = (quotes["bid_price_f"] + quotes["ask_price_f"]) / 2.0
    quotes["spread_f"] = quotes["ask_price_f"] - quotes["bid_price_f"]
    quotes["spread_bps"] = quotes["spread_f"] / quotes["mid_f"] * 10_000.0
    quotes = add_datetime(quotes)
    return quotes.sort_values(["ts_init", "_file_index", "_row"], kind="stable").reset_index(drop=True)


def load_trades(catalog: Path, instrument: str) -> pd.DataFrame:
    trades = read_catalog_type(catalog, "trade_tick", instrument)
    trades["price_f"] = decode_fixed_column(trades["price"], signed=True)
    trades["size_f"] = decode_fixed_column(trades["size"], signed=False)
    trades = add_datetime(trades)
    return trades.sort_values(["ts_init", "_file_index", "_row"], kind="stable").reset_index(drop=True)


def reconstruct_top_of_book(deltas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    snapshots = []

    for row in deltas.itertuples(index=False):
        action = int(row.action)
        side = int(row.side)
        price = float(row.price_f)
        size = float(row.size_f)

        if action == ACTION_CLEAR:
            bids.clear()
            asks.clear()
        elif action in (ACTION_ADD, ACTION_UPDATE):
            book = bids if side == SIDE_BUY else asks if side == SIDE_SELL else None
            if book is None:
                raise ValueError(f"Unexpected side={side} for action={action}")
            if size <= 0.0:
                book.pop(price, None)
            else:
                book[price] = size
        elif action == ACTION_DELETE:
            book = bids if side == SIDE_BUY else asks if side == SIDE_SELL else None
            if book is None:
                raise ValueError(f"Unexpected side={side} for delete")
            book.pop(price, None)
        else:
            raise ValueError(f"Unexpected action={action}")

        if bids and asks:
            best_bid = max(bids)
            best_ask = min(asks)
            bid_size = bids[best_bid]
            ask_size = asks[best_ask]
            mid = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
            snapshots.append(
                {
                    "dt": row.dt,
                    "ts_event": row.ts_event,
                    "ts_init": row.ts_init,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "bid_size": bid_size,
                    "ask_size": ask_size,
                    "mid": mid,
                    "spread": spread,
                    "spread_bps": spread / mid * 10_000.0,
                    "top_imbalance": (bid_size - ask_size) / (bid_size + ask_size),
                    "bid_levels": len(bids),
                    "ask_levels": len(asks),
                }
            )

    if not snapshots:
        raise ValueError("No valid top-of-book snapshots reconstructed")

    top = pd.DataFrame(snapshots)
    depth_rows = [{"side": "bid", "price": price, "size": size} for price, size in bids.items()]
    depth_rows.extend({"side": "ask", "price": price, "size": size} for price, size in asks.items())
    depth = pd.DataFrame(depth_rows)
    return top, depth


def save_top_of_book(top: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(top["dt"], top["best_bid"], label="best bid", linewidth=1.2)
    ax.plot(top["dt"], top["best_ask"], label="best ask", linewidth=1.2)
    ax.plot(top["dt"], top["mid"], label="mid", linewidth=1.0, alpha=0.8)
    ax.set_title("Reconstructed L2 Top Of Book")
    ax.set_xlabel("time")
    ax.set_ylabel("price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_spread_imbalance(top: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(top["dt"], top["spread_bps"], color="#8b2f2f", linewidth=1.0)
    axes[0].set_title("Spread")
    axes[0].set_ylabel("bps")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(top["dt"], top["top_imbalance"], color="#245c7a", linewidth=1.0)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_title("Top Level Imbalance")
    axes[1].set_ylabel("(bid_size - ask_size) / total")
    axes[1].set_xlabel("time")
    axes[1].grid(True, alpha=0.25)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_depth_snapshot(depth: pd.DataFrame, path: Path) -> None:
    if depth.empty:
        raise ValueError("Cannot plot empty depth snapshot")

    bids = depth[depth["side"] == "bid"].sort_values("price", ascending=False).head(20)
    asks = depth[depth["side"] == "ask"].sort_values("price", ascending=True).head(20)
    plot_frame = pd.concat([bids.assign(plot_size=-bids["size"]), asks.assign(plot_size=asks["size"])])

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2c7a51" if side == "ask" else "#9b3d3d" for side in plot_frame["side"]]
    ax.barh(plot_frame["price"], plot_frame["plot_size"], color=colors, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("Last Reconstructed L2 Depth")
    ax.set_xlabel("size, bids negative")
    ax.set_ylabel("price")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_trades_quotes(quotes: pd.DataFrame, trades: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(quotes["dt"], quotes["mid_f"], label="quote mid", linewidth=1.0)
    axes[0].scatter(trades["dt"], trades["price_f"], s=12, c=trades["size_f"], cmap="viridis", label="trades")
    axes[0].set_title("Quotes And Trades")
    axes[0].set_ylabel("price")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    per_second = trades.set_index("dt")["size_f"].resample("1s").agg(["count", "sum"]).fillna(0.0)
    axes[1].bar(per_second.index, per_second["count"], width=0.000008, label="trade count")
    axes[1].set_ylabel("count / second")
    axes[1].set_xlabel("time")
    axes[1].grid(True, alpha=0.25)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report_html(out_dir: Path, instrument: str, summary: dict[str, object]) -> None:
    rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(instrument)} catalog report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    h1 {{ font-size: 24px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin-bottom: 18px; }}
  </style>
</head>
<body>
  <h1>{html.escape(instrument)} Catalog Report</h1>
  <table>{rows}</table>
  <h2>Reconstructed Top Of Book</h2>
  <img src="top_of_book.png">
  <h2>Spread And Imbalance</h2>
  <img src="spread_imbalance.png">
  <h2>Last L2 Depth</h2>
  <img src="depth_snapshot.png">
  <h2>Quotes And Trades</h2>
  <img src="trades_quotes.png">
</body>
</html>
"""
    (out_dir / "index.html").write_text(body, encoding="utf-8")


def write_index_html(out_dir: Path, reports: list[tuple[str, Path]]) -> None:
    links = "\n".join(
        f'<li><a href="{html.escape(path.name)}/index.html">{html.escape(instrument)}</a></li>'
        for instrument, path in reports
    )
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Catalog Reports</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    h1 {{ font-size: 24px; }}
    li {{ margin: 8px 0; }}
  </style>
</head>
<body>
  <h1>Catalog Reports</h1>
  <ul>{links}</ul>
</body>
</html>
"""
    (out_dir / "index.html").write_text(body, encoding="utf-8")


def report_directory_name(instrument: str) -> str:
    return quote(instrument.replace("/", "_"), safe=".-_")


def build_report(catalog: Path, instrument: str, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    deltas = load_deltas(catalog, instrument)
    quotes = load_quotes(catalog, instrument)
    trades = load_trades(catalog, instrument)
    top, depth = reconstruct_top_of_book(deltas)

    top.to_csv(out_dir / "reconstructed_top_of_book.csv", index=False)
    depth.sort_values(["side", "price"]).to_csv(out_dir / "last_depth_snapshot.csv", index=False)

    summary = {
        "catalog": catalog,
        "instrument": instrument,
        "deltas": len(deltas),
        "quotes": len(quotes),
        "trades": len(trades),
        "top snapshots": len(top),
        "start": min(deltas["dt"].min(), quotes["dt"].min(), trades["dt"].min()),
        "end": max(deltas["dt"].max(), quotes["dt"].max(), trades["dt"].max()),
        "last bid": top["best_bid"].iloc[-1],
        "last ask": top["best_ask"].iloc[-1],
        "last spread bps": round(top["spread_bps"].iloc[-1], 4),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)

    save_top_of_book(top, out_dir / "top_of_book.png")
    save_spread_imbalance(top, out_dir / "spread_imbalance.png")
    save_depth_snapshot(depth, out_dir / "depth_snapshot.png")
    save_trades_quotes(quotes, trades, out_dir / "trades_quotes.png")
    write_report_html(out_dir, instrument, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("catalog"))
    parser.add_argument("-i", "--instrument", action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/catalog_visualization"))
    args = parser.parse_args()

    catalog = args.catalog.expanduser()
    instruments = args.instrument if args.instrument is not None else discover_instruments(catalog)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for instrument in instruments:
        report_dir = args.out_dir / report_directory_name(instrument)
        build_report(catalog, instrument, report_dir)
        reports.append((instrument, report_dir))
        print(f"Wrote {instrument} report to {report_dir / 'index.html'}")

    write_index_html(args.out_dir, reports)
    print(f"Wrote report index to {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
