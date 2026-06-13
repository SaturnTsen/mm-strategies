#!/usr/bin/env python3
"""Generate a first-pass Hyperliquid book preview from a Nautilus catalog."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
from nautilus_trader.persistence.catalog import ParquetDataCatalog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("mm-strategies/data/catalog"))
    parser.add_argument("--instrument", default="BTC-USD-PERP.HYPERLIQUID")
    parser.add_argument("--out-dir", type=Path, default=Path("mm-strategies/reports/book_preview"))
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(str(args.catalog))
    depths = catalog.order_book_depth10(instrument_ids=[args.instrument])
    if not depths:
        raise SystemExit(f"No OrderBookDepth10 records found for {args.instrument} in {args.catalog}")

    df = depth_frame(depths[-args.limit :])
    df.to_csv(args.out_dir / "depth10_preview.csv", index=False)
    plot_price_and_spread(df, args.out_dir / "top_of_book.png")
    plot_imbalance(df, args.out_dir / "imbalance.png")
    plot_depth_heatmap(df, args.out_dir / "depth_heatmap.png")
    print(f"Wrote preview files to {args.out_dir}")


def depth_frame(depths) -> pd.DataFrame:
    rows = []
    for depth in depths:
        bid0 = depth.bids[0]
        ask0 = depth.asks[0]
        bid_price = float(str(bid0.price)) if float(str(bid0.size)) > 0 else None
        ask_price = float(str(ask0.price)) if float(str(ask0.size)) > 0 else None
        bid_size = float(str(bid0.size))
        ask_size = float(str(ask0.size))
        bid_depth = sum(float(str(order.size)) for order in depth.bids)
        ask_depth = sum(float(str(order.size)) for order in depth.asks)
        mid = (bid_price + ask_price) / 2 if bid_price and ask_price else None
        spread = ask_price - bid_price if bid_price and ask_price else None
        microprice = (
            (ask_price * bid_size + bid_price * ask_size) / (bid_size + ask_size)
            if bid_price and ask_price and bid_size + ask_size > 0
            else None
        )
        imbalance = (
            (bid_depth - ask_depth) / (bid_depth + ask_depth)
            if bid_depth + ask_depth > 0
            else None
        )
        row = {
            "ts_event": depth.ts_event,
            "time": pd.to_datetime(depth.ts_event, unit="ns", utc=True),
            "bid": bid_price,
            "ask": ask_price,
            "mid": mid,
            "spread": spread,
            "microprice": microprice,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "imbalance": imbalance,
        }
        for i in range(10):
            row[f"bid_size_{i}"] = float(str(depth.bids[i].size))
            row[f"ask_size_{i}"] = float(str(depth.asks[i].size))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ts_event")


def plot_price_and_spread(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(df["time"], df["bid"], label="bid", linewidth=0.9)
    axes[0].plot(df["time"], df["ask"], label="ask", linewidth=0.9)
    axes[0].plot(df["time"], df["mid"], label="mid", linewidth=1.1)
    axes[0].plot(df["time"], df["microprice"], label="microprice", linewidth=1.1)
    axes[0].legend(loc="best")
    axes[0].set_title("Top of Book")
    axes[1].plot(df["time"], df["spread"], color="tab:red", linewidth=1.0)
    axes[1].set_title("Spread")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_imbalance(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["time"], df["imbalance"], linewidth=1.0)
    axes[0].set_title("Depth Imbalance")
    axes[1].hist(df["imbalance"].dropna(), bins=50)
    axes[1].set_title("Imbalance Distribution")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_depth_heatmap(df: pd.DataFrame, path: Path) -> None:
    cols = [f"bid_size_{i}" for i in range(10)] + [f"ask_size_{i}" for i in range(10)]
    matrix = df[cols].tail(400).T
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title("Top-10 Depth Size Heatmap")
    fig.colorbar(image, ax=ax, label="size")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
