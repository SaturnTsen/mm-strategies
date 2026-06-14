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

from data_utils import discover_instruments
from data_utils import depth_snapshot_to_long
from data_utils import depths_to_top
from data_utils import load_depths
from data_utils import load_quotes
from data_utils import load_trades


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

    depths = load_depths(catalog, instrument)
    quotes = load_quotes(catalog, instrument)
    trades = load_trades(catalog, instrument)
    top = depths_to_top(depths)
    depth = depth_snapshot_to_long(depths)

    top.to_csv(out_dir / "reconstructed_top_of_book.csv", index=False)
    depth.sort_values(["side", "price"]).to_csv(out_dir / "last_depth_snapshot.csv", index=False)

    summary = {
        "catalog": catalog,
        "instrument": instrument,
        "depth snapshots": len(depths),
        "quotes": len(quotes),
        "trades": len(trades),
        "top snapshots": len(top),
        "start": min(depths["dt"].min(), quotes["dt"].min(), trades["dt"].min()),
        "end": max(depths["dt"].max(), quotes["dt"].max(), trades["dt"].max()),
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
