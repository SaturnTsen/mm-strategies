#!/usr/bin/env python3

from __future__ import annotations

import os
import re
from ast import literal_eval
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nautilus_trader.model.data import OrderBookDepth10


def book_depth10_to_frame(book_data: list[object]) -> pd.DataFrame:
    rows = []
    for item in book_data:
        if not isinstance(item, OrderBookDepth10):
            raise TypeError(f"Expected OrderBookDepth10, got {type(item).__name__}")
        bid = item.bids[0].price.as_double()
        ask = item.asks[0].price.as_double()
        rows.append(
            {
                "ts_event": pd.Timestamp(item.ts_event, unit="ns", tz="UTC"),
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2.0,
            },
        )
    if not rows:
        raise ValueError("book_data is empty")
    return pd.DataFrame(rows)


def trades_to_frame(trades: list[object]) -> pd.DataFrame:
    rows = [
        {
            "ts_event": pd.Timestamp(item.ts_event, unit="ns", tz="UTC"),
            "price": item.price.as_double(),
            "size": item.size.as_double(),
            "aggressor_side": str(item.aggressor_side),
        }
        for item in trades
    ]
    return pd.DataFrame(rows, columns=["ts_event", "price", "size", "aggressor_side"])


def quote_records_to_frame(records: list[dict[str, float | int]]) -> pd.DataFrame:
    quotes = pd.DataFrame(records)
    if quotes.empty:
        raise ValueError("quote records are empty")
    quotes["ts_event"] = pd.to_datetime(quotes["ts_event"], unit="ns", utc=True)
    return quotes.sort_values("ts_event")


def initial_equity_usdt(initial_balances: dict[str, float], first_mid: float) -> float:
    return initial_balances.get("USDT", 0.0) + initial_balances.get("BTC", 0.0) * first_mid


def normalize_account_report(account_report: pd.DataFrame) -> pd.DataFrame:
    account = account_report.copy()
    if account.empty:
        raise ValueError("account report is empty")
    account.index.name = account.index.name or "ts_event"
    account = account.reset_index().rename(columns={account.index.name or "index": "ts_event"})
    if "ts_event" not in account.columns:
        account = account.rename(columns={account.columns[0]: "ts_event"})
    account["ts_event"] = pd.to_datetime(account["ts_event"], utc=True)
    account["total"] = account["total"].astype(float)
    return account


def pnl_frame(account_report: pd.DataFrame, book_frame: pd.DataFrame, initial_balances: dict[str, float]) -> pd.DataFrame:
    account = normalize_account_report(account_report)
    balances = (
        account.pivot_table(index="ts_event", columns="currency", values="total", aggfunc="last")
        .sort_index()
        .ffill()
        .fillna(0.0)
    )
    if "USDT" not in balances.columns:
        balances["USDT"] = 0.0
    if "BTC" not in balances.columns:
        balances["BTC"] = 0.0

    marks = book_frame[["ts_event", "mid"]].sort_values("ts_event")
    values = pd.merge_asof(
        balances.reset_index().sort_values("ts_event"),
        marks,
        on="ts_event",
        direction="backward",
    ).ffill()
    first_mid = float(book_frame["mid"].iloc[0])
    values["equity_usdt"] = values["USDT"] + values["BTC"] * values["mid"]
    values["pnl_usdt"] = values["equity_usdt"] - initial_equity_usdt(initial_balances, first_mid)
    values["initial_inventory_pnl_usdt"] = initial_balances.get("BTC", 0.0) * (values["mid"] - first_mid)
    values["trading_pnl_usdt"] = values["pnl_usdt"] - values["initial_inventory_pnl_usdt"]
    values["return_pct"] = values["equity_usdt"] / values["equity_usdt"].iloc[0] - 1.0
    values["drawdown_pct"] = 1.0 - values["equity_usdt"] / values["equity_usdt"].cummax()
    values["trading_drawdown_pct"] = 1.0 - (
        values["trading_pnl_usdt"] / values["equity_usdt"].iloc[0] + 1.0
    ) / (values["trading_pnl_usdt"] / values["equity_usdt"].iloc[0] + 1.0).cummax()
    return values


def normalize_fills_report(fills_report: pd.DataFrame) -> pd.DataFrame:
    fills = fills_report.copy()
    if fills.empty:
        return pd.DataFrame(
            columns=[
                "ts_last",
                "side",
                "filled_qty",
                "avg_px",
                "notional",
                "commission_usdt",
                "signed_qty",
            ],
        )
    for column in ("ts_last", "side", "filled_qty", "avg_px", "commissions"):
        if column not in fills.columns:
            raise ValueError(f"fills_report missing required column: {column}")
    fills["ts_last"] = pd.to_datetime(fills["ts_last"], utc=True)
    fills["filled_qty"] = fills["filled_qty"].astype(float)
    fills["avg_px"] = fills["avg_px"].astype(float)
    fills = fills[fills["filled_qty"] > 0.0].copy()
    fills["notional"] = fills["filled_qty"] * fills["avg_px"]
    fills["commission_usdt"] = fills["commissions"].map(parse_commission_usdt)
    fills["signed_qty"] = np.where(fills["side"] == "BUY", fills["filled_qty"], -fills["filled_qty"])
    return fills.sort_values("ts_last")


def parse_commission_usdt(value: object) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, list):
        items = value
    else:
        text = str(value)
        if text in ("", "[]"):
            return 0.0
        items = literal_eval(text)
    total = 0.0
    for item in items:
        amount, currency = str(item).split()
        if currency != "USDT":
            raise ValueError(f"Unsupported commission currency: {currency}")
        total += float(amount)
    return total


def normalize_positions_report(positions_report: pd.DataFrame) -> pd.DataFrame:
    positions = positions_report.copy()
    if positions.empty:
        return pd.DataFrame(
            columns=[
                "ts_opened",
                "ts_closed",
                "quantity",
                "avg_px_open",
                "avg_px_close",
                "realized_return",
                "realized_pnl_usdt",
                "is_closed",
            ],
        )
    for column in ("ts_opened", "quantity", "avg_px_open", "realized_pnl"):
        if column not in positions.columns:
            raise ValueError(f"positions_report missing required column: {column}")
    positions["ts_opened"] = pd.to_datetime(positions["ts_opened"], utc=True)
    positions["ts_closed"] = pd.to_datetime(positions.get("ts_closed"), utc=True)
    positions["quantity"] = positions["quantity"].astype(float)
    positions["avg_px_open"] = positions["avg_px_open"].astype(float)
    positions["avg_px_close"] = pd.to_numeric(positions.get("avg_px_close"), errors="coerce")
    positions["realized_return"] = pd.to_numeric(positions.get("realized_return"), errors="coerce")
    positions["realized_pnl_usdt"] = positions["realized_pnl"].map(parse_money_usdt)
    positions["is_closed"] = positions["ts_closed"].notna()
    return positions


def parse_money_usdt(value: object) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value)
    if text in ("", "nan", "None"):
        return 0.0
    amount, currency = text.split()
    if currency != "USDT":
        raise ValueError(f"Unsupported money currency: {currency}")
    return float(amount)


def compute_drawdown_duration(drawdown: pd.Series) -> tuple[pd.Timedelta | float, pd.Timedelta | float]:
    underwater = drawdown > 0.0
    if not underwater.any():
        return np.nan, np.nan
    starts = drawdown.index[underwater & ~underwater.shift(fill_value=False)]
    ends = drawdown.index[~underwater & underwater.shift(fill_value=False)]
    if len(ends) < len(starts):
        ends = ends.append(pd.Index([drawdown.index[-1]]))
    durations = pd.Series(ends.values - starts.values)
    return durations.max(), durations.mean()


def compute_backtest_stats(
    pnl: pd.DataFrame,
    book_frame: pd.DataFrame,
    fills: pd.DataFrame,
    positions: pd.DataFrame,
    initial_balances: dict[str, float],
) -> pd.Series:
    values = pnl.set_index("ts_event").sort_index()
    equity = values["equity_usdt"]
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    duration = values.index[-1] - values.index[0]
    initial_equity = float(equity.iloc[0])
    final_equity = float(equity.iloc[-1])
    drawdown = values["drawdown_pct"]
    max_dd_duration, avg_dd_duration = compute_drawdown_duration(drawdown)
    closed_positions = positions[positions["is_closed"]] if not positions.empty else positions
    trade_returns = closed_positions["realized_return"].dropna() if not closed_positions.empty else pd.Series(dtype=float)
    trade_pnl = closed_positions["realized_pnl_usdt"].dropna() if not closed_positions.empty else pd.Series(dtype=float)
    win_rate = float((trade_pnl > 0.0).mean() * 100.0) if len(trade_pnl) else np.nan
    gross_profit = trade_pnl[trade_pnl > 0.0].sum()
    gross_loss = -trade_pnl[trade_pnl < 0.0].sum()
    inventory_delta = (values["BTC"] - initial_balances.get("BTC", 0.0)).abs()
    exposure = float((inventory_delta > 1e-12).mean() * 100.0)
    first_mid = float(book_frame["mid"].iloc[0])
    last_mid = float(book_frame["mid"].iloc[-1])
    annualized_return, annualized_volatility, sharpe = annualized_metrics(returns, duration)

    stats = pd.Series(dtype=object)
    stats.loc["Start"] = values.index[0]
    stats.loc["End"] = values.index[-1]
    stats.loc["Duration"] = duration
    stats.loc["Initial USDT"] = initial_balances.get("USDT", 0.0)
    stats.loc["Initial BTC"] = initial_balances.get("BTC", 0.0)
    stats.loc["Initial Mark"] = first_mid
    stats.loc["Equity Initial [USDT]"] = initial_equity
    stats.loc["Equity Final [USDT]"] = final_equity
    stats.loc["Equity Peak [USDT]"] = float(equity.max())
    stats.loc["Return [%]"] = (final_equity / initial_equity - 1.0) * 100.0
    stats.loc["Trading PnL [USDT]"] = float(values["trading_pnl_usdt"].iloc[-1])
    stats.loc["Initial Inventory PnL [USDT]"] = float(values["initial_inventory_pnl_usdt"].iloc[-1])
    stats.loc["Buy & Hold Return [%]"] = (last_mid / first_mid - 1.0) * 100.0
    stats.loc["Return (Ann.) [%]"] = annualized_return * 100.0
    stats.loc["Volatility (Ann.) [%]"] = annualized_volatility * 100.0
    stats.loc["Sharpe Ratio"] = sharpe
    stats.loc["Max. Drawdown [%]"] = -float(drawdown.max()) * 100.0
    stats.loc["Avg. Drawdown [%]"] = -float(drawdown[drawdown > 0.0].mean()) * 100.0
    stats.loc["Max. Drawdown Duration"] = max_dd_duration
    stats.loc["Avg. Drawdown Duration"] = avg_dd_duration
    stats.loc["Exposure Time [%]"] = exposure
    stats.loc["# Orders Filled"] = len(fills)
    stats.loc["# Positions"] = len(positions)
    stats.loc["# Closed Positions"] = int(positions["is_closed"].sum()) if not positions.empty else 0
    stats.loc["Open Positions"] = int((~positions["is_closed"]).sum()) if not positions.empty else 0
    stats.loc["Filled Notional [USDT]"] = float(fills["notional"].sum()) if not fills.empty else 0.0
    stats.loc["Commissions [USDT]"] = float(fills["commission_usdt"].sum()) if not fills.empty else 0.0
    stats.loc["Win Rate [%]"] = win_rate
    stats.loc["Best Trade [%]"] = float(trade_returns.max() * 100.0) if len(trade_returns) else np.nan
    stats.loc["Worst Trade [%]"] = float(trade_returns.min() * 100.0) if len(trade_returns) else np.nan
    stats.loc["Avg. Trade [%]"] = float(trade_returns.mean() * 100.0) if len(trade_returns) else np.nan
    stats.loc["Profit Factor"] = float(gross_profit / gross_loss) if gross_loss > 0.0 else np.nan
    stats.loc["SQN"] = float(np.sqrt(len(trade_pnl)) * trade_pnl.mean() / trade_pnl.std()) if len(trade_pnl) > 1 and trade_pnl.std() else np.nan
    return stats


def annualized_metrics(returns: pd.Series, duration: pd.Timedelta) -> tuple[float, float, float]:
    if returns.empty or duration.total_seconds() <= 0.0:
        return np.nan, np.nan, np.nan
    seconds_per_year = 365.0 * 24.0 * 60.0 * 60.0
    periods_per_year = len(returns) * seconds_per_year / duration.total_seconds()
    total_return = (returns + 1.0).prod() - 1.0
    annualized_return = (1.0 + total_return) ** (periods_per_year / len(returns)) - 1.0
    annualized_volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = annualized_return / annualized_volatility if annualized_volatility > 0.0 else np.nan
    return float(annualized_return), float(annualized_volatility), float(sharpe)


def plot_time_price_quote(
    book_frame: pd.DataFrame,
    quotes: pd.DataFrame,
    output_path: Path,
    initial_text: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(book_frame["ts_event"], book_frame["mid"], color="#1f77b4", linewidth=1.0, label="mid")
    ax.plot(quotes["ts_event"], quotes["bid_quote"], color="#2ca02c", linewidth=0.8, label="bid quote")
    ax.plot(quotes["ts_event"], quotes["ask_quote"], color="#d62728", linewidth=0.8, label="ask quote")
    ax.fill_between(
        book_frame["ts_event"],
        book_frame["bid"],
        book_frame["ask"],
        color="#1f77b4",
        alpha=0.12,
        label="top of book",
    )
    ax.text(
        0.01,
        0.98,
        initial_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    ax.set_title("Time Price Quote")
    ax.set_xlabel("time")
    ax.set_ylabel("price")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_trades(
    book_frame: pd.DataFrame,
    trades: pd.DataFrame,
    fills: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(book_frame["ts_event"], book_frame["mid"], color="#1f77b4", linewidth=1.0, label="mid")
    if not trades.empty:
        step = max(1, len(trades) // 50_000)
        sampled = trades.iloc[::step]
        ax.scatter(sampled["ts_event"], sampled["price"], s=2, color="#888888", alpha=0.25, label="market trades")
    if not fills.empty:
        fills = fills.copy()
        fills["ts_last"] = pd.to_datetime(fills["ts_last"], utc=True)
        fills["avg_px"] = fills["avg_px"].astype(float)
        buy = fills[fills["side"] == "BUY"]
        sell = fills[fills["side"] == "SELL"]
        ax.scatter(buy["ts_last"], buy["avg_px"], s=70, marker="^", color="#2ca02c", label="buy fills")
        ax.scatter(sell["ts_last"], sell["avg_px"], s=70, marker="v", color="#d62728", label="sell fills")
    ax.set_title("Trades")
    ax.set_xlabel("time")
    ax.set_ylabel("price")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_pnl(pnl: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(pnl["ts_event"], pnl["pnl_usdt"], color="#111111", linewidth=1.2, label="total equity PnL")
    ax.plot(
        pnl["ts_event"],
        pnl["initial_inventory_pnl_usdt"],
        color="#1f77b4",
        linewidth=1.0,
        label="initial inventory PnL",
    )
    ax.plot(
        pnl["ts_event"],
        pnl["trading_pnl_usdt"],
        color="#d62728",
        linewidth=1.0,
        label="trading PnL",
    )
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_title("PnL")
    ax.set_xlabel("time")
    ax.set_ylabel("USDT")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def sampled_market_trades(trades: pd.DataFrame, max_points: int = 50_000) -> pd.DataFrame:
    if trades.empty:
        return trades
    step = max(1, len(trades) // max_points)
    return trades.iloc[::step].copy()


def stats_table_values(stats: pd.Series) -> tuple[list[str], list[str]]:
    return list(stats.index.astype(str)), [format_stat_value(value) for value in stats.values]


def format_stat_value(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.8f}".rstrip("0").rstrip(".")
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    return str(value)


def add_fill_markers(fig: go.Figure, fills: pd.DataFrame, row: int, col: int) -> None:
    if fills.empty:
        return
    buy = fills[fills["side"] == "BUY"]
    sell = fills[fills["side"] == "SELL"]
    if not buy.empty:
        fig.add_trace(
            go.Scattergl(
                x=buy["ts_last"],
                y=buy["avg_px"],
                mode="markers",
                name="buy fills",
                marker={"symbol": "triangle-up", "size": 11, "color": "#2ca02c", "line": {"color": "black", "width": 0.6}},
                customdata=np.column_stack([buy["filled_qty"], buy["notional"], buy["commission_usdt"]]),
                hovertemplate="buy<br>%{x}<br>price=%{y:.2f}<br>qty=%{customdata[0]:.8f}<br>notional=%{customdata[1]:.2f}<br>fee=%{customdata[2]:.6f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
    if not sell.empty:
        fig.add_trace(
            go.Scattergl(
                x=sell["ts_last"],
                y=sell["avg_px"],
                mode="markers",
                name="sell fills",
                marker={"symbol": "triangle-down", "size": 11, "color": "#d62728", "line": {"color": "black", "width": 0.6}},
                customdata=np.column_stack([sell["filled_qty"], sell["notional"], sell["commission_usdt"]]),
                hovertemplate="sell<br>%{x}<br>price=%{y:.2f}<br>qty=%{customdata[0]:.8f}<br>notional=%{customdata[1]:.2f}<br>fee=%{customdata[2]:.6f}<extra></extra>",
            ),
            row=row,
            col=col,
        )


def write_interactive_report(
    output_path: Path,
    book_frame: pd.DataFrame,
    market_trades: pd.DataFrame,
    quotes: pd.DataFrame,
    fills: pd.DataFrame,
    pnl: pd.DataFrame,
    stats: pd.Series,
) -> None:
    stat_names, stat_values = stats_table_values(stats)
    table_height = max(520, 30 * (len(stat_names) + 1))
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.34, 0.24, 0.16, 0.09, 0.085, 0.085],
        specs=[
            [{"type": "table"}],
            [{"type": "xy"}],
            [{"type": "xy", "secondary_y": True}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
        ],
        subplot_titles=("", "Price / Quotes / Fills", "Equity and PnL", "Drawdown", "Inventory", "Quote Size"),
    )
    fig.add_trace(
        go.Table(
            header={"values": ["Metric", "Value"], "fill_color": "#111827", "font": {"color": "white", "size": 13}, "align": "left", "height": 30},
            cells={"values": [stat_names, stat_values], "fill_color": "#f8fafc", "font": {"color": "#111827", "size": 12}, "align": "left", "height": 26},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(x=book_frame["ts_event"], y=book_frame["mid"], mode="lines", name="mid", line={"color": "#1f77b4", "width": 1.2}),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(x=book_frame["ts_event"], y=book_frame["bid"], mode="lines", name="best bid", line={"color": "#6aa84f", "width": 0.7}),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(x=book_frame["ts_event"], y=book_frame["ask"], mode="lines", name="best ask", line={"color": "#cc0000", "width": 0.7}),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(x=quotes["ts_event"], y=quotes["bid_quote"], mode="lines", name="strategy bid", line={"color": "#2ca02c", "width": 1.0}),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(x=quotes["ts_event"], y=quotes["ask_quote"], mode="lines", name="strategy ask", line={"color": "#d62728", "width": 1.0}),
        row=2,
        col=1,
    )
    sampled_trades = sampled_market_trades(market_trades)
    if not sampled_trades.empty:
        fig.add_trace(
            go.Scattergl(
                x=sampled_trades["ts_event"],
                y=sampled_trades["price"],
                mode="markers",
                name="market trades",
                marker={"size": 3, "color": "rgba(90,90,90,0.25)"},
                customdata=np.column_stack([sampled_trades["size"], sampled_trades["aggressor_side"]]),
                hovertemplate="market trade<br>%{x}<br>price=%{y:.2f}<br>size=%{customdata[0]:.8f}<br>side=%{customdata[1]}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    add_fill_markers(fig, fills, 2, 1)

    fig.add_trace(
        go.Scattergl(x=pnl["ts_event"], y=pnl["equity_usdt"], mode="lines", name="equity", line={"color": "#111111", "width": 1.3}),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scattergl(x=pnl["ts_event"], y=pnl["pnl_usdt"], mode="lines", name="total PnL", line={"color": "#6366f1", "width": 1.0}),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scattergl(x=pnl["ts_event"], y=pnl["initial_inventory_pnl_usdt"], mode="lines", name="initial inventory PnL", line={"color": "#1f77b4", "width": 1.0}),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scattergl(x=pnl["ts_event"], y=pnl["trading_pnl_usdt"], mode="lines", name="trading PnL", line={"color": "#d62728", "width": 1.0}),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scattergl(x=pnl["ts_event"], y=-100.0 * pnl["drawdown_pct"], mode="lines", name="drawdown", fill="tozeroy", line={"color": "#ef4444", "width": 1.0}),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(x=pnl["ts_event"], y=pnl["BTC"], mode="lines", name="BTC balance", line={"color": "#0f766e", "width": 1.0}),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(x=quotes["ts_event"], y=quotes["inventory"], mode="lines", name="strategy inventory", line={"color": "#f97316", "width": 1.0}),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(x=quotes["ts_event"], y=quotes["bid_size"], mode="lines", name="bid size", line={"color": "#2ca02c", "width": 1.0}),
        row=6,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(x=quotes["ts_event"], y=quotes["ask_size"], mode="lines", name="ask size", line={"color": "#d62728", "width": 1.0}),
        row=6,
        col=1,
    )
    fig.update_layout(
        template="plotly_white",
        height=table_height + 1180,
        width=1600,
        autosize=False,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
        margin={"l": 70, "r": 30, "t": 70, "b": 40},
        title={"text": "Nautilus Backtest Report", "x": 0.01},
    )
    fig.update_yaxes(title_text="price", row=2, col=1)
    fig.update_yaxes(title_text="equity USDT", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="PnL USDT", row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="drawdown %", row=4, col=1)
    fig.update_yaxes(title_text="BTC", row=5, col=1)
    fig.update_yaxes(title_text="BTC", row=6, col=1)
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    html = center_plotly_html(html, max_width=1600)
    output_path.write_text(html, encoding="utf-8")


def center_plotly_html(html: str, max_width: int) -> str:
    style = (
        "<style>"
        "body{margin:0;background:#f3f4f6;}"
        f".plotly-graph-div{{max-width:{max_width}px;margin:0 auto;box-sizing:border-box;}}"
        "</style>"
    )
    html = html.replace("</head>", f"{style}</head>")
    html = re.sub(
        r'<div style="height:([^;]+); width:[^;]+;">',
        rf'<div style="height:\1; width:100%; max-width:{max_width}px; margin:0 auto;">',
        html,
        count=1,
    )
    return re.sub(
        r'<div id="([^"]+)" class="plotly-graph-div" style="height:([^;]+); width:([^;]+);">',
        r'<div id="\1" class="plotly-graph-div" style="height:\2; width:100%;">',
        html,
        count=1,
    )


def visualize_backtest(
    book_data: list[object],
    trades: list[object],
    quote_records: list[dict[str, float | int]],
    fills_report: pd.DataFrame,
    account_report: pd.DataFrame,
    positions_report: pd.DataFrame,
    output_dir: Path,
    label: str,
    initial_balances: dict[str, float],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    book_frame = book_depth10_to_frame(book_data)
    trade_frame = trades_to_frame(trades)
    quote_frame = quote_records_to_frame(quote_records)
    fills = normalize_fills_report(fills_report)
    positions = normalize_positions_report(positions_report)
    pnl = pnl_frame(account_report, book_frame, initial_balances)
    stats = compute_backtest_stats(pnl, book_frame, fills, positions, initial_balances)

    quotes_path = output_dir / f"{label}_quotes.csv"
    pnl_path = output_dir / f"{label}_pnl.csv"
    stats_path = output_dir / f"{label}_stats.csv"
    report_html_path = output_dir / f"{label}_report.html"
    time_price_quote_path = output_dir / f"{label}_time_price_quote.png"
    trade_path = output_dir / f"{label}_trade.png"
    pnl_plot_path = output_dir / f"{label}_pnl.png"

    quote_frame.to_csv(quotes_path, index=False)
    pnl.to_csv(pnl_path, index=False)
    stats.to_frame("value").to_csv(stats_path)

    first_mid = float(book_frame["mid"].iloc[0])
    initial_text = (
        f"initial USDT={initial_balances.get('USDT', 0.0):.8f}\n"
        f"initial BTC={initial_balances.get('BTC', 0.0):.8f}\n"
        f"initial mark={first_mid:.2f}\n"
        f"initial equity={initial_equity_usdt(initial_balances, first_mid):.8f} USDT"
    )
    plot_time_price_quote(book_frame, quote_frame, time_price_quote_path, initial_text)
    plot_trades(book_frame, trade_frame, fills, trade_path)
    plot_pnl(pnl, pnl_plot_path)
    write_interactive_report(report_html_path, book_frame, trade_frame, quote_frame, fills, pnl, stats)

    return {
        "quotes_path": quotes_path,
        "pnl_path": pnl_path,
        "stats_path": stats_path,
        "report_html_path": report_html_path,
        "time_price_quote_path": time_price_quote_path,
        "trade_path": trade_path,
        "pnl_plot_path": pnl_plot_path,
    }
