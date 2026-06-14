#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash import dash_table
from plotly.subplots import make_subplots

from data_utils import (
    depth_snapshot_to_long,
    depths_to_top,
    discover_instruments,
    load_depths,
    load_quotes,
    load_trades,
)


DEFAULT_CATALOG_PATH = Path("catalog")
DEFAULT_PORT = 5007
DEFAULT_HOST = "127.0.0.1"
MAX_TABLE_ROWS = 200
DEFAULT_ANIMATION_INTERVAL_MS = 750
PRICE_STACK_HEIGHT = 320


@lru_cache(maxsize=16)
def load_instrument_data(catalog: str, instrument: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    catalog_path = Path(catalog)
    depths = load_depth_data(catalog, instrument)
    quotes = load_quotes(catalog_path, instrument)
    trades = load_trades(catalog_path, instrument)
    top = depths_to_top(depths)
    return quotes, trades, top, depths


@lru_cache(maxsize=16)
def load_depth_data(catalog: str, instrument: str) -> pd.DataFrame:
    return load_depths(Path(catalog), instrument)


@lru_cache(maxsize=128)
def reconstruct_depth_at(catalog: str, instrument: str, end_ms: int) -> pd.DataFrame:
    depths = load_depth_data(catalog, instrument)
    depths_until_end = depths[depths["dt"] <= from_ms(end_ms)]
    if depths_until_end.empty:
        raise ValueError(f"No order book depth before selected end time for {instrument}")
    return depth_snapshot_to_long(depths_until_end)


def to_ms(value: pd.Timestamp) -> int:
    timestamp = value.tz_convert("UTC") if value.tzinfo is not None else value.tz_localize("UTC")
    return timestamp.value // 1_000_000


def from_ms(value: int | float) -> pd.Timestamp:
    return pd.to_datetime(int(value), unit="ms", utc=True)


def format_utc_ms(value: int) -> str:
    return from_ms(value).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def parse_utc_ms(value: str) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return to_ms(timestamp)


def clamp_ms(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def normalized_range(start_ms: int, end_ms: int, lower: int, upper: int) -> list[int]:
    start = clamp_ms(start_ms, lower, upper)
    end = clamp_ms(end_ms, lower, upper)
    if start >= end:
        raise ValueError("Start time must be less than end time")
    return [start, end]


def selected_instruments(value: list[str] | str | None) -> list[str]:
    if value is None or value == []:
        raise ValueError("At least one instrument is required")
    if isinstance(value, str):
        return [value]
    return value


def parse_positive_float(value: object, name: str) -> float:
    if value is None or value == "":
        raise ValueError(f"{name} is required")
    parsed = float(value) # type: ignore
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def plot_dt(frame: pd.DataFrame) -> pd.Series:
    return frame["dt"].dt.tz_localize(None)


def filter_time(frame: pd.DataFrame, time_range: list[int] | tuple[int, int]) -> pd.DataFrame:
    start = from_ms(time_range[0])
    end = from_ms(time_range[1])
    return frame[(frame["dt"] >= start) & (frame["dt"] <= end)]


def time_bounds(catalog: str, instrument: str) -> tuple[int, int]:
    quotes, trades, top, _ = load_instrument_data(catalog, instrument)
    start = min(quotes["dt"].min(), trades["dt"].min(), top["dt"].min())
    end = max(quotes["dt"].max(), trades["dt"].max(), top["dt"].max())
    return to_ms(start), to_ms(end)


def instruments_time_bounds(catalog: str, instruments: list[str]) -> tuple[int, int]:
    bounds = [time_bounds(catalog, instrument) for instrument in instruments]
    return min(start for start, _ in bounds), max(end for _, end in bounds)


def valid_window_bounds(catalog: str, instrument: str) -> tuple[int, int]:
    quotes, _, top, _ = load_instrument_data(catalog, instrument)
    start = max(quotes["dt"].min(), top["dt"].min())
    end = min(quotes["dt"].max(), top["dt"].max())
    return to_ms(start), to_ms(end)


def instruments_valid_window_bounds(catalog: str, instruments: list[str]) -> tuple[int, int]:
    bounds = [valid_window_bounds(catalog, instrument) for instrument in instruments]
    start = max(start for start, _ in bounds)
    end = min(end for _, end in bounds)
    if start >= end:
        raise ValueError("Selected instruments have no common valid animation window")
    return start, end


def slider_marks(start_ms: int, end_ms: int) -> dict[int, str]:
    start = from_ms(start_ms)
    end = from_ms(end_ms)
    middle = start + (end - start) / 2
    return {
        start_ms: start.strftime("%H:%M:%S"),
        to_ms(middle): middle.strftime("%H:%M:%S"),
        end_ms: end.strftime("%H:%M:%S"),
    }


def figure_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 48, "r": 24, "t": 48, "b": 42},
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    return fig


def render_summary(catalog: str, instrument: str, time_range: list[int]) -> html.Div:
    quotes, trades, top, _ = load_instrument_data(catalog, instrument)
    quotes_window = filter_time(quotes, time_range)
    trades_window = filter_time(trades, time_range)
    top_window = filter_time(top, time_range)

    if quotes_window.empty or top_window.empty:
        return html.Div(
            [
                html.Div([html.Div("instrument", className="summary-key"), html.Div(instrument, className="summary-value")]),
                html.Div([html.Div("window", className="summary-key"), html.Div("no quote/top data", className="summary-value")]),
                html.Div([html.Div("start", className="summary-key"), html.Div(str(from_ms(time_range[0])), className="summary-value")]),
                html.Div([html.Div("end", className="summary-key"), html.Div(str(from_ms(time_range[1])), className="summary-value")]),
            ],
            className="summary-grid",
        )

    values = [
        ("instrument", instrument),
        ("quotes", len(quotes_window)),
        ("trades", len(trades_window)),
        ("top snapshots", len(top_window)),
        ("depth levels", int(top_window["bid_levels"].iloc[-1] + top_window["ask_levels"].iloc[-1])),
        ("start", str(top_window["dt"].min())),
        ("end", str(top_window["dt"].max())),
        ("last bid", f"{top_window['best_bid'].iloc[-1]:.8g}"),
        ("last ask", f"{top_window['best_ask'].iloc[-1]:.8g}"),
        ("last spread bps", f"{top_window['spread_bps'].iloc[-1]:.4f}"),
    ]
    return html.Div(
        [html.Div([html.Div(key, className="summary-key"), html.Div(value, className="summary-value")]) for key, value in values],
        className="summary-grid",
    )


def render_price_figure(catalog: str, instrument: str, time_range: list[int]) -> go.Figure:
    _, trades, top, _ = load_instrument_data(catalog, instrument)
    top_window = filter_time(top, time_range)
    trades_window = filter_time(trades, time_range)
    if top_window.empty:
        raise ValueError(f"No top-of-book data in selected window for {instrument}")

    fig = go.Figure()
    for column, color in [("best_bid", "#1f7a4d"), ("best_ask", "#b33b3b"), ("mid", "#245c7a")]:
        fig.add_trace(
            go.Scattergl(
                x=plot_dt(top_window),
                y=top_window[column],
                mode="lines",
                name=column,
                line={"width": 1.2, "color": color},
            )
        )

    if not trades_window.empty:
        fig.add_trace(
            go.Scattergl(
                x=plot_dt(trades_window),
                y=trades_window["price_f"],
                mode="markers",
                name="trades",
                marker={
                    "size": 5,
                    "color": trades_window["size_f"],
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "size"},
                    "opacity": 0.6,
                },
            )
        )

    fig.update_xaxes(title_text="time")
    fig.update_yaxes(title_text="price")
    fig.update_layout(title=f"{instrument} top of book")
    return figure_layout(fig, 520)


def render_price_stack_figure(catalog: str, instruments: list[str], time_range: list[int]) -> go.Figure:
    fig = make_subplots(
        rows=len(instruments),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=instruments,
    )
    for row, instrument in enumerate(instruments, start=1):
        _, trades, top, _ = load_instrument_data(catalog, instrument)
        top_window = filter_time(top, time_range)
        trades_window = filter_time(trades, time_range)
        if top_window.empty:
            raise ValueError(f"No top-of-book data in selected window for {instrument}")

        for column, color in [("best_bid", "#1f7a4d"), ("best_ask", "#b33b3b"), ("mid", "#245c7a")]:
            fig.add_trace(
                go.Scattergl(
                    x=plot_dt(top_window),
                    y=top_window[column],
                    mode="lines",
                    name=column,
                    legendgroup=column,
                    showlegend=row == 1,
                    line={"width": 1.2, "color": color},
                ),
                row=row,
                col=1,
            )

        if not trades_window.empty:
            fig.add_trace(
                go.Scattergl(
                    x=plot_dt(trades_window),
                    y=trades_window["price_f"],
                    mode="markers",
                    name="trades",
                    legendgroup="trades",
                    showlegend=row == 1,
                    marker={"size": 5, "color": "#7c3aed", "opacity": 0.55},
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(title_text="price", row=row, col=1)

    fig.update_xaxes(title_text="time", row=len(instruments), col=1)
    fig.update_layout(title="Selected instruments top of book")
    return figure_layout(fig, max(420, PRICE_STACK_HEIGHT * len(instruments)))


def render_microstructure_figure(catalog: str, instrument: str, time_range: list[int]) -> go.Figure:
    _, _, top, _ = load_instrument_data(catalog, instrument)
    top_window = filter_time(top, time_range)
    if top_window.empty:
        raise ValueError(f"No top-of-book data in selected window for {instrument}")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("spread", "top level imbalance"))
    fig.add_trace(
        go.Scattergl(x=plot_dt(top_window), y=top_window["spread_bps"], mode="lines", name="spread bps"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(x=plot_dt(top_window), y=top_window["top_imbalance"], mode="lines", name="imbalance"),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line_width=1, line_color="#555", opacity=0.45, row=2, col=1) # type: ignore
    fig.update_xaxes(title_text="time", row=2, col=1)
    fig.update_yaxes(title_text="bps", row=1, col=1)
    fig.update_yaxes(title_text="imbalance", row=2, col=1)
    return figure_layout(fig, 650)


def render_activity_figure(catalog: str, instrument: str, time_range: list[int]) -> go.Figure:
    quotes, trades, top, _ = load_instrument_data(catalog, instrument)
    quotes_window = filter_time(quotes, time_range)
    trades_window = filter_time(trades, time_range)
    top_window = filter_time(top, time_range)
    if quotes_window.empty or top_window.empty:
        raise ValueError(f"No activity data in selected window for {instrument}")

    quote_counts = quotes_window.set_index("dt")["mid_f"].resample("1s").count().rename("quotes")
    trade_counts = trades_window.set_index("dt")["price_f"].resample("1s").count().rename("trades")
    counts = pd.concat([quote_counts, trade_counts], axis=1).fillna(0.0).reset_index()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("event rate", "book levels"))
    for column in ["quotes", "trades"]:
        fig.add_trace(
            go.Scattergl(x=counts["dt"].dt.tz_localize(None), y=counts[column], mode="lines", name=column, line_shape="hv"),
            row=1,
            col=1,
        )
    for column in ["bid_levels", "ask_levels"]:
        fig.add_trace(go.Scattergl(x=plot_dt(top_window), y=top_window[column], mode="lines", name=column), row=2, col=1)

    fig.update_xaxes(title_text="time", row=2, col=1)
    fig.update_yaxes(title_text="events / second", row=1, col=1)
    fig.update_yaxes(title_text="levels", row=2, col=1)
    return figure_layout(fig, 650)


def render_depth_figure(catalog: str, instrument: str, time_range: list[int], levels: int, height: int = 620) -> go.Figure:
    depth = reconstruct_depth_at(catalog, instrument, time_range[1])
    if depth.empty:
        raise ValueError(f"No depth data for {instrument}")

    bids = depth[depth["side"] == "bid"].sort_values("price", ascending=False).head(levels)
    asks = depth[depth["side"] == "ask"].sort_values("price", ascending=True).head(levels)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=-bids["size"], y=bids["price"], orientation="h", name="bid", marker_color="#b33b3b"))
    fig.add_trace(go.Bar(x=asks["size"], y=asks["price"], orientation="h", name="ask", marker_color="#1f7a4d"))
    fig.update_xaxes(title_text="signed size")
    fig.update_yaxes(title_text="price")
    fig.update_layout(title=f"{instrument} reconstructed depth at {from_ms(time_range[1])}", barmode="overlay")
    return figure_layout(fig, height)


def table_data(frame: pd.DataFrame, columns: list[str]) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    table = frame[columns].tail(MAX_TABLE_ROWS).copy()
    if "dt" in table.columns:
        table["dt"] = table["dt"].astype(str)
    return table.to_dict("records"), [{"name": column, "id": column} for column in table.columns] # type: ignore


def data_table(frame: pd.DataFrame, columns: list[str]) -> dash_table.DataTable:
    data, table_columns = table_data(frame, columns)
    return dash_table.DataTable(
        data=data, # type: ignore
        columns=table_columns, # type: ignore
        page_size=25,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto", "height": "520px", "overflowY": "auto"},
        style_cell={
            "fontFamily": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
            "fontSize": "12px",
            "padding": "6px",
            "textAlign": "right",
            "whiteSpace": "nowrap",
        },
        style_header={"fontWeight": "600", "backgroundColor": "#f4f6f8"},
    )


def render_tables(catalog: str, instrument: str, time_range: list[int]) -> dcc.Tabs:
    quotes, trades, top, _ = load_instrument_data(catalog, instrument)
    quotes_window = filter_time(quotes, time_range)
    trades_window = filter_time(trades, time_range)
    top_window = filter_time(top, time_range)
    depth = reconstruct_depth_at(catalog, instrument, time_range[1])

    return dcc.Tabs(
        [
            dcc.Tab(
                label="Top",
                children=data_table(
                    top_window,
                    [
                        "dt",
                        "best_bid",
                        "best_ask",
                        "bid_size",
                        "ask_size",
                        "mid",
                        "spread_bps",
                        "top_imbalance",
                        "bid_levels",
                        "ask_levels",
                    ],
                ),
            ),
            dcc.Tab(
                label="Quotes",
                children=data_table(
                    quotes_window,
                    ["dt", "bid_price_f", "ask_price_f", "bid_size_f", "ask_size_f", "mid_f", "spread_bps"],
                ),
            ),
            dcc.Tab(
                label="Trades",
                children=data_table(trades_window, ["dt", "price_f", "size_f", "aggressor_side", "trade_id"]),
            ),
            dcc.Tab(label="Depth", children=data_table(depth, ["side", "price", "size"])),
        ]
    )


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return figure_layout(fig, 420)


def app_index_string() -> str:
    return """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
        body { margin: 0; background: #f7f8fa; color: #1f2933; font-family: Inter, Arial, sans-serif; }
        .app-shell { display: grid; grid-template-columns: 320px minmax(0, 1fr); min-height: 100vh; }
        .sidebar { background: #ffffff; border-right: 1px solid #d9dee5; padding: 22px 18px; }
        .main { padding: 22px 26px 32px; min-width: 0; }
        .title { font-size: 22px; font-weight: 700; margin: 0 0 6px; }
        .subtitle { font-size: 12px; color: #64748b; margin-bottom: 22px; overflow-wrap: anywhere; }
        .control-label { font-size: 12px; font-weight: 700; color: #475569; margin: 18px 0 8px; text-transform: uppercase; }
        .vertical-controls { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
        .vertical-control { background: #f8fafc; border: 1px solid #d9dee5; border-radius: 6px; padding: 10px 8px 16px; }
        .vertical-title { font-size: 11px; color: #64748b; font-weight: 700; text-align: center; margin-bottom: 10px; }
        .input-grid { display: grid; grid-template-columns: 1fr; gap: 8px; }
        .input-grid input { width: 100%; box-sizing: border-box; padding: 7px 8px; border: 1px solid #cbd5e1; border-radius: 4px; font-size: 12px; }
        .action-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
        .action-row button, .wide-button { border: 1px solid #245c7a; background: #245c7a; color: #fff; border-radius: 4px; padding: 7px 8px; cursor: pointer; }
        .wide-button { width: 100%; margin-top: 8px; }
        .status-text { font-size: 12px; color: #64748b; margin-top: 8px; line-height: 1.35; overflow-wrap: anywhere; }
        .summary-grid { display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap: 10px; margin-bottom: 16px; }
        .summary-grid > div { background: #fff; border: 1px solid #d9dee5; border-radius: 6px; padding: 10px 12px; min-width: 0; }
        .summary-key { font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 700; }
        .summary-value { font-size: 14px; color: #111827; margin-top: 4px; overflow-wrap: anywhere; }
        .chart-card { background: #fff; border: 1px solid #d9dee5; border-radius: 6px; padding: 10px; }
        .price-depth-grid { display: grid; grid-template-columns: 3fr 1fr; gap: 12px; align-items: stretch; }
        .price-depth-grid > div { min-width: 0; }
        .instrument-stack { display: grid; grid-template-columns: 1fr; gap: 12px; }
        .dash-dropdown, .rc-slider { margin-bottom: 8px; }
        @media (max-width: 900px) {
            .app-shell { grid-template-columns: 1fr; }
            .sidebar { border-right: 0; border-bottom: 1px solid #d9dee5; }
            .summary-grid { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
            .price-depth-grid { grid-template-columns: 1fr; }
        }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """


def build_app(catalog: Path) -> Dash:
    catalog_path = catalog.expanduser().resolve()
    catalog_key = str(catalog_path)
    instruments = discover_instruments(catalog_path)
    initial_instrument = instruments[0]
    start_ms, end_ms = time_bounds(catalog_key, initial_instrument)
    animation_start_ms, animation_end_ms = valid_window_bounds(catalog_key, initial_instrument)

    app = Dash(__name__, title="Nautilus Catalog Viewer")
    app.index_string = app_index_string()
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Aside(
                        [
                            html.H1("Nautilus Catalog Viewer", className="title"),
                            html.Div(f"Catalog: {catalog_path}", className="subtitle"),
                            html.Div("Instrument", className="control-label"),
                            dcc.Dropdown(
                                id="instrument",
                                options=[{"label": value, "value": value} for value in instruments],
                                value=[initial_instrument],
                                clearable=False,
                                multi=True,
                            ),
                            html.Div("Time Window", className="control-label"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div("time", className="vertical-title"),
                                            dcc.RangeSlider(
                                                id="time-range",
                                                min=start_ms,
                                                max=end_ms,
                                                value=[start_ms, end_ms],
                                                step=1_000,
                                                marks=slider_marks(start_ms, end_ms),
                                                tooltip={"placement": "right", "always_visible": False},
                                                allowCross=False,
                                                vertical=True,
                                                verticalHeight=360,
                                            ),
                                        ],
                                        className="vertical-control",
                                    ),
                                    html.Div(
                                        [
                                            html.Div("depth", className="vertical-title"),
                                            dcc.Slider(
                                                id="depth-levels",
                                                min=1,
                                                max=100,
                                                step=1,
                                                value=20,
                                                marks={1: "1", 20: "20", 50: "50", 100: "100"},
                                                tooltip={"placement": "right", "always_visible": False},
                                                vertical=True,
                                                verticalHeight=260,
                                            ),
                                        ],
                                        className="vertical-control",
                                    ),
                                ],
                                className="vertical-controls",
                            ),
                            html.Div("Exact Window UTC", className="control-label"),
                            html.Div(
                                [
                                    dcc.Input(id="time-start-input", type="text", value=format_utc_ms(start_ms), debounce=True),
                                    dcc.Input(id="time-end-input", type="text", value=format_utc_ms(end_ms), debounce=True),
                                    html.Button("Apply Window", id="apply-window", n_clicks=0, className="wide-button"),
                                ],
                                className="input-grid",
                            ),
                            html.Div("Animation UTC", className="control-label"),
                            html.Div(
                                [
                                    dcc.Input(
                                        id="animation-start-input",
                                        type="text",
                                        value=format_utc_ms(animation_start_ms),
                                        placeholder="start UTC",
                                        debounce=True,
                                    ),
                                    dcc.Input(
                                        id="animation-end-input",
                                        type="text",
                                        value=format_utc_ms(animation_end_ms),
                                        placeholder="end UTC",
                                        debounce=True,
                                    ),
                                    dcc.Input(id="animation-increment-sec", type="text", value="10", placeholder="increment sec"),
                                    dcc.Input(
                                        id="animation-window-length-sec",
                                        type="text",
                                        value="10",
                                        placeholder="window length sec",
                                    ),
                                    dcc.Input(
                                        id="animation-interval-ms",
                                        type="text",
                                        value=str(DEFAULT_ANIMATION_INTERVAL_MS),
                                        placeholder="render interval ms",
                                    ),
                                    html.Div(
                                        [
                                            html.Button("Start", id="animation-start", n_clicks=0),
                                            html.Button("Stop", id="animation-stop", n_clicks=0),
                                        ],
                                        className="action-row",
                                    ),
                                ],
                                className="input-grid",
                            ),
                            dcc.Interval(id="animation-tick", interval=DEFAULT_ANIMATION_INTERVAL_MS, disabled=True),
                            dcc.Store(
                                id="animation-state",
                                data={
                                    "cursor_ms": animation_start_ms,
                                    "end_ms": animation_end_ms,
                                    "increment_ms": 10_000,
                                    "window_length_ms": 10_000,
                                },
                            ),
                            html.Div(id="animation-status", className="status-text"),
                        ],
                        className="sidebar",
                    ),
                    html.Main(
                        [
                            html.Div(id="summary"),
                            dcc.Tabs(
                                id="view",
                                value="price",
                                children=[
                                    dcc.Tab(label="Price + Depth", value="price"),
                                    dcc.Tab(label="Microstructure", value="microstructure"),
                                    dcc.Tab(label="Activity", value="activity"),
                                    dcc.Tab(label="Depth", value="depth"),
                                    dcc.Tab(label="Tables", value="tables"),
                                ],
                            ),
                            html.Div(id="view-content", className="chart-card"),
                        ],
                        className="main",
                    ),
                ],
                className="app-shell",
            ),
        ]
    )

    @app.callback(
        Output("time-range", "min"),
        Output("time-range", "max"),
        Output("time-range", "value"),
        Output("time-range", "marks"),
        Output("time-start-input", "value"),
        Output("time-end-input", "value"),
        Output("animation-start-input", "value"),
        Output("animation-end-input", "value"),
        Input("instrument", "value"),
    )
    def update_time_range(instrument: list[str] | str):
        instruments = selected_instruments(instrument)
        next_start_ms, next_end_ms = instruments_time_bounds(catalog_key, instruments)
        animation_start_ms, animation_end_ms = instruments_valid_window_bounds(catalog_key, instruments)
        return (
            next_start_ms,
            next_end_ms,
            [next_start_ms, next_end_ms],
            slider_marks(next_start_ms, next_end_ms),
            format_utc_ms(next_start_ms),
            format_utc_ms(next_end_ms),
            format_utc_ms(animation_start_ms),
            format_utc_ms(animation_end_ms),
        )

    @app.callback(
        Output("time-start-input", "value", allow_duplicate=True),
        Output("time-end-input", "value", allow_duplicate=True),
        Input("time-range", "value"),
        prevent_initial_call=True,
    )
    def update_time_inputs(time_range: list[int]):
        return format_utc_ms(time_range[0]), format_utc_ms(time_range[1])

    @app.callback(
        Output("time-range", "value", allow_duplicate=True),
        Input("apply-window", "n_clicks"),
        State("time-start-input", "value"),
        State("time-end-input", "value"),
        State("time-range", "min"),
        State("time-range", "max"),
        prevent_initial_call=True,
    )
    def apply_window(_: int, start_value: str, end_value: str, lower: int, upper: int):
        return normalized_range(parse_utc_ms(start_value), parse_utc_ms(end_value), lower, upper)

    @app.callback(
        Output("animation-tick", "disabled"),
        Output("animation-tick", "interval"),
        Output("animation-state", "data"),
        Output("time-range", "value", allow_duplicate=True),
        Output("animation-status", "children"),
        Input("animation-start", "n_clicks"),
        Input("animation-stop", "n_clicks"),
        State("animation-start-input", "value"),
        State("animation-end-input", "value"),
        State("animation-increment-sec", "value"),
        State("animation-window-length-sec", "value"),
        State("animation-interval-ms", "value"),
        State("time-range", "min"),
        State("time-range", "max"),
        prevent_initial_call=True,
    )
    def configure_animation(
        _: int,
        __: int,
        start_value: str,
        end_value: str,
        increment_sec: str,
        window_length_sec: str,
        interval_ms: str,
        lower: int,
        upper: int,
    ):
        if ctx.triggered_id == "animation-stop":
            return True, no_update, no_update, no_update, "stopped"

        try:
            animation_range = normalized_range(parse_utc_ms(start_value), parse_utc_ms(end_value), lower, upper)
            increment_ms = int(parse_positive_float(increment_sec, "increment_sec") * 1_000)
            window_length_ms = int(parse_positive_float(window_length_sec, "window_length_sec") * 1_000)
            interval = int(parse_positive_float(interval_ms, "interval_ms"))
        except ValueError as error:
            return True, no_update, no_update, no_update, f"error: {error}"

        frame_start_ms = animation_range[0]
        frame_end_ms = min(frame_start_ms + window_length_ms, animation_range[1])
        if frame_start_ms >= frame_end_ms:
            return True, no_update, no_update, no_update, "error: animation window is empty"
        return (
            frame_end_ms >= animation_range[1],
            interval,
            {
                "cursor_ms": frame_start_ms + increment_ms,
                "end_ms": animation_range[1],
                "increment_ms": increment_ms,
                "window_length_ms": window_length_ms,
            },
            [frame_start_ms, frame_end_ms],
            f"running: {format_utc_ms(frame_start_ms)} -> {format_utc_ms(frame_end_ms)}",
        )

    @app.callback(
        Output("time-range", "value", allow_duplicate=True),
        Output("animation-state", "data", allow_duplicate=True),
        Output("animation-tick", "disabled", allow_duplicate=True),
        Output("animation-status", "children", allow_duplicate=True),
        Input("animation-tick", "n_intervals"),
        State("animation-state", "data"),
        State("time-range", "min"),
        State("time-range", "max"),
        prevent_initial_call=True,
    )
    def advance_animation(_: int, state: dict[str, int], lower: int, upper: int):
        cursor_ms = clamp_ms(int(state["cursor_ms"]), lower, upper)
        end_ms = clamp_ms(int(state["end_ms"]), lower, upper)
        increment_ms = int(state["increment_ms"])
        window_length_ms = int(state["window_length_ms"])
        if cursor_ms >= end_ms:
            frame_start_ms = max(lower, end_ms - window_length_ms)
            return [frame_start_ms, end_ms], state, True, "finished"

        frame_end_ms = min(cursor_ms + window_length_ms, end_ms)
        next_cursor_ms = cursor_ms + increment_ms
        next_state = {
            "cursor_ms": next_cursor_ms,
            "end_ms": end_ms,
            "increment_ms": increment_ms,
            "window_length_ms": window_length_ms,
        }
        disabled = frame_end_ms >= end_ms
        status = "finished" if disabled else f"running: {format_utc_ms(cursor_ms)} -> {format_utc_ms(frame_end_ms)}"
        return [cursor_ms, frame_end_ms], next_state, disabled, status

    @app.callback(
        Output("summary", "children"),
        Output("view-content", "children"),
        Input("instrument", "value"),
        Input("time-range", "value"),
        Input("depth-levels", "value"),
        Input("view", "value"),
    )
    def update_view(instrument: list[str] | str, time_range: list[int], depth_levels: int, view: str):
        try:
            instruments = selected_instruments(instrument)
            summary = (
                render_summary(catalog_key, instruments[0], time_range)
                if len(instruments) == 1
                else html.Div(
                    [render_summary(catalog_key, value, time_range) for value in instruments],
                    className="instrument-stack",
                )
            )

            if view == "price":
                if len(instruments) == 1:
                    content = html.Div(
                        [
                            dcc.Graph(
                                figure=render_price_figure(catalog_key, instruments[0], time_range),
                                config={"displaylogo": False},
                            ),
                            dcc.Graph(
                                figure=render_depth_figure(catalog_key, instruments[0], time_range, depth_levels, height=520),
                                config={"displaylogo": False},
                            ),
                        ],
                        className="price-depth-grid",
                    )
                else:
                    content = html.Div(
                        [
                            dcc.Graph(
                                figure=render_price_stack_figure(catalog_key, instruments, time_range),
                                config={"displaylogo": False},
                            ),
                            html.Div(
                                [
                                    dcc.Graph(
                                        figure=render_depth_figure(catalog_key, value, time_range, depth_levels, height=420),
                                        config={"displaylogo": False},
                                    )
                                    for value in instruments
                                ],
                                className="instrument-stack",
                            ),
                        ],
                        className="instrument-stack",
                    )
            elif view == "microstructure":
                content = html.Div(
                    [
                        dcc.Graph(figure=render_microstructure_figure(catalog_key, value, time_range), config={"displaylogo": False})
                        for value in instruments
                    ],
                    className="instrument-stack",
                )
            elif view == "activity":
                content = html.Div(
                    [
                        dcc.Graph(figure=render_activity_figure(catalog_key, value, time_range), config={"displaylogo": False})
                        for value in instruments
                    ],
                    className="instrument-stack",
                )
            elif view == "depth":
                content = html.Div(
                    [
                        dcc.Graph(figure=render_depth_figure(catalog_key, value, time_range, depth_levels), config={"displaylogo": False})
                        for value in instruments
                    ],
                    className="instrument-stack",
                )
            elif view == "tables":
                content = (
                    render_tables(catalog_key, instruments[0], time_range)
                    if len(instruments) == 1
                    else dcc.Tabs(
                        [
                            dcc.Tab(label=value, children=render_tables(catalog_key, value, time_range))
                            for value in instruments
                        ]
                    )
                )
            else:
                raise ValueError(f"Unexpected view: {view}")
        except ValueError as error:
            summary = html.Div()
            content = dcc.Graph(figure=empty_figure(str(error)), config={"displaylogo": False})
        return summary, content

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_app(args.catalog).run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
