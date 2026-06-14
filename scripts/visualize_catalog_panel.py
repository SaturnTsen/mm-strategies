#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from dash import dash_table
from plotly.subplots import make_subplots

from visualize_catalog import (
    discover_instruments,
    load_deltas,
    load_quotes,
    load_trades,
    reconstruct_top_of_book,
)


DEFAULT_CATALOG_PATH = Path("catalog")
DEFAULT_PORT = 5007
DEFAULT_HOST = "127.0.0.1"
MAX_TABLE_ROWS = 200


@lru_cache(maxsize=16)
def load_instrument_data(catalog: str, instrument: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    catalog_path = Path(catalog)
    deltas = load_deltas(catalog_path, instrument)
    quotes = load_quotes(catalog_path, instrument)
    trades = load_trades(catalog_path, instrument)
    top, depth = reconstruct_top_of_book(deltas)
    return quotes, trades, top, depth


def to_ms(value: pd.Timestamp) -> int:
    timestamp = value.tz_convert("UTC") if value.tzinfo is not None else value.tz_localize("UTC")
    return timestamp.value // 1_000_000


def from_ms(value: int | float) -> pd.Timestamp:
    return pd.to_datetime(int(value), unit="ms", utc=True)


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
    quotes, trades, top, depth = load_instrument_data(catalog, instrument)
    quotes_window = filter_time(quotes, time_range)
    trades_window = filter_time(trades, time_range)
    top_window = filter_time(top, time_range)

    if quotes_window.empty or top_window.empty:
        raise ValueError(f"No quote/top-of-book data in selected window for {instrument}")

    values = [
        ("instrument", instrument),
        ("quotes", len(quotes_window)),
        ("trades", len(trades_window)),
        ("top snapshots", len(top_window)),
        ("depth levels", len(depth)),
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
    fig.add_hline(y=0.0, line_width=1, line_color="#555", opacity=0.45, row=2, col=1)
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


def render_depth_figure(catalog: str, instrument: str, levels: int) -> go.Figure:
    _, _, _, depth = load_instrument_data(catalog, instrument)
    if depth.empty:
        raise ValueError(f"No depth data for {instrument}")

    bids = depth[depth["side"] == "bid"].sort_values("price", ascending=False).head(levels)
    asks = depth[depth["side"] == "ask"].sort_values("price", ascending=True).head(levels)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=-bids["size"], y=bids["price"], orientation="h", name="bid", marker_color="#b33b3b"))
    fig.add_trace(go.Bar(x=asks["size"], y=asks["price"], orientation="h", name="ask", marker_color="#1f7a4d"))
    fig.update_xaxes(title_text="signed size")
    fig.update_yaxes(title_text="price")
    fig.update_layout(title=f"{instrument} last reconstructed depth", barmode="overlay")
    return figure_layout(fig, 620)


def table_data(frame: pd.DataFrame, columns: list[str]) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    table = frame[columns].tail(MAX_TABLE_ROWS).copy()
    if "dt" in table.columns:
        table["dt"] = table["dt"].astype(str)
    return table.to_dict("records"), [{"name": column, "id": column} for column in table.columns]


def data_table(frame: pd.DataFrame, columns: list[str]) -> dash_table.DataTable:
    data, table_columns = table_data(frame, columns)
    return dash_table.DataTable(
        data=data,
        columns=table_columns,
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
    quotes, trades, top, depth = load_instrument_data(catalog, instrument)
    quotes_window = filter_time(quotes, time_range)
    trades_window = filter_time(trades, time_range)
    top_window = filter_time(top, time_range)

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
        .summary-grid { display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap: 10px; margin-bottom: 16px; }
        .summary-grid > div { background: #fff; border: 1px solid #d9dee5; border-radius: 6px; padding: 10px 12px; min-width: 0; }
        .summary-key { font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 700; }
        .summary-value { font-size: 14px; color: #111827; margin-top: 4px; overflow-wrap: anywhere; }
        .chart-card { background: #fff; border: 1px solid #d9dee5; border-radius: 6px; padding: 10px; }
        .dash-dropdown, .rc-slider { margin-bottom: 8px; }
        @media (max-width: 900px) {
            .app-shell { grid-template-columns: 1fr; }
            .sidebar { border-right: 0; border-bottom: 1px solid #d9dee5; }
            .summary-grid { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
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
                                value=initial_instrument,
                                clearable=False,
                            ),
                            html.Div("Time Window", className="control-label"),
                            dcc.RangeSlider(
                                id="time-range",
                                min=start_ms,
                                max=end_ms,
                                value=[start_ms, end_ms],
                                step=1_000,
                                marks=slider_marks(start_ms, end_ms),
                                tooltip={"placement": "bottom", "always_visible": False},
                                allowCross=False,
                            ),
                            html.Div("Depth Levels", className="control-label"),
                            dcc.Slider(id="depth-levels", min=1, max=100, step=1, value=20, marks={1: "1", 20: "20", 50: "50", 100: "100"}),
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
                                    dcc.Tab(label="Price", value="price"),
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
        Input("instrument", "value"),
    )
    def update_time_range(instrument: str):
        next_start_ms, next_end_ms = time_bounds(catalog_key, instrument)
        return next_start_ms, next_end_ms, [next_start_ms, next_end_ms], slider_marks(next_start_ms, next_end_ms)

    @app.callback(
        Output("summary", "children"),
        Output("view-content", "children"),
        Input("instrument", "value"),
        Input("time-range", "value"),
        Input("depth-levels", "value"),
        Input("view", "value"),
    )
    def update_view(instrument: str, time_range: list[int], depth_levels: int, view: str):
        summary = render_summary(catalog_key, instrument, time_range)
        if view == "price":
            content = dcc.Graph(figure=render_price_figure(catalog_key, instrument, time_range), config={"displaylogo": False})
        elif view == "microstructure":
            content = dcc.Graph(figure=render_microstructure_figure(catalog_key, instrument, time_range), config={"displaylogo": False})
        elif view == "activity":
            content = dcc.Graph(figure=render_activity_figure(catalog_key, instrument, time_range), config={"displaylogo": False})
        elif view == "depth":
            content = dcc.Graph(figure=render_depth_figure(catalog_key, instrument, depth_levels), config={"displaylogo": False})
        elif view == "tables":
            content = render_tables(catalog_key, instrument, time_range)
        else:
            raise ValueError(f"Unexpected view: {view}")
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
