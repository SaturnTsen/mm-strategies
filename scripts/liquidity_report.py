#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import (  # noqa: E402
    DEPTH_LEVELS,
    CatalogSegment,
    depths_to_top,
    discover_instruments,
    load_depths,
    load_trades,
    parse_catalog_file_segment,
)


DEFAULT_CATALOG_PATH = Path("catalog")
DEFAULT_DEPTH_BPS = (1.0, 5.0, 10.0)
DEFAULT_IMPACT_NOTIONALS = (1_000.0, 10_000.0, 50_000.0, 100_000.0)


def discover_delta_segments(catalog: Path, instruments: list[str]) -> list[CatalogSegment]:
    intervals: list[CatalogSegment] = []
    for instrument in instruments:
        directory = catalog / "data" / "order_book_deltas" / instrument
        for path in sorted(directory.glob("*.parquet")):
            intervals.append(parse_catalog_file_segment(path))
    if not intervals:
        raise FileNotFoundError(f"No order_book_deltas parquet files found under {catalog / 'data' / 'order_book_deltas'}")

    merged: list[CatalogSegment] = []
    for interval in sorted(intervals, key=lambda value: value.start):
        if not merged or interval.start > merged[-1].end:
            merged.append(interval)
        else:
            merged[-1] = CatalogSegment(merged[-1].start, max(merged[-1].end, interval.end))
    return merged


def resolve_instruments(catalog: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    try:
        return discover_instruments(catalog)
    except FileNotFoundError:
        deltas = {path.name for path in (catalog / "data" / "order_book_deltas").iterdir() if path.is_dir()}
        trades = {path.name for path in (catalog / "data" / "trades").iterdir() if path.is_dir()}
        instruments = sorted(deltas & trades)
        if not instruments:
            raise
        return instruments


def valid_depth_rows(depths: pd.DataFrame) -> pd.DataFrame:
    valid = (
        depths["bid_price_0_f"].notna()
        & depths["ask_price_0_f"].notna()
        & (depths["bid_price_0_f"] > 0.0)
        & (depths["ask_price_0_f"] > 0.0)
    )
    return depths.loc[valid].reset_index(drop=True)


def load_market_data(catalog: Path, instruments: list[str], segment_key: str) -> dict[str, dict[str, pd.DataFrame]]:
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for instrument in instruments:
        depths = load_depths(catalog, instrument, segment_key)
        top = depths_to_top(depths)
        rows = valid_depth_rows(depths)
        if len(rows) != len(top):
            raise ValueError(f"Depth/top row mismatch for {instrument}: depths={len(rows)} top={len(top)}")
        trades = load_trades(catalog, instrument, segment_key)
        data[instrument] = {"depths": rows, "top": top, "trades": trades}
    return data


def liquidity_summary(data: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for instrument, frames in data.items():
        top = frames["top"]
        trades = frames["trades"]
        start = min(top["dt"].min(), trades["dt"].min()) if not trades.empty else top["dt"].min()
        end = max(top["dt"].max(), trades["dt"].max()) if not trades.empty else top["dt"].max()
        seconds = max((end - start).total_seconds(), 1e-9)
        bid_notional = top["best_bid"] * top["bid_size"]
        ask_notional = top["best_ask"] * top["ask_size"]
        trade_notional = trades["price_f"] * trades["size_f"] if not trades.empty else pd.Series(dtype=float)
        rows.append(
            {
                "instrument": instrument,
                "seconds": seconds,
                "top_rows": len(top),
                "trades": len(trades),
                "spread_med_bps": top["spread_bps"].median(),
                "spread_p90_bps": top["spread_bps"].quantile(0.90),
                "spread_p99_bps": top["spread_bps"].quantile(0.99),
                "top_bid_notional_med": bid_notional.median(),
                "top_ask_notional_med": ask_notional.median(),
                "top_two_sided_notional_med": (bid_notional + ask_notional).median(),
                "top_update_s": len(top) / seconds,
                "trade_s": len(trades) / seconds,
                "trade_notional_sum": trade_notional.sum(),
                "trade_notional_med": trade_notional.median() if not trade_notional.empty else np.nan,
                "bid_levels_med": top["bid_levels"].median(),
                "ask_levels_med": top["ask_levels"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values(["spread_med_bps", "top_two_sided_notional_med"], ascending=[True, False])


def depth_within_bps(data: dict[str, dict[str, pd.DataFrame]], bands: list[float]) -> pd.DataFrame:
    rows = []
    for instrument, frames in data.items():
        depths = frames["depths"]
        top = frames["top"]
        mid = top["mid"].to_numpy(dtype=float)
        for bps in bands:
            bid_notional = np.zeros(len(depths))
            ask_notional = np.zeros(len(depths))
            bid_lower = mid * (1.0 - bps / 10_000.0)
            ask_upper = mid * (1.0 + bps / 10_000.0)
            for level in range(DEPTH_LEVELS):
                bid_price = depths[f"bid_price_{level}_f"].to_numpy(dtype=float)
                bid_size = depths[f"bid_size_{level}_f"].to_numpy(dtype=float)
                bid_count = depths[f"bid_count_{level}"].to_numpy(dtype=int) > 0
                ask_price = depths[f"ask_price_{level}_f"].to_numpy(dtype=float)
                ask_size = depths[f"ask_size_{level}_f"].to_numpy(dtype=float)
                ask_count = depths[f"ask_count_{level}"].to_numpy(dtype=int) > 0
                bid_mask = bid_count & np.isfinite(bid_price) & (bid_price >= bid_lower) & (bid_size > 0.0)
                ask_mask = ask_count & np.isfinite(ask_price) & (ask_price <= ask_upper) & (ask_size > 0.0)
                bid_notional += np.where(bid_mask, bid_price * bid_size, 0.0)
                ask_notional += np.where(ask_mask, ask_price * ask_size, 0.0)
            rows.append(
                {
                    "instrument": instrument,
                    "bps": bps,
                    "bid_notional_med": float(np.median(bid_notional)),
                    "ask_notional_med": float(np.median(ask_notional)),
                    "two_sided_notional_med": float(np.median(bid_notional + ask_notional)),
                    "bid_notional_p10": float(np.quantile(bid_notional, 0.10)),
                    "ask_notional_p10": float(np.quantile(ask_notional, 0.10)),
                }
            )
    return pd.DataFrame(rows).sort_values(["bps", "two_sided_notional_med"], ascending=[True, False])


def execution_costs(data: dict[str, dict[str, pd.DataFrame]], notionals: list[float], stride: int) -> pd.DataFrame:
    rows = []
    for instrument, frames in data.items():
        depths = frames["depths"].iloc[::stride].reset_index(drop=True)
        top = frames["top"].iloc[::stride].reset_index(drop=True)
        mids = top["mid"].to_numpy(dtype=float)
        level_data = []
        for _, row in depths.iterrows():
            bids = []
            asks = []
            for level in range(DEPTH_LEVELS):
                bid_price = float(row[f"bid_price_{level}_f"])
                bid_size = float(row[f"bid_size_{level}_f"])
                bid_count = int(row[f"bid_count_{level}"])
                ask_price = float(row[f"ask_price_{level}_f"])
                ask_size = float(row[f"ask_size_{level}_f"])
                ask_count = int(row[f"ask_count_{level}"])
                if bid_count > 0 and np.isfinite(bid_price) and bid_size > 0.0:
                    bids.append((bid_price, bid_size))
                if ask_count > 0 and np.isfinite(ask_price) and ask_size > 0.0:
                    asks.append((ask_price, ask_size))
            level_data.append((bids, asks))

        for notional in notionals:
            buy_costs = []
            sell_costs = []
            for mid, (bids, asks) in zip(mids, level_data, strict=True):
                remaining_quote = notional
                bought_base = 0.0
                for price, size in asks:
                    quote_capacity = price * size
                    take_quote = min(remaining_quote, quote_capacity)
                    bought_base += take_quote / price
                    remaining_quote -= take_quote
                    if remaining_quote <= 1e-9:
                        break
                if remaining_quote <= 1e-9 and bought_base > 0.0:
                    buy_avg = notional / bought_base
                    buy_costs.append((buy_avg / mid - 1.0) * 10_000.0)
                else:
                    buy_costs.append(np.nan)

                target_base = notional / mid
                remaining_base = target_base
                proceeds = 0.0
                for price, size in bids:
                    take_base = min(remaining_base, size)
                    proceeds += take_base * price
                    remaining_base -= take_base
                    if remaining_base <= 1e-12:
                        break
                if remaining_base <= 1e-12 and target_base > 0.0:
                    sell_avg = proceeds / target_base
                    sell_costs.append((1.0 - sell_avg / mid) * 10_000.0)
                else:
                    sell_costs.append(np.nan)

            buy = pd.Series(buy_costs, dtype=float)
            sell = pd.Series(sell_costs, dtype=float)
            rows.append(
                {
                    "instrument": instrument,
                    "notional": notional,
                    "snapshots": len(depths),
                    "buy_cost_med_bps": buy.median(), # type: ignore
                    "sell_cost_med_bps": sell.median(), # type: ignore
                    "buy_fill_rate": buy.notna().mean(),
                    "sell_fill_rate": sell.notna().mean(),
                }
            )
    return pd.DataFrame(rows).sort_values(["notional", "buy_cost_med_bps", "sell_cost_med_bps"], ascending=[True, True, True])


def aligned_returns(data: dict[str, dict[str, pd.DataFrame]], freq_ms: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    starts = []
    ends = []
    mids: dict[str, pd.Series] = {}
    for instrument, frames in data.items():
        top = frames["top"][["dt", "mid"]].dropna().sort_values("dt")
        series = top.groupby("dt")["mid"].last()
        starts.append(series.index.min())
        ends.append(series.index.max())
        mids[instrument] = series

    start = max(starts)
    end = min(ends)
    if start >= end:
        raise ValueError("Selected instruments have no common mid-price interval")

    grid = pd.date_range(start=start, end=end, freq=pd.Timedelta(milliseconds=freq_ms))
    aligned = {}
    for instrument, series in mids.items():
        aligned[instrument] = series.reindex(series.index.union(grid)).sort_index().ffill().reindex(grid)
    prices = pd.DataFrame(aligned).dropna()
    returns = np.log(prices).diff().dropna() # type: ignore
    if returns.empty:
        raise ValueError("Aligned mid-price returns are empty")
    return prices, returns


def lead_lag_correlations(returns: pd.DataFrame, freq_ms: int, max_lag_ms: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_steps = max(1, max_lag_ms // freq_ms)
    values = {column: returns[column].to_numpy(dtype=float) for column in returns.columns}
    prefixes = {}
    for column, value in values.items():
        prefixes[column] = {
            "x": np.concatenate([[0.0], np.cumsum(value)]),
            "x2": np.concatenate([[0.0], np.cumsum(value * value)]),
        }
    rows = []
    for source in returns.columns:
        for target in returns.columns:
            if source == target:
                continue
            source_values = values[source]
            target_values = values[target]
            cross = correlate(target_values, source_values, mode="full", method="fft")
            lags = correlation_lags(len(target_values), len(source_values), mode="full")
            lag_mask = (lags >= 1) & (lags <= max_steps)
            lag_values = lags[lag_mask]
            sxy_values = cross[lag_mask]

            n = len(source_values) - lag_values
            source_prefix = prefixes[source]
            target_prefix = prefixes[target]
            sx = source_prefix["x"][len(source_values) - lag_values] - source_prefix["x"][0]
            sx2 = source_prefix["x2"][len(source_values) - lag_values] - source_prefix["x2"][0]
            sy = target_prefix["x"][len(target_values)] - target_prefix["x"][lag_values]
            sy2 = target_prefix["x2"][len(target_values)] - target_prefix["x2"][lag_values]
            numerator = sxy_values - sx * sy / n
            denominator = np.sqrt((sx2 - sx * sx / n) * (sy2 - sy * sy / n))
            corr_values = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float), where=denominator > 0.0)
            finite = np.isfinite(corr_values)
            if finite.any():
                finite_indices = np.flatnonzero(finite)
                local = finite_indices[int(np.argmax(corr_values[finite]))]
                best_corr = float(corr_values[local])
                best_lag = int(lag_values[local] * freq_ms)
            else:
                best_corr = np.nan
                best_lag = 0
            rows.append({"source": source, "target": target, "best_lag_ms": best_lag, "best_corr": best_corr})
    pairwise = pd.DataFrame(rows).sort_values("best_corr", ascending=False)

    score_rows = []
    instruments = list(returns.columns)
    for instrument in instruments:
        out_score = pairwise.loc[pairwise["source"] == instrument, "best_corr"].sum()
        in_score = pairwise.loc[pairwise["target"] == instrument, "best_corr"].sum()
        wins = 0
        for other in instruments:
            if other == instrument:
                continue
            left = pairwise[(pairwise["source"] == instrument) & (pairwise["target"] == other)]["best_corr"].iloc[0]
            right = pairwise[(pairwise["source"] == other) & (pairwise["target"] == instrument)]["best_corr"].iloc[0]
            wins += int(left > right)
        score_rows.append(
            {
                "instrument": instrument,
                "corr_out_sum": out_score,
                "corr_in_sum": in_score,
                "corr_net": out_score - in_score,
                "pair_wins": wins,
            }
        )
    scores = pd.DataFrame(score_rows).sort_values(["pair_wins", "corr_net"], ascending=False)
    return pairwise, scores


def ols_r2(y: pd.Series, x: pd.DataFrame) -> float:
    frame = pd.concat([y.rename("y"), x], axis=1).dropna()
    if len(frame) <= len(x.columns) + 2:
        return np.nan
    y_values = frame["y"].to_numpy(dtype=float)
    x_values = frame.drop(columns=["y"]).to_numpy(dtype=float)
    x_values = np.column_stack([np.ones(len(x_values)), x_values])
    beta, *_ = np.linalg.lstsq(x_values, y_values, rcond=None)
    residuals = y_values - x_values @ beta
    ss_res = float(np.dot(residuals, residuals))
    centered = y_values - y_values.mean()
    ss_tot = float(np.dot(centered, centered))
    if ss_tot <= 0.0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def fit_ols(y_values: np.ndarray, x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    design = np.column_stack([np.ones(len(x_values)), x_values])
    beta, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    fitted = design @ beta
    residuals = y_values - fitted
    ss_res = float(np.dot(residuals, residuals))
    centered = y_values - y_values.mean()
    ss_tot = float(np.dot(centered, centered))
    r2 = np.nan if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    fitted_centered = fitted - fitted.mean()
    denominator = np.sqrt(float(np.dot(centered, centered) * np.dot(fitted_centered, fitted_centered)))
    corr = np.nan if denominator <= 0.0 else float(np.dot(centered, fitted_centered) / denominator)
    return beta, fitted, r2, corr


def focused_lag_ols(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    target: str,
    sources: list[str],
    freq_ms: int,
    max_lag_ms: int,
    horizon_ms: int,
    active_only: bool,
    progress: bool,
) -> pd.DataFrame:
    missing = [instrument for instrument in [target, *sources] if instrument not in returns.columns]
    if missing:
        raise ValueError(f"Focused OLS instruments missing from returns: {', '.join(missing)}")

    max_steps = max(1, max_lag_ms // freq_ms)
    horizon_steps = max(1, horizon_ms // freq_ms)
    target_return = np.log(prices[target].shift(-horizon_steps) / prices[target]).rename("target") # type: ignore
    rows = []
    lag_values = range(1, max_steps + 1)
    iterator = tqdm(
        lag_values,
        desc=f"focused OLS h={horizon_steps * freq_ms}ms",
        unit="lag",
        disable=not progress,
    )
    for lag in iterator:
        frame = pd.DataFrame({"target": target_return})
        for source in sources:
            frame[source] = returns[source].shift(lag)
        frame = frame.dropna()
        if active_only:
            active = frame["target"].ne(0.0)
            for source in sources:
                active = active | frame[source].ne(0.0)
            frame = frame.loc[active]
        if len(frame) <= len(sources) + 2:
            continue

        y_values = frame["target"].to_numpy(dtype=float)
        x_values = frame[sources].to_numpy(dtype=float)
        beta, _, r2, corr = fit_ols(y_values, x_values)
        row = {
            "target": target,
            "sources": ",".join(sources),
            "lag_ms": lag * freq_ms,
            "horizon_ms": horizon_steps * freq_ms,
            "rows": len(frame),
            "r2": r2,
            "corr_fitted_target": corr,
            "intercept": beta[0],
        }
        for index, source in enumerate(sources, start=1):
            row[f"beta_{source}"] = beta[index]
        rows.append(row)

    if not rows:
        raise ValueError("Focused OLS produced no fitted lag rows")
    return pd.DataFrame(rows).sort_values(["r2", "corr_fitted_target"], ascending=False)


def lead_lag_ols(returns: pd.DataFrame, freq_ms: int, max_lag_ms: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_steps = max(1, max_lag_ms // freq_ms)
    lagged = pd.DataFrame(
        {
            f"{instrument}_lag{lag}": returns[instrument].shift(lag)
            for instrument in returns.columns
            for lag in range(1, max_steps + 1)
        }
    )

    rows = []
    for target in returns.columns:
        all_columns = list(lagged.columns)
        full_r2 = ols_r2(returns[target], lagged[all_columns])
        for source in returns.columns:
            if source == target:
                continue
            without_source = [column for column in all_columns if not column.startswith(f"{source}_lag")]
            reduced_r2 = ols_r2(returns[target], lagged[without_source])
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "full_r2": full_r2,
                    "r2_gain": full_r2 - reduced_r2 if pd.notna(full_r2) and pd.notna(reduced_r2) else np.nan,
                }
            )
    pairwise = pd.DataFrame(rows).sort_values("r2_gain", ascending=False)
    scores = (
        pairwise.groupby("source", as_index=False)["r2_gain"]
        .sum()
        .rename(columns={"source": "instrument", "r2_gain": "ols_r2_gain_sum"}) # type: ignore
        .sort_values("ols_r2_gain_sum", ascending=False)
    )
    return pairwise, scores


def write_csvs(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def print_table(title: str, table: pd.DataFrame, max_rows: int | None = None) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    value = table if max_rows is None else table.head(max_rows)
    print(value.to_string(index=False, float_format=lambda x: f"{x:.6g}"))


def parse_float_list(values: list[str] | None, default: tuple[float, ...]) -> list[float]:
    if values is None:
        return list(default)
    parsed = []
    for value in values:
        parsed.extend(float(item) for item in value.split(",") if item)
    if not parsed:
        raise ValueError("At least one numeric value is required")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report liquidity and lead-lag statistics from a Nautilus parquet catalog.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--segment", default=None, help="Catalog segment key. Defaults to latest order_book_deltas segment.")
    parser.add_argument("-i", "--instrument", action="append", default=None)
    parser.add_argument("--freq-ms", type=int, default=1)
    parser.add_argument("--max-lag-ms", type=int, default=250)
    parser.add_argument("--with-ols", action="store_true", help="Also compute OLS R2 gains. This can be slow at 1ms grids.")
    parser.add_argument("--ols-max-lag-ms", type=int, default=None, help="OLS lag window in ms. Defaults to min(max-lag-ms, 10) for speed.")
    parser.add_argument("--focused-target", default=None, help="Target instrument for joint lag OLS.")
    parser.add_argument("--focused-source", action="append", default=None, help="Source instrument for joint lag OLS. Repeat for multiple sources.")
    parser.add_argument("--focused-horizon-ms", action="append", default=None, help="Forward target return horizon in ms. Comma-separated values are accepted.")
    parser.add_argument("--focused-active-only", action=argparse.BooleanOptionalAction, default=True, help="Drop all-zero target/source rows in focused OLS.")
    parser.add_argument("--focused-top", type=int, default=20)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth-bps", action="append", default=None, help="Comma-separated bps bands, default 1,5,10.")
    parser.add_argument("--impact-notional", action="append", default=None, help="Comma-separated quote notionals, default 1000,10000,50000,100000.")
    parser.add_argument("--impact-stride", type=int, default=10, help="Use every Nth depth snapshot for impact costs. Use 1 for exact tick-level scan.")
    parser.add_argument("--csv-dir", type=Path, default=None)
    parser.add_argument("--top-leads", type=int, default=20)
    args = parser.parse_args()
    if args.freq_ms <= 0:
        raise ValueError("--freq-ms must be positive")
    if args.max_lag_ms < args.freq_ms:
        raise ValueError("--max-lag-ms must be greater than or equal to --freq-ms")
    if args.ols_max_lag_ms is not None and args.ols_max_lag_ms < args.freq_ms:
        raise ValueError("--ols-max-lag-ms must be greater than or equal to --freq-ms")
    if args.impact_stride <= 0:
        raise ValueError("--impact-stride must be positive")
    if args.focused_target is not None and not args.focused_source:
        raise ValueError("--focused-target requires at least one --focused-source")
    return args


def main() -> None:
    args = parse_args()
    catalog = args.catalog.expanduser()
    instruments = resolve_instruments(catalog, args.instrument)
    segment_key = args.segment or discover_delta_segments(catalog, instruments)[-1].key
    depth_bps = parse_float_list(args.depth_bps, DEFAULT_DEPTH_BPS)
    impact_notionals = parse_float_list(args.impact_notional, DEFAULT_IMPACT_NOTIONALS)
    focused_horizons = parse_float_list(args.focused_horizon_ms, (args.freq_ms,))
    ols_max_lag_ms = args.ols_max_lag_ms if args.ols_max_lag_ms is not None else min(args.max_lag_ms, 10)

    print(f"catalog={catalog}")
    print(f"segment={segment_key}")
    print(f"instruments={','.join(instruments)}")
    print(f"lead_lag_grid_ms={args.freq_ms} max_lag_ms={args.max_lag_ms} with_ols={args.with_ols} ols_max_lag_ms={ols_max_lag_ms}")
    print(f"impact_stride={args.impact_stride}")

    data = load_market_data(catalog, instruments, segment_key)
    summary = liquidity_summary(data)
    depth = depth_within_bps(data, depth_bps)
    impact = execution_costs(data, impact_notionals, args.impact_stride)
    prices, returns = aligned_returns(data, args.freq_ms)
    corr_pairs, corr_scores = lead_lag_correlations(returns, args.freq_ms, args.max_lag_ms)
    if args.focused_target is not None:
        focused_frames = [
            focused_lag_ols(
                returns=returns,
                prices=prices,
                target=args.focused_target,
                sources=args.focused_source,
                freq_ms=args.freq_ms,
                max_lag_ms=args.max_lag_ms,
                horizon_ms=int(horizon),
                active_only=args.focused_active_only,
                progress=args.progress,
            )
            for horizon in focused_horizons
        ]
        focused = pd.concat(focused_frames, ignore_index=True).sort_values(["r2", "corr_fitted_target"], ascending=False)
    else:
        focused = pd.DataFrame()
    if args.with_ols:
        ols_pairs, ols_scores = lead_lag_ols(returns, args.freq_ms, ols_max_lag_ms)
    else:
        ols_pairs = pd.DataFrame(columns=["source", "target", "full_r2", "r2_gain"])
        ols_scores = pd.DataFrame({"instrument": list(returns.columns), "ols_r2_gain_sum": 0.0})
    leader_scores = corr_scores.merge(ols_scores, on="instrument", how="left").sort_values(
        ["pair_wins", "corr_net", "ols_r2_gain_sum"],
        ascending=False,
    )
    alignment = pd.DataFrame(
        [
            {
                "grid_rows": len(prices),
                "return_rows": len(returns),
                "active_return_rows": int(returns.ne(0.0).any(axis=1).sum()),
                "start": prices.index.min(),
                "end": prices.index.max(),
            }
        ]
    )

    tables = {
        "liquidity_summary": summary,
        "depth_within_bps": depth,
        "execution_costs": impact,
        "lead_lag_corr_pairs": corr_pairs,
        "lead_lag_corr_scores": corr_scores,
        "lead_lag_ols_pairs": ols_pairs,
        "lead_lag_ols_scores": ols_scores,
        "leader_scores": leader_scores,
        "alignment": alignment,
    }
    if args.focused_target is not None:
        tables["focused_lag_ols"] = focused

    print_table("Liquidity Summary", summary)
    print_table("Depth Within Bps", depth)
    print_table("Execution Costs", impact)
    print_table("Lead-Lag Alignment", alignment)
    print_table("Leader Scores", leader_scores)
    print_table("Top Lead-Lag Correlations", corr_pairs, args.top_leads)
    if args.focused_target is not None:
        print_table("Focused Joint Lag OLS", focused, args.focused_top)
    print_table("Top OLS R2 Gains", ols_pairs, args.top_leads)

    if args.csv_dir is not None:
        write_csvs(args.csv_dir, tables)
        print(f"\nwrote_csv_dir={args.csv_dir}")


if __name__ == "__main__":
    main()
