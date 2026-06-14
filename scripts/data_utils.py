from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


FIXED_SCALAR = 10**16
DEPTH_LEVELS = 10
DATA_TYPES = ("order_book_deltas", "quote_tick", "trade_tick")
ACTION_ADD = 1
ACTION_UPDATE = 2
ACTION_DELETE = 3
ACTION_CLEAR = 4
SIDE_BUY = 1
SIDE_SELL = 2


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


def fixed_to_float(value: bytes, signed: bool) -> float:
    return int.from_bytes(value, "little", signed=signed) / FIXED_SCALAR


def decode_fixed_column(series: pd.Series, signed: bool) -> pd.Series:
    return series.map(lambda value: fixed_to_float(value, signed=signed)) # type: ignore


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


def load_depths(catalog: Path, instrument: str) -> pd.DataFrame:
    try:
        depths = read_catalog_type(catalog, "order_book_depths", instrument)
    except FileNotFoundError:
        return reconstruct_depths(load_deltas(catalog, instrument))

    for side in ("bid", "ask"):
        for level in range(DEPTH_LEVELS):
            depths[f"{side}_price_{level}_f"] = decode_fixed_column(depths[f"{side}_price_{level}"], signed=True)
            depths[f"{side}_size_{level}_f"] = decode_fixed_column(depths[f"{side}_size_{level}"], signed=False)
    depths = add_datetime(depths)
    return depths.sort_values(["ts_init", "_file_index", "_row"], kind="stable").reset_index(drop=True)


def reconstruct_depths(deltas: pd.DataFrame) -> pd.DataFrame:
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    rows = []

    for _, row in deltas.iterrows():
        action = int(row["action"])
        side = int(row["side"])
        price = float(row["price_f"])
        size = float(row["size_f"])

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

        if not bids or not asks:
            continue

        bid_levels = sorted(bids.items(), reverse=True)[:DEPTH_LEVELS]
        ask_levels = sorted(asks.items())[:DEPTH_LEVELS]
        depth_row = {
            "dt": row["dt"],
            "ts_event": row["ts_event"],
            "ts_init": row["ts_init"],
            "_file_index": row["_file_index"],
            "_row": row["_row"],
            "source": "reconstructed_from_deltas",
        }
        for level in range(DEPTH_LEVELS):
            bid_price, bid_size = bid_levels[level] if level < len(bid_levels) else (float("nan"), 0.0)
            ask_price, ask_size = ask_levels[level] if level < len(ask_levels) else (float("nan"), 0.0)
            depth_row[f"bid_price_{level}_f"] = bid_price
            depth_row[f"bid_size_{level}_f"] = bid_size
            depth_row[f"ask_price_{level}_f"] = ask_price
            depth_row[f"ask_size_{level}_f"] = ask_size
            depth_row[f"bid_count_{level}"] = 1 if level < len(bid_levels) else 0
            depth_row[f"ask_count_{level}"] = 1 if level < len(ask_levels) else 0
        rows.append(depth_row)

    if not rows:
        raise ValueError("No valid depth snapshots reconstructed from order book deltas")

    return pd.DataFrame(rows).reset_index(drop=True)


def depths_to_top(depths: pd.DataFrame) -> pd.DataFrame:
    top = depths[["dt", "ts_event", "ts_init", "bid_price_0_f", "ask_price_0_f", "bid_size_0_f", "ask_size_0_f"]].copy()
    top = top.rename(
        columns={
            "bid_price_0_f": "best_bid",
            "ask_price_0_f": "best_ask",
            "bid_size_0_f": "bid_size",
            "ask_size_0_f": "ask_size",
        }
    )
    top = top[(top["best_bid"].notna()) & (top["best_ask"].notna()) & (top["best_bid"] > 0.0) & (top["best_ask"] > 0.0)]
    if top.empty:
        raise ValueError("No valid top-of-book snapshots in depth data")

    top["mid"] = (top["best_bid"] + top["best_ask"]) / 2.0
    top["spread"] = top["best_ask"] - top["best_bid"]
    top["spread_bps"] = top["spread"] / top["mid"] * 10_000.0
    top["top_imbalance"] = (top["bid_size"] - top["ask_size"]) / (top["bid_size"] + top["ask_size"])
    top["bid_levels"] = depths.loc[top.index, [f"bid_count_{level}" for level in range(DEPTH_LEVELS)]].gt(0).sum(axis=1)
    top["ask_levels"] = depths.loc[top.index, [f"ask_count_{level}" for level in range(DEPTH_LEVELS)]].gt(0).sum(axis=1)
    return top.reset_index(drop=True)


def depth_snapshot_to_long(depths: pd.DataFrame, row_index: int = -1) -> pd.DataFrame:
    if depths.empty:
        raise ValueError("Cannot convert empty depth data")

    row = depths.iloc[row_index]
    rows = []
    for side in ("bid", "ask"):
        for level in range(DEPTH_LEVELS):
            price = float(row[f"{side}_price_{level}_f"])
            size = float(row[f"{side}_size_{level}_f"])
            count = int(row[f"{side}_count_{level}"])
            if count > 0 and pd.notna(price) and size > 0.0:
                rows.append({"side": side, "price": price, "size": size, "level": level})

    if not rows:
        raise ValueError("No valid levels in depth snapshot")
    return pd.DataFrame(rows)
