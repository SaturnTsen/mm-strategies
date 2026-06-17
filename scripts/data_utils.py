import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from nautilus_trader.model.data import BookOrder
from nautilus_trader.model.data import OrderBookDepth10
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog


FIXED_SCALAR = 10**16
DEPTH_LEVELS = 10
DATA_TYPES = ("order_book_deltas", "quote_tick", "trade_tick")
RECONSTRUCT_DATA_TYPE = "reconstruct"
RECONSTRUCT_MATCH_MIN_RATIO = 0.90
ACTION_ADD = 1
ACTION_UPDATE = 2
ACTION_DELETE = 3
ACTION_CLEAR = 4
SIDE_BUY = 1
SIDE_SELL = 2
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogSegment:
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def key(self) -> str:
        return f"{timestamp_key(self.start)}_{timestamp_key(self.end)}"

    @property
    def label(self) -> str:
        return f"{self.start.strftime('%Y-%m-%d %H:%M:%S')} -> {self.end.strftime('%H:%M:%S')} UTC"


def timestamp_key(value: pd.Timestamp) -> str:
    timestamp = value.tz_convert("UTC") if value.tzinfo is not None else value.tz_localize("UTC")
    fraction = f"{timestamp.microsecond:06d}{timestamp.nanosecond:03d}"
    return f"{timestamp.strftime('%Y-%m-%dT%H-%M-%S')}-{fraction}Z"


def parse_catalog_timestamp(value: str) -> pd.Timestamp:
    date, time_value = value.split("T", maxsplit=1)
    hour, minute, second, fraction = time_value.rstrip("Z").split("-", maxsplit=3)
    padded_fraction = (fraction + "0" * 9)[:9]
    return pd.Timestamp(f"{date}T{hour}:{minute}:{second}.{padded_fraction}Z")


def parse_catalog_file_segment(path: Path) -> CatalogSegment:
    start_value, end_value = path.stem.split("_", maxsplit=1)
    return CatalogSegment(parse_catalog_timestamp(start_value), parse_catalog_timestamp(end_value))


def segment_from_key(segment_key: str) -> CatalogSegment:
    start_value, end_value = segment_key.split("_", maxsplit=1)
    return CatalogSegment(parse_catalog_timestamp(start_value), parse_catalog_timestamp(end_value))


def resolve_catalog_segment(catalog: Path, instrument: str, requested: str) -> str | None:
    if requested == "all":
        return None
    if requested != "latest":
        return requested

    directory = catalog / "data" / "order_book_deltas" / instrument
    if not directory.exists():
        raise FileNotFoundError(f"No order_book_deltas directory: {directory}")
    segments = [parse_catalog_file_segment(path) for path in directory.glob("*.parquet")]
    if not segments:
        raise FileNotFoundError(f"No order_book_deltas parquet files under {directory}")
    return max(segments, key=lambda segment: segment.end).key


def load_nautilus_backtest_data(
    catalog_path: Path,
    instrument: str,
    segment_key: str | None,
    max_rows: int | None,
    max_minutes: float | None,
    book_data_type: str,
) -> tuple[list[object], list[object], pd.Timestamp, pd.Timestamp]:
    catalog = ParquetDataCatalog(str(catalog_path))
    if book_data_type == "depth10":
        book_data = load_depth10_data(catalog_path, instrument, segment_key)
    elif book_data_type == "deltas":
        query = {}
        if segment_key is not None:
            segment = segment_from_key(segment_key)
            query["start"] = segment.start
            query["end"] = segment.end
        book_data = catalog.order_book_deltas([instrument], batched=True, **query)
    else:
        raise ValueError("book_data_type must be depth10 or deltas")

    if not book_data:
        raise ValueError(f"No {book_data_type} book data loaded for {instrument} segment={segment_key or 'all'}")
    if max_minutes is not None:
        cutoff_ns = book_data[0].ts_event + int(max_minutes * 60.0 * 1_000_000_000)
        book_data = [item for item in book_data if item.ts_event <= cutoff_ns]
        if not book_data:
            raise ValueError(f"No book data loaded within --max-minutes={max_minutes}")
    if max_rows is not None:
        book_data = book_data[:max_rows]

    start = pd.Timestamp(book_data[0].ts_event, unit="ns", tz="UTC")
    end = pd.Timestamp(book_data[-1].ts_event, unit="ns", tz="UTC")
    trades = catalog.trade_ticks([instrument], start=start, end=end)
    return book_data, trades, start, end


def load_depth10_data(catalog_path: Path, instrument: str, segment_key: str | None) -> list[OrderBookDepth10]:
    instrument_id = InstrumentId.from_str(instrument)
    depths = load_depths(catalog_path, instrument, segment_key)
    depths = depths.sort_values(["ts_init", "_cache_file_index", "_file_index", "_row"], kind="stable")
    depths = depths.groupby("ts_event", sort=False).tail(1).reset_index(drop=True)
    depths = depths[
        (depths["bid_price_0_f"] > 0.0)
        & (depths["ask_price_0_f"] > 0.0)
        & (depths["bid_price_0_f"] < depths["ask_price_0_f"])
    ].reset_index(drop=True)
    if depths.empty:
        raise ValueError(f"No valid depth10 top-of-book rows for {instrument} segment={segment_key or 'all'}")

    empty_bid = BookOrder(OrderSide.NO_ORDER_SIDE, Price(0, precision=8), Quantity(0, precision=8), 0)
    empty_ask = BookOrder(OrderSide.NO_ORDER_SIDE, Price(0, precision=8), Quantity(0, precision=8), 0)
    data: list[OrderBookDepth10] = []
    for sequence, row in enumerate(depths.itertuples(index=False)):
        bids = []
        asks = []
        bid_counts = []
        ask_counts = []
        for level in range(DEPTH_LEVELS):
            bid_price = getattr(row, f"bid_price_{level}_f")
            bid_size = getattr(row, f"bid_size_{level}_f")
            bid_count = int(getattr(row, f"bid_count_{level}"))
            if bid_count > 0 and pd.notna(bid_price) and pd.notna(bid_size) and bid_price > 0.0 and bid_size > 0.0:
                bids.append(BookOrder(OrderSide.BUY, Price(float(bid_price), precision=8), Quantity(float(bid_size), precision=8), 0))
                bid_counts.append(bid_count)
            else:
                bids.append(empty_bid)
                bid_counts.append(0)

            ask_price = getattr(row, f"ask_price_{level}_f")
            ask_size = getattr(row, f"ask_size_{level}_f")
            ask_count = int(getattr(row, f"ask_count_{level}"))
            if ask_count > 0 and pd.notna(ask_price) and pd.notna(ask_size) and ask_price > 0.0 and ask_size > 0.0:
                asks.append(BookOrder(OrderSide.SELL, Price(float(ask_price), precision=8), Quantity(float(ask_size), precision=8), 0))
                ask_counts.append(ask_count)
            else:
                asks.append(empty_ask)
                ask_counts.append(0)

        data.append(
            OrderBookDepth10(
                instrument_id=instrument_id,
                bids=bids,
                asks=asks,
                bid_counts=bid_counts,
                ask_counts=ask_counts,
                flags=0,
                sequence=sequence,
                ts_event=int(row.ts_event),
                ts_init=int(row.ts_init),
            ),
        )
    return data


def segments_overlap(left: CatalogSegment, right: CatalogSegment) -> bool:
    return left.start <= right.end and right.start <= left.end


def segment_overlap_ratio(segment: CatalogSegment, target: CatalogSegment) -> float:
    target_duration = target.end - target.start
    if target_duration <= pd.Timedelta(0):
        raise ValueError(f"Invalid segment duration: {target}")
    overlap_start = max(segment.start, target.start)
    overlap_end = min(segment.end, target.end)
    if overlap_end <= overlap_start:
        return 0.0
    return float((overlap_end - overlap_start) / target_duration)


def discover_segments(catalog: Path) -> list[CatalogSegment]:
    started = time.perf_counter()
    intervals: list[CatalogSegment] = []
    for instrument in discover_instruments(catalog):
        directory = catalog / "data" / "order_book_deltas" / instrument
        for path in sorted(directory.glob("*.parquet")):
            intervals.append(parse_catalog_file_segment(path))
    if not intervals:
        raise FileNotFoundError(f"No parquet segments found under {catalog / 'data' / 'order_book_deltas'}")

    merged: list[CatalogSegment] = []
    for interval in sorted(intervals, key=lambda value: value.start):
        if not merged or interval.start > merged[-1].end:
            merged.append(interval)
        else:
            merged[-1] = CatalogSegment(merged[-1].start, max(merged[-1].end, interval.end))

    LOGGER.info("catalog segments discovered count=%d elapsed=%.3fs", len(merged), time.perf_counter() - started)
    return merged


def discover_instruments(catalog: Path) -> list[str]:
    started = time.perf_counter()
    data_root = catalog / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"Catalog data directory does not exist: {data_root}")

    instrument_sets = []
    for data_type in DATA_TYPES:
        type_started = time.perf_counter()
        directory = data_root / data_type
        if not directory.exists():
            raise FileNotFoundError(f"Catalog data type directory does not exist: {directory}")
        instruments = {path.name for path in directory.iterdir() if path.is_dir()}
        LOGGER.info(
            "catalog discovery data_type=%s instruments=%d elapsed=%.3fs",
            data_type,
            len(instruments),
            time.perf_counter() - type_started,
        )
        instrument_sets.append(instruments)

    instruments = sorted(set.intersection(*instrument_sets))
    if not instruments:
        raise FileNotFoundError(f"No instruments with all required data types: {', '.join(DATA_TYPES)}")
    LOGGER.info("catalog discovery complete instruments=%d elapsed=%.3fs", len(instruments), time.perf_counter() - started)
    return instruments


def _read_catalog_type(catalog: Path, data_type: str, instrument: str, segment_key: str | None = None) -> pd.DataFrame:
    started = time.perf_counter()
    directory = catalog / "data" / data_type / instrument
    files = sorted(directory.glob("*.parquet"))
    if segment_key is not None:
        segment = segment_from_key(segment_key)
        files = [path for path in files if segments_overlap(parse_catalog_file_segment(path), segment)]
    if not files:
        segment_detail = f" for segment {segment_key}" if segment_key is not None else ""
        raise FileNotFoundError(f"No parquet files found under {directory}{segment_detail}")

    LOGGER.info("read catalog type start data_type=%s instrument=%s segment=%s files=%d", data_type, instrument, segment_key, len(files))
    frames = []
    for file_index, path in enumerate(files):
        file_started = time.perf_counter()
        table = pq.read_table(path)
        frame = table.to_pandas()
        frame["_file_index"] = file_index
        frame["_row"] = range(len(frame))
        frames.append(frame)
        LOGGER.info(
            "read parquet data_type=%s instrument=%s segment=%s file=%d/%d rows=%d path=%s elapsed=%.3fs",
            data_type,
            instrument,
            segment_key,
            file_index + 1,
            len(files),
            len(frame),
            path,
            time.perf_counter() - file_started,
        )

    result = pd.concat(frames, ignore_index=True)
    LOGGER.info(
        "read catalog type done data_type=%s instrument=%s segment=%s rows=%d elapsed=%.3fs",
        data_type,
        instrument,
        segment_key,
        len(result),
        time.perf_counter() - started,
    )
    return result


def fixed_to_float(value: bytes, signed: bool) -> float:
    return int.from_bytes(value, "little", signed=signed) / FIXED_SCALAR


def decode_fixed_column(series: pd.Series, signed: bool) -> pd.Series:
    return series.map(lambda value: fixed_to_float(value, signed=signed)) # type: ignore


def add_datetime(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["dt"] = pd.to_datetime(frame["ts_event"], unit="ns", utc=True)
    return frame


def load_deltas(catalog: Path, instrument: str, segment_key: str | None = None) -> pd.DataFrame:
    started = time.perf_counter()
    deltas = _read_catalog_type(catalog, "order_book_deltas", instrument, segment_key)
    deltas["price_f"] = decode_fixed_column(deltas["price"], signed=True)
    deltas["size_f"] = decode_fixed_column(deltas["size"], signed=False)
    deltas = add_datetime(deltas)
    result = deltas.sort_values(["ts_init", "_file_index", "_row"], kind="stable").reset_index(drop=True)
    LOGGER.info("load deltas done instrument=%s segment=%s rows=%d elapsed=%.3fs", instrument, segment_key, len(result), time.perf_counter() - started)
    return result


def load_quotes(catalog: Path, instrument: str, segment_key: str | None = None) -> pd.DataFrame:
    started = time.perf_counter()
    quotes = _read_catalog_type(catalog, "quote_tick", instrument, segment_key)
    quotes["bid_price_f"] = decode_fixed_column(quotes["bid_price"], signed=True)
    quotes["ask_price_f"] = decode_fixed_column(quotes["ask_price"], signed=True)
    quotes["bid_size_f"] = decode_fixed_column(quotes["bid_size"], signed=False)
    quotes["ask_size_f"] = decode_fixed_column(quotes["ask_size"], signed=False)
    quotes["mid_f"] = (quotes["bid_price_f"] + quotes["ask_price_f"]) / 2.0
    quotes["spread_f"] = quotes["ask_price_f"] - quotes["bid_price_f"]
    quotes["spread_bps"] = quotes["spread_f"] / quotes["mid_f"] * 10_000.0
    quotes = add_datetime(quotes)
    result = quotes.sort_values(["ts_init", "_file_index", "_row"], kind="stable").reset_index(drop=True)
    LOGGER.info("load quotes done instrument=%s segment=%s rows=%d elapsed=%.3fs", instrument, segment_key, len(result), time.perf_counter() - started)
    return result


def load_trades(catalog: Path, instrument: str, segment_key: str | None = None) -> pd.DataFrame:
    started = time.perf_counter()
    trades = _read_catalog_type(catalog, "trade_tick", instrument, segment_key)
    trades["price_f"] = decode_fixed_column(trades["price"], signed=True)
    trades["size_f"] = decode_fixed_column(trades["size"], signed=False)
    trades = add_datetime(trades)
    result = trades.sort_values(["ts_init", "_file_index", "_row"], kind="stable").reset_index(drop=True)
    LOGGER.info("load trades done instrument=%s segment=%s rows=%d elapsed=%.3fs", instrument, segment_key, len(result), time.perf_counter() - started)
    return result


def load_depths(catalog: Path, instrument: str, segment_key: str | None = None) -> pd.DataFrame:
    started = time.perf_counter()
    try:
        depths = _read_catalog_type(catalog, "order_book_depths", instrument, segment_key)
    except FileNotFoundError:
        LOGGER.info("order_book_depths missing instrument=%s segment=%s; loading reconstructed cache", instrument, segment_key)
        result = load_reconstructed_depths(catalog, instrument, segment_key)
        LOGGER.info("load reconstructed depths done instrument=%s segment=%s rows=%d elapsed=%.3fs", instrument, segment_key, len(result), time.perf_counter() - started)
        return result

    for side in ("bid", "ask"):
        for level in range(DEPTH_LEVELS):
            depths[f"{side}_price_{level}_f"] = decode_fixed_column(depths[f"{side}_price_{level}"], signed=True)
            depths[f"{side}_size_{level}_f"] = decode_fixed_column(depths[f"{side}_size_{level}"], signed=False)
    depths = add_datetime(depths)
    result = depths.sort_values(["ts_init", "_file_index", "_row"], kind="stable").reset_index(drop=True)
    LOGGER.info("load depths done instrument=%s segment=%s rows=%d elapsed=%.3fs", instrument, segment_key, len(result), time.perf_counter() - started)
    return result


def reconstructed_depths_path(catalog: Path, instrument: str, segment_key: str | None) -> Path:
    file_name = f"{segment_key}.parquet" if segment_key is not None else "depths.parquet"
    return catalog / "data" / RECONSTRUCT_DATA_TYPE / instrument / file_name


def load_reconstructed_depths(catalog: Path, instrument: str, segment_key: str | None = None) -> pd.DataFrame:
    started = time.perf_counter()
    exact_path = reconstructed_depths_path(catalog, instrument, segment_key)
    if exact_path.exists():
        paths = [exact_path]
    else:
        directory = catalog / "data" / RECONSTRUCT_DATA_TYPE / instrument
        if segment_key is None:
            paths = sorted(directory.glob("*.parquet"))
        else:
            segment = segment_from_key(segment_key)
            matches = [
                (segment_overlap_ratio(parse_catalog_file_segment(path), segment), path)
                for path in sorted(directory.glob("*.parquet"))
            ]
            matches = [(ratio, path) for ratio, path in matches if ratio >= RECONSTRUCT_MATCH_MIN_RATIO]
            paths = [max(matches, key=lambda match: (match[0], match[1].name))[1]] if matches else []
        if not paths:
            raise FileNotFoundError(f"No reconstructed depth cache found: {exact_path}. Run scripts/convert.py for the live instance first.")

    LOGGER.info("read reconstructed depths start instrument=%s segment=%s files=%d", instrument, segment_key, len(paths))
    frames = []
    for file_index, path in enumerate(paths):
        file_started = time.perf_counter()
        frame = pd.read_parquet(path)
        frame["_cache_file_index"] = file_index
        frames.append(frame)
        LOGGER.info(
            "read reconstructed depth file instrument=%s segment=%s file=%d/%d rows=%d path=%s elapsed=%.3fs",
            instrument,
            segment_key,
            file_index + 1,
            len(paths),
            len(frame),
            path,
            time.perf_counter() - file_started,
        )

    depths = pd.concat(frames, ignore_index=True)
    depths["dt"] = pd.to_datetime(depths["dt"], utc=True)
    result = depths.sort_values(["ts_init", "_cache_file_index", "_file_index", "_row"], kind="stable").reset_index(drop=True)
    LOGGER.info("read reconstructed depths done instrument=%s segment=%s rows=%d elapsed=%.3fs", instrument, segment_key, len(result), time.perf_counter() - started)
    return result


def depths_to_top(depths: pd.DataFrame) -> pd.DataFrame:
    started = time.perf_counter()
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
    result = top.reset_index(drop=True)
    LOGGER.info("depths to top done depth_rows=%d top_rows=%d elapsed=%.3fs", len(depths), len(result), time.perf_counter() - started)
    return result


def depth_snapshot_to_long(depths: pd.DataFrame, row_index: int = -1) -> pd.DataFrame:
    started = time.perf_counter()
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
    result = pd.DataFrame(rows)
    LOGGER.info("depth snapshot to long done input_rows=%d output_rows=%d elapsed=%.3fs", len(depths), len(result), time.perf_counter() - started)
    return result
