#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import (
    ACTION_ADD,
    ACTION_CLEAR,
    ACTION_DELETE,
    ACTION_UPDATE,
    DEPTH_LEVELS,
    SIDE_BUY,
    SIDE_SELL,
    load_deltas,
    reconstructed_depths_path,
    timestamp_key,
)


DEFAULT_CATALOG_PATH = Path("catalog")
STREAMING_DATA_TYPES = ("order_book_deltas", "quotes", "trades")
DATA_TYPE_ALIASES = {
    "quote_tick": "quotes",
    "trade_tick": "trades",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert live feather data to parquet and reconstruct depth cache from order_book_deltas.",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--instance", default=None, help="Live instance UUID. Use latest to select the newest instance.")
    parser.add_argument("--data-type", action="append", choices=sorted((*STREAMING_DATA_TYPES, *DATA_TYPE_ALIASES)), default=None)
    parser.add_argument("-i", "--instrument", action="append", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--overwrite-reconstruct", action="store_true")
    return parser.parse_args()


def catalog_filename(start: int, end: int) -> str:
    start_key = timestamp_key(pd.Timestamp(start, unit="ns", tz="UTC"))
    end_key = timestamp_key(pd.Timestamp(end, unit="ns", tz="UTC"))
    return f"{start_key}_{end_key}.parquet"


def read_feather_table(path: Path) -> pa.Table:
    with path.open("rb") as file:
        return pa.ipc.open_stream(file).read_all()


def normalize_stream_table(table: pa.Table) -> pa.Table:
    if table.num_rows == 0:
        return table
    if "ts_init" not in table.schema.names:
        raise ValueError("Feather table has no ts_init column")

    ts_init = table.column("ts_init")
    is_sorted = pc.all(pc.greater_equal(ts_init.slice(1), ts_init.slice(0, len(ts_init) - 1))).as_py() # type: ignore
    if not is_sorted:
        table = table.take(pc.sort_indices(table, sort_keys=[("ts_init", "ascending")])) # type: ignore
    return table


def write_stream_parquet(table: pa.Table, target_path: Path) -> None:
    table = normalize_stream_table(table)
    if table.num_rows == 0:
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp.parquet")
    pq.write_table(table, tmp_path, row_group_size=5000)
    tmp_path.replace(target_path)


def feather_ts_init_bounds(path: Path) -> tuple[int, int]:
    minimum: int | None = None
    maximum: int | None = None
    with path.open("rb") as file:
        reader = pa.ipc.open_stream(file)
        ts_init_index = reader.schema.get_field_index("ts_init")
        if ts_init_index < 0:
            raise ValueError(f"Feather file has no ts_init column: {path}")

        for batch in reader:
            column = batch.column(ts_init_index)
            batch_min = pc.min(column).as_py() # type: ignore
            batch_max = pc.max(column).as_py() # type: ignore
            if batch_min is None or batch_max is None:
                continue
            minimum = int(batch_min) if minimum is None else min(minimum, int(batch_min))
            maximum = int(batch_max) if maximum is None else max(maximum, int(batch_max))

    if minimum is None or maximum is None:
        raise ValueError(f"Feather file has no ts_init values: {path}")
    return minimum, maximum


def reconstruct_depths(deltas: pd.DataFrame, label: str) -> pd.DataFrame:
    started = time.perf_counter()
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    rows = []
    total_rows = len(deltas)
    print(f"reconstruct depths start deltas={total_rows}")

    progress = tqdm(deltas.iterrows(), total=total_rows, desc=label, unit="delta", dynamic_ncols=True)
    for processed, (_, row) in enumerate(progress, start=1):
        action = int(row["action"])
        side = int(row["side"])
        price = float(row["price_f"])
        size = float(row["size_f"])

        if action == ACTION_CLEAR:
            bids.clear()
            asks.clear()
        elif action in (ACTION_ADD, ACTION_UPDATE):
            if side == SIDE_BUY:
                book = bids
            elif side == SIDE_SELL:
                book = asks
            else:
                raise ValueError(f"Unexpected side={side} for action={action}")
            if size <= 0.0:
                book.pop(price, None)
            else:
                book[price] = size
        elif action == ACTION_DELETE:
            if side == SIDE_BUY:
                bids.pop(price, None)
            elif side == SIDE_SELL:
                asks.pop(price, None)
            else:
                raise ValueError(f"Unexpected side={side} for delete")
        else:
            raise ValueError(f"Unexpected action={action}")

        if not bids or not asks:
            if processed % 100_000 == 0 or processed == total_rows:
                progress.set_postfix(snapshots=len(rows), bids=len(bids), asks=len(asks))
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
        if "_segment_key" in row.index:
            depth_row["_segment_key"] = row["_segment_key"]
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
        if processed % 100_000 == 0 or processed == total_rows:
            progress.set_postfix(snapshots=len(rows), bids=len(bids), asks=len(asks))

    if not rows:
        raise ValueError("No valid depth snapshots reconstructed from order book deltas")

    result = pd.DataFrame(rows).reset_index(drop=True)
    print(f"reconstruct depths done snapshots={len(result)} elapsed={time.perf_counter() - started:.3f}s")
    return result


def load_instrument_delta_segments(catalog_path: Path, instrument: str, segment_keys: list[str]) -> pd.DataFrame:
    frames = []
    for segment_index, segment_key in enumerate(segment_keys):
        frame = load_deltas(catalog_path, instrument, segment_key)
        frame["_segment_key"] = segment_key
        frame["_segment_index"] = segment_index
        frames.append(frame)
    if not frames:
        raise ValueError(f"No delta segments selected for {instrument}")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["ts_init", "_segment_index", "_file_index", "_row"],
        kind="stable",
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    data_types = None if args.data_type is None else {DATA_TYPE_ALIASES.get(value, value) for value in args.data_type}
    catalog_path = args.catalog.expanduser()
    live_path = catalog_path / "live"
    if not live_path.exists():
        raise FileNotFoundError(f"Live catalog directory does not exist: {live_path}")

    selections = []
    for instance_path in sorted(path for path in live_path.iterdir() if path.is_dir()):
        instance_id = instance_path.name
        for data_type in sorted(STREAMING_DATA_TYPES):
            data_type_path = instance_path / data_type
            if not data_type_path.exists():
                continue
            for instrument_path in sorted(path for path in data_type_path.iterdir() if path.is_dir()):
                feather_paths = sorted(instrument_path.glob("*.feather"))
                if not feather_paths:
                    continue
                selections.append(
                    {
                        "instance_id": instance_id,
                        "data_type": data_type,
                        "instrument": instrument_path.name,
                        "paths": feather_paths,
                        "files": len(feather_paths),
                        "bytes": sum(path.stat().st_size for path in feather_paths),
                        "instance_mtime_ns": instance_path.stat().st_mtime_ns,
                        "feather_mtime_ns": max(path.stat().st_mtime_ns for path in feather_paths),
                    },
                )

    if not selections:
        raise FileNotFoundError(f"No live feather files found under {live_path}")

    if args.instance is None:
        instance_ids = sorted({selection["instance_id"] for selection in selections})
        if len(instance_ids) != 1 and not args.list:
            raise ValueError("Multiple live instances match. Use --instance latest or --instance <uuid>.")
        instance_id = instance_ids[0] if len(instance_ids) == 1 else None
    elif args.instance == "latest":
        instance_id = max((selection["feather_mtime_ns"], selection["instance_id"]) for selection in selections)[1]
    else:
        instance_id = args.instance

    selected = selections
    if instance_id is not None:
        selected = [selection for selection in selected if selection["instance_id"] == instance_id]
    if data_types is not None:
        selected = [selection for selection in selected if selection["data_type"] in data_types]
    if args.instrument is not None:
        selected = [selection for selection in selected if selection["instrument"] in args.instrument]
    selected = sorted(
        selected,
        key=lambda selection: (
            selection["feather_mtime_ns"],
            selection["instance_mtime_ns"],
            selection["instance_id"],
            selection["data_type"],
            selection["instrument"],
        ),
    )

    for selection in selected:
        target_paths = []
        for feather_path in selection["paths"]:
            start, end = feather_ts_init_bounds(feather_path)
            target_paths.append(catalog_path / "data" / selection["data_type"] / selection["instrument"] / catalog_filename(start, end))

        parquet_done = sum(path.exists() for path in target_paths)
        depth_done = (
            sum(reconstructed_depths_path(catalog_path, selection["instrument"], path.stem).exists() for path in target_paths)
            if selection["data_type"] == "order_book_deltas"
            else 0
        )
        selection["parquet_status"] = (
            "done"
            if parquet_done == selection["files"]
            else "todo"
            if parquet_done == 0
            else f"partial {parquet_done}/{selection['files']}"
        )
        selection["depth_status"] = (
            "-"
            if selection["data_type"] != "order_book_deltas"
            else "done"
            if depth_done == selection["files"]
            else "todo"
            if depth_done == 0
            else f"partial {depth_done}/{selection['files']}"
        )

    if selected:
        header = (
            f"{'instance':36}  {'data_type':18}  {'instrument':28}  "
            f"{'parquet':>12}  {'depth':>12}  "
            f"{'size':>10}  {'instance_mtime':19}  {'feather_mtime':19}"
        )
        print(header)
        print("-" * len(header))
        for selection in selected:
            size = float(selection["bytes"])
            size_text = f"{size:.1f}B"
            for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
                size_text = f"{size:.1f}{unit}"
                if size < 1024.0 or unit == "TiB":
                    break
                size /= 1024.0
            print(
                f"{selection['instance_id']:36}  "
                f"{selection['data_type']:18}  "
                f"{selection['instrument']:28}  "
                f"{selection['parquet_status']:>12}  "
                f"{selection['depth_status']:>12}  "
                f"{size_text:>10}  "
                f"{datetime.fromtimestamp(selection['instance_mtime_ns'] / 1_000_000_000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'):19}  "
                f"{datetime.fromtimestamp(selection['feather_mtime_ns'] / 1_000_000_000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'):19}"
            )
    else:
        print("No matching live feather files.")

    if args.list:
        return
    if not selected:
        raise FileNotFoundError("No matching live feather files")

    pending_conversions = []
    delta_segments = {}
    skipped_conversions = 0
    for selection in selected:
        for feather_path in selection["paths"]:
            start, end = feather_ts_init_bounds(feather_path)
            target_path = catalog_path / "data" / selection["data_type"] / selection["instrument"] / catalog_filename(start, end)
            if selection["data_type"] == "order_book_deltas":
                delta_segments[(selection["instrument"], target_path.stem)] = target_path
            if target_path.exists():
                skipped_conversions += 1
                print(f"skip existing target: {target_path}")
                continue
            pending_conversions.append((selection, feather_path, target_path))

    print(f"conversion plan pending={len(pending_conversions)} skipped_existing={skipped_conversions}")
    for selection, feather_path, target_path in pending_conversions:
        print(f"convert {feather_path} -> {target_path}")
        write_stream_parquet(read_feather_table(feather_path), target_path)

    pending_reconstructs_by_instrument = {}
    delta_segments_by_instrument = {}
    skipped_reconstructs = 0
    for instrument, segment_key in sorted(delta_segments):
        delta_segments_by_instrument.setdefault(instrument, []).append(segment_key)
        target_path = reconstructed_depths_path(catalog_path, instrument, segment_key)
        if target_path.exists() and not args.overwrite_reconstruct:
            skipped_reconstructs += 1
        else:
            pending_reconstructs_by_instrument.setdefault(instrument, []).append((segment_key, target_path))

    if not delta_segments:
        print("reconstruct plan pending=0 skipped_no_order_book_deltas")
        return

    pending_count = sum(len(values) for values in pending_reconstructs_by_instrument.values())
    print(f"reconstruct plan pending={pending_count} skipped_existing={skipped_reconstructs}")
    for instrument, pending_reconstructs in sorted(pending_reconstructs_by_instrument.items()):
        segment_keys = sorted(delta_segments_by_instrument[instrument])
        print(f"reconstruct instrument={instrument} segments={len(segment_keys)}")
        depths = reconstruct_depths(load_instrument_delta_segments(catalog_path, instrument, segment_keys), instrument)
        depth_groups = {segment_key: frame.drop(columns=["_segment_key", "_segment_index"], errors="ignore") for segment_key, frame in depths.groupby("_segment_key", sort=False)}
        for segment_key, target_path in pending_reconstructs:
            if segment_key not in depth_groups:
                raise ValueError(f"No valid depth snapshots reconstructed for {instrument} segment={segment_key}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = target_path.with_suffix(".tmp.parquet")
            segment_depths = depth_groups[segment_key].reset_index(drop=True)
            print(f"write reconstructed depths rows={len(segment_depths)} path={target_path}")
            segment_depths.to_parquet(tmp_path, index=False)
            tmp_path.replace(target_path)


if __name__ == "__main__":
    main()
