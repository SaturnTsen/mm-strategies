#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import (
    ACTION_ADD,
    ACTION_CLEAR,
    ACTION_DELETE,
    ACTION_UPDATE,
    CatalogSegment,
    DEPTH_LEVELS,
    SIDE_BUY,
    SIDE_SELL,
    load_deltas,
    parse_catalog_file_segment,
    reconstructed_depths_path,
    segments_overlap,
)
from nautilus_trader.model.data import OrderBookDeltas, QuoteTick, TradeTick
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.persistence.catalog.parquet import _timestamps_to_filename


DEFAULT_CATALOG_PATH = Path("catalog")
STREAMING_DATA_TYPES_BY_NAME = {
    "quote_tick": QuoteTick,
    "trade_tick": TradeTick,
    "order_book_deltas": OrderBookDeltas,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert live feather data to parquet and reconstruct depth cache from order_book_deltas.",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--instance", default=None, help="Live instance UUID. Use latest to select the newest instance.")
    parser.add_argument("--data-type", action="append", choices=sorted(STREAMING_DATA_TYPES_BY_NAME), default=None)
    parser.add_argument("-i", "--instrument", action="append", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--overwrite-reconstruct", action="store_true")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    catalog_path = args.catalog.expanduser()
    live_path = catalog_path / "live"
    if not live_path.exists():
        raise FileNotFoundError(f"Live catalog directory does not exist: {live_path}")

    selections = []
    for config_path in sorted(live_path.glob("*/config.json")):
        instance_id = config_path.parent.name
        with config_path.open() as file:
            json.load(file)
        for data_type in sorted(STREAMING_DATA_TYPES_BY_NAME):
            data_type_path = config_path.parent / data_type
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
                        "config_mtime_ns": config_path.stat().st_mtime_ns,
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
    if args.data_type is not None:
        selected = [selection for selection in selected if selection["data_type"] in args.data_type]
    if args.instrument is not None:
        selected = [selection for selection in selected if selection["instrument"] in args.instrument]
    selected = sorted(
        selected,
        key=lambda selection: (
            selection["feather_mtime_ns"],
            selection["config_mtime_ns"],
            selection["instance_id"],
            selection["data_type"],
            selection["instrument"],
        ),
    )

    for selection in selected:
        selection_segment = CatalogSegment(
            pd.Timestamp(selection["config_mtime_ns"], unit="ns", tz="UTC"),
            pd.Timestamp(selection["feather_mtime_ns"], unit="ns", tz="UTC"),
        )
        parquet_files = [
            path
            for path in sorted((catalog_path / "data" / selection["data_type"] / selection["instrument"]).glob("*.parquet"))
            if segments_overlap(parse_catalog_file_segment(path), selection_segment)
        ]
        depth_files = [
            path
            for path in sorted((catalog_path / "data" / "reconstruct" / selection["instrument"]).glob("*.parquet"))
            if segments_overlap(parse_catalog_file_segment(path), selection_segment)
        ]
        parquet_done = min(len(parquet_files), selection["files"])
        depth_done = min(len(depth_files), selection["files"]) if selection["data_type"] == "order_book_deltas" else 0
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
            f"{'size':>10}  {'config_mtime':19}  {'feather_mtime':19}"
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
                f"{datetime.fromtimestamp(selection['config_mtime_ns'] / 1_000_000_000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'):19}  "
                f"{datetime.fromtimestamp(selection['feather_mtime_ns'] / 1_000_000_000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'):19}"
            )
    else:
        print("No matching live feather files.")

    if args.list:
        return
    if not selected:
        raise FileNotFoundError("No matching live feather files")

    catalog = ParquetDataCatalog(str(catalog_path))
    pending_conversions = []
    delta_segments = {}
    skipped_conversions = 0
    for selection in selected:
        for feather_path in selection["paths"]:
            start, end = feather_ts_init_bounds(feather_path)
            target_path = catalog_path / "data" / selection["data_type"] / selection["instrument"] / _timestamps_to_filename(start, end)
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
        feather_table = catalog._read_feather_file(str(feather_path))
        if feather_table is None:
            raise RuntimeError(f"Cannot read feather file: {feather_path}")
        catalog._convert_feather_table_to_parquet(
            feather_table=feather_table,
            feather_path=str(feather_path),
            data_cls=STREAMING_DATA_TYPES_BY_NAME[selection["data_type"]],
            used_catalog=catalog,
        )

    pending_reconstructs = []
    skipped_reconstructs = 0
    for instrument, segment_key in sorted(delta_segments):
        target_path = reconstructed_depths_path(catalog_path, instrument, segment_key)
        if target_path.exists() and not args.overwrite_reconstruct:
            skipped_reconstructs += 1
        else:
            pending_reconstructs.append((instrument, segment_key, target_path))

    if not delta_segments:
        print("reconstruct plan pending=0 skipped_no_order_book_deltas")
        return

    print(f"reconstruct plan pending={len(pending_reconstructs)} skipped_existing={skipped_reconstructs}")
    for instrument, segment_key, target_path in pending_reconstructs:
        print(f"reconstruct instrument={instrument} segment={segment_key}")
        depths = reconstruct_depths(load_deltas(catalog_path, instrument, segment_key), f"{instrument} {segment_key}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(".tmp.parquet")
        print(f"write reconstructed depths rows={len(depths)} path={target_path}")
        depths.to_parquet(tmp_path, index=False)
        tmp_path.replace(target_path)


if __name__ == "__main__":
    main()
