// Experimental rust live recorder for Hyperliquid market data.
// This is a temporary workaround until the official LiveNode streaming recorder is available.

use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Debug,
    fs,
    io::Cursor,
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, mpsc},
    thread,
    time::{Duration, Instant},
};

use anyhow::Result;
use arrow::ipc::reader::StreamReader;
use clap::Parser;
use log::LevelFilter;
use mm_strategies::{DEFAULT_CATALOG_DIR, DEFAULT_INSTRUMENT, DEFAULT_SNAPSHOT_INTERVAL_MS};
use nautilus_common::{
    actor::{DataActor, DataActorConfig, DataActorCore},
    enums::Environment,
    live::clock::LiveClock,
    logging::logger::LoggerConfig,
    msgbus::{
        mstr::MStr, subscribe_book_deltas, subscribe_book_depth10, subscribe_quotes,
        subscribe_trades, typed_handler::TypedHandler, unsubscribe_book_deltas,
        unsubscribe_book_depth10, unsubscribe_quotes, unsubscribe_trades,
    },
};
use nautilus_hyperliquid::{HyperliquidDataClientConfig, HyperliquidDataClientFactory};
use nautilus_live::node::LiveNode;
use nautilus_model::{
    data::{
        BookOrder, Data, HasTsInit, OrderBookDelta, OrderBookDeltas, OrderBookDepth10, QuoteTick,
        TradeTick,
    },
    enums::{BookAction, BookType, RecordFlag},
    identifiers::{ActorId, InstrumentId, TraderId},
};
use nautilus_persistence::backend::{
    catalog::{CatalogPathPrefix, ParquetDataCatalog},
    feather::{FeatherWriter, RotationConfig},
};
use nautilus_serialization::arrow::{DecodeDataFromRecordBatch, EncodeToRecordBatch};
use object_store::local::LocalFileSystem;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = DEFAULT_INSTRUMENT)]
    instrument: String,
    #[arg(long, default_value = DEFAULT_CATALOG_DIR)]
    catalog: PathBuf,
    #[arg(long, default_value = "hl-live")]
    instance_id: String,
    #[arg(long, default_value_t = DEFAULT_SNAPSHOT_INTERVAL_MS)]
    snapshot_interval_ms: u64,
    #[arg(long, default_value_t = 10)]
    depth: usize,
    #[arg(long, default_value_t = 1000)]
    flush_interval_ms: u64,
    #[arg(long)]
    duration_secs: Option<u64>,
    #[arg(long, default_value_t = false)]
    testnet: bool,
}

#[derive(Debug)]
struct SubscriptionActor {
    core: DataActorCore,
    instrument_id: InstrumentId,
    snapshot_interval_ms: NonZeroUsize,
    depth: usize,
}

impl SubscriptionActor {
    fn new(instrument_id: InstrumentId, snapshot_interval_ms: u64, depth: usize) -> Self {
        let config = DataActorConfig {
            actor_id: Some(ActorId::from("HL-RECORDER-001")),
            ..Default::default()
        };
        Self {
            core: DataActorCore::new(config),
            instrument_id,
            snapshot_interval_ms: NonZeroUsize::new(snapshot_interval_ms as usize).unwrap_or_else(
                || NonZeroUsize::new(DEFAULT_SNAPSHOT_INTERVAL_MS as usize).unwrap(),
            ),
            depth,
        }
    }
}

impl Deref for SubscriptionActor {
    type Target = DataActorCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl DerefMut for SubscriptionActor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.core
    }
}

impl DataActor for SubscriptionActor {
    fn on_start(&mut self) -> Result<()> {
        self.subscribe_book_deltas(
            self.instrument_id,
            BookType::L2_MBP,
            NonZeroUsize::new(self.depth),
            None,
            true,
            None,
        );
        self.subscribe_book_at_interval(
            self.instrument_id,
            BookType::L2_MBP,
            NonZeroUsize::new(self.depth),
            self.snapshot_interval_ms,
            None,
            None,
        );
        self.subscribe_quotes(self.instrument_id, None, None);
        self.subscribe_trades(self.instrument_id, None, None);
        log::info!("Subscribed to {}", self.instrument_id);
        Ok(())
    }
}

enum WriterMessage {
    Quote(QuoteTick),
    Trade(TradeTick),
    OrderBookDelta(OrderBookDelta),
    OrderBookDepth(OrderBookDepth10),
}

fn send_writer_message(tx: &mpsc::Sender<WriterMessage>, message: WriterMessage) {
    tx.send(message)
        .expect("feather writer thread stopped while recorder is running");
}

struct FeatherWriterSubscriptions {
    deltas: TypedHandler<OrderBookDeltas>,
    depth10: TypedHandler<OrderBookDepth10>,
    quotes: TypedHandler<QuoteTick>,
    trades: TypedHandler<TradeTick>,
}

fn subscribe_feather_writer(tx: mpsc::Sender<WriterMessage>) -> FeatherWriterSubscriptions {
    let deltas_tx = tx.clone();
    let deltas = TypedHandler::from(move |deltas: &OrderBookDeltas| {
        for delta in normalize_snapshot_deltas(deltas) {
            send_writer_message(&deltas_tx, WriterMessage::OrderBookDelta(delta));
        }
    });
    subscribe_book_deltas(MStr::pattern("data.book.deltas.*"), deltas.clone(), None);

    let depth10_tx = tx.clone();
    let depth10 = TypedHandler::from(move |depth: &OrderBookDepth10| {
        send_writer_message(&depth10_tx, WriterMessage::OrderBookDepth(*depth));
    });
    subscribe_book_depth10(MStr::pattern("data.book.depth10.*"), depth10.clone(), None);

    let quotes_tx = tx.clone();
    let quotes = TypedHandler::from(move |quote: &QuoteTick| {
        send_writer_message(&quotes_tx, WriterMessage::Quote(*quote));
    });
    subscribe_quotes(MStr::pattern("data.quotes.*"), quotes.clone(), None);

    let trades = TypedHandler::from(move |trade: &TradeTick| {
        send_writer_message(&tx, WriterMessage::Trade(*trade));
    });
    subscribe_trades(MStr::pattern("data.trades.*"), trades.clone(), None);

    FeatherWriterSubscriptions {
        deltas,
        depth10,
        quotes,
        trades,
    }
}

fn normalize_snapshot_deltas(deltas: &OrderBookDeltas) -> Vec<OrderBookDelta> {
    if !deltas
        .deltas
        .first()
        .is_some_and(|delta| delta.action == BookAction::Clear)
    {
        return deltas.deltas.clone();
    }

    let mut normalized = Vec::with_capacity(deltas.deltas.len());
    let last_add_index = deltas
        .deltas
        .iter()
        .rposition(|delta| delta.action == BookAction::Add);
    let mut order_id = 1_u64;

    for (index, delta) in deltas.deltas.iter().enumerate() {
        if delta.action == BookAction::Add {
            let flags = if Some(index) == last_add_index {
                RecordFlag::F_SNAPSHOT as u8 | RecordFlag::F_LAST as u8
            } else {
                RecordFlag::F_SNAPSHOT as u8
            };
            let order = BookOrder::new(
                delta.order.side,
                delta.order.price,
                delta.order.size,
                order_id,
            );
            normalized.push(OrderBookDelta::new(
                delta.instrument_id,
                delta.action,
                order,
                flags,
                delta.sequence,
                delta.ts_event,
                delta.ts_init,
            ));
            order_id += 1;
        } else {
            normalized.push(*delta);
        }
    }

    normalized
}

fn unsubscribe_feather_writer(subscriptions: &FeatherWriterSubscriptions) {
    unsubscribe_book_deltas(MStr::pattern("data.book.deltas.*"), &subscriptions.deltas);
    unsubscribe_book_depth10(MStr::pattern("data.book.depth10.*"), &subscriptions.depth10);
    unsubscribe_quotes(MStr::pattern("data.quotes.*"), &subscriptions.quotes);
    unsubscribe_trades(MStr::pattern("data.trades.*"), &subscriptions.trades);
}

fn spawn_feather_writer(
    catalog: PathBuf,
    stream_path: String,
    flush_interval_ms: u64,
    rx: mpsc::Receiver<WriterMessage>,
) -> thread::JoinHandle<Result<(), String>> {
    thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;

        runtime.block_on(async move {
            let mut writer = FeatherWriter::new(
                stream_path,
                Arc::new(LocalFileSystem::new_with_prefix(catalog).map_err(|e| e.to_string())?),
                Rc::new(RefCell::new(LiveClock::default())),
                RotationConfig::NoRotation,
                Some(HashSet::from([
                    "order_book_deltas".to_string(),
                    "order_book_depths".to_string(),
                    "quotes".to_string(),
                    "trades".to_string(),
                ])),
                Some(HashSet::from([
                    "order_book_deltas".to_string(),
                    "order_book_depths".to_string(),
                    "quotes".to_string(),
                    "trades".to_string(),
                ])),
                Some(flush_interval_ms),
            );

            let mut messages = 0_u64;
            let mut file_paths = HashSet::new();
            let mut last_log = Instant::now();
            let log_interval = Duration::from_secs(5);

            log::info!("Feather writer thread started");

            loop {
                let message = match rx.recv_timeout(log_interval) {
                    Ok(message) => message,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        log::info!(
                            "Feather writer progress: messages={}, files={}",
                            messages,
                            file_paths.len()
                        );
                        last_log = Instant::now();
                        continue;
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                };

                match message {
                    WriterMessage::Quote(data) => writer.write(data).await,
                    WriterMessage::Trade(data) => writer.write(data).await,
                    WriterMessage::OrderBookDelta(data) => writer.write(data).await,
                    WriterMessage::OrderBookDepth(data) => writer.write(data).await,
                }
                .map_err(|e| e.to_string())?;

                messages += 1;
                for (_, path) in writer.get_current_file_info().values() {
                    file_paths.insert(path.clone());
                }

                if last_log.elapsed() >= log_interval {
                    log::info!(
                        "Feather writer progress: messages={}, files={}",
                        messages,
                        file_paths.len()
                    );
                    last_log = Instant::now();
                }
            }

            log::info!(
                "Feather writer closing: messages={}, files={}",
                messages,
                file_paths.len()
            );
            writer.close().await.map_err(|e| e.to_string())?;
            log::info!(
                "Feather writer closed: messages={}, files={}",
                messages,
                file_paths.len()
            );
            Ok(())
        })
    })
}

fn convert_live_stream_to_parquet(catalog_path: PathBuf, instance_id: &str) -> Result<()> {
    let instance_id = instance_id.to_string();
    let handle = thread::spawn(move || -> Result<(), String> {
        let mut catalog = ParquetDataCatalog::new(catalog_path.clone(), None, None, None, None);

        let deltas = convert_feather_files::<OrderBookDelta>(
            &mut catalog,
            &catalog_path,
            &instance_id,
            "order_book_deltas",
        )
        .map_err(|e| e.to_string())?;
        let depth = convert_feather_files::<OrderBookDepth10>(
            &mut catalog,
            &catalog_path,
            &instance_id,
            "order_book_depths",
        )
        .map_err(|e| e.to_string())?;
        let quotes =
            convert_feather_files::<QuoteTick>(&mut catalog, &catalog_path, &instance_id, "quotes")
                .map_err(|e| e.to_string())?;
        let trades =
            convert_feather_files::<TradeTick>(&mut catalog, &catalog_path, &instance_id, "trades")
                .map_err(|e| e.to_string())?;

        log::info!(
            "Converted live stream to parquet catalog: order_book_deltas={}, order_book_depths={}, quotes={}, trades={}",
            deltas,
            depth,
            quotes,
            trades
        );
        Ok(())
    });

    handle
        .join()
        .map_err(|_| anyhow::anyhow!("parquet conversion thread panicked"))?
        .map_err(|e| anyhow::anyhow!("failed to convert live stream to parquet: {e}"))?;
    Ok(())
}

fn convert_feather_files<T>(
    catalog: &mut ParquetDataCatalog,
    catalog_path: &Path,
    instance_id: &str,
    data_cls: &str,
) -> Result<usize>
where
    T: DecodeDataFromRecordBatch
        + TryFrom<Data>
        + HasTsInit
        + EncodeToRecordBatch
        + CatalogPathPrefix,
{
    let files = list_live_feather_files(catalog_path, instance_id, data_cls)?;
    log::info!(
        "Converting live stream to parquet: {data_cls}, files={}",
        files.len()
    );

    let file_count = files.len();
    let mut data = Vec::new();
    for file in files {
        data.extend(read_feather_file::<T>(&file)?);
    }

    if !data.is_empty() {
        data.sort_by_key(|item| item.ts_init());
        catalog.write_to_parquet(data, None, None, None)?;
    }

    Ok(file_count)
}

fn list_live_feather_files(
    catalog_path: &Path,
    instance_id: &str,
    data_cls: &str,
) -> Result<Vec<PathBuf>> {
    let run_dir = catalog_path.join("live").join(instance_id);
    let mut files = Vec::new();

    collect_feather_files(&run_dir.join(data_cls), &mut files)?;

    files.sort();
    Ok(files)
}

fn collect_feather_files(directory: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if !directory.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_feather_files(&path, files)?;
        } else if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension == "feather")
        {
            files.push(path);
        }
    }

    Ok(())
}

fn read_feather_file<T>(path: &Path) -> Result<Vec<T>>
where
    T: DecodeDataFromRecordBatch + TryFrom<Data>,
{
    let bytes = fs::read(path)?;
    let reader = StreamReader::try_new(Cursor::new(bytes), None)?;
    let mut output = Vec::new();

    for batch in reader {
        let batch = batch?;
        let metadata = batch.schema().metadata().clone();
        let data = T::decode_data_batch(&metadata, batch)?;

        for item in data {
            output.push(
                T::try_from(item)
                    .map_err(|_| anyhow::anyhow!("decoded feather data has unexpected type"))?,
            );
        }
    }

    Ok(output)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let trader_id = TraderId::from("HLREC-001");
    let instrument_id = InstrumentId::from(args.instrument.as_str());
    let stream_path = PathBuf::from("live")
        .join(&args.instance_id)
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("stream path is not valid UTF-8"))?
        .to_string();
    std::fs::create_dir_all(&args.catalog)?;

    let (writer_tx, writer_rx) = mpsc::channel();
    let writer_handle = spawn_feather_writer(
        args.catalog.clone(),
        stream_path,
        args.flush_interval_ms,
        writer_rx,
    );

    let mut node = LiveNode::builder(trader_id, Environment::Live)?
        .with_name("HL-MARKET-DATA-V1".to_string())
        .with_logging(LoggerConfig {
            stdout_level: LevelFilter::Info,
            ..Default::default()
        })
        .with_delay_post_stop_secs(2)
        .add_data_client(
            None,
            Box::new(HyperliquidDataClientFactory::new()),
            Box::new(HyperliquidDataClientConfig {
                is_testnet: args.testnet,
                ..Default::default()
            }),
        )?
        .build()?;

    let streaming_handler = subscribe_feather_writer(writer_tx);

    node.add_actor(SubscriptionActor::new(
        instrument_id,
        args.snapshot_interval_ms,
        args.depth,
    ))?;

    if let Some(duration_secs) = args.duration_secs {
        let handle = node.handle();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(duration_secs)).await;
            handle.stop();
        });
    }

    let run_result = node.run().await;
    unsubscribe_feather_writer(&streaming_handler);
    drop(streaming_handler);
    let writer_result = writer_handle
        .join()
        .map_err(|_| anyhow::anyhow!("feather writer thread panicked"))?;
    run_result?;
    writer_result.map_err(|e| anyhow::anyhow!("feather writer failed: {e}"))?;
    convert_live_stream_to_parquet(args.catalog, &args.instance_id)?;
    Ok(())
}
