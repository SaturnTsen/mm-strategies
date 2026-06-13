use std::{
    fmt::Debug,
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
    path::PathBuf,
    time::Duration,
};

use anyhow::Result;
use clap::Parser;
use hl_market_data_v1::{
    DEFAULT_INSTRUMENT, DEFAULT_OUT_DIR, DEFAULT_SNAPSHOT_INTERVAL_MS,
    records::{DeltaRecord, DepthSnapshotRecord, QuoteRecord, TradeRecord},
    writers::RecordingWriters,
};
use log::LevelFilter;
use nautilus_common::{
    actor::{DataActor, DataActorConfig, DataActorCore},
    enums::Environment,
    logging::logger::LoggerConfig,
};
use nautilus_hyperliquid::{HyperliquidDataClientConfig, HyperliquidDataClientFactory};
use nautilus_live::node::LiveNode;
use nautilus_model::{
    data::{OrderBookDeltas, QuoteTick, TradeTick},
    enums::BookType,
    identifiers::{ActorId, InstrumentId, TraderId},
    orderbook::OrderBook,
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = DEFAULT_INSTRUMENT)]
    instrument: String,
    #[arg(long, default_value = DEFAULT_OUT_DIR)]
    out_dir: PathBuf,
    #[arg(long, default_value_t = DEFAULT_SNAPSHOT_INTERVAL_MS)]
    snapshot_interval_ms: u64,
    #[arg(long, default_value_t = 10)]
    depth: usize,
    #[arg(long)]
    duration_secs: Option<u64>,
    #[arg(long, default_value_t = false)]
    testnet: bool,
}

struct RecorderActor {
    core: DataActorCore,
    instrument_id: InstrumentId,
    writers: RecordingWriters,
    snapshot_interval_ms: NonZeroUsize,
    depth: usize,
}

impl Debug for RecorderActor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecorderActor")
            .field("instrument_id", &self.instrument_id)
            .field("depth", &self.depth)
            .finish()
    }
}

impl RecorderActor {
    fn new(instrument_id: InstrumentId, out_dir: PathBuf, snapshot_interval_ms: u64, depth: usize) -> Result<Self> {
        let config = DataActorConfig {
            actor_id: Some(ActorId::from("HL-RECORDER-001")),
            ..Default::default()
        };
        Ok(Self {
            core: DataActorCore::new(config),
            instrument_id,
            writers: RecordingWriters::new(out_dir)?,
            snapshot_interval_ms: NonZeroUsize::new(snapshot_interval_ms as usize)
                .unwrap_or_else(|| NonZeroUsize::new(DEFAULT_SNAPSHOT_INTERVAL_MS as usize).unwrap()),
            depth,
        })
    }
}

impl Deref for RecorderActor {
    type Target = DataActorCore;

    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl DerefMut for RecorderActor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.core
    }
}

impl DataActor for RecorderActor {
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
        log::info!("Recorder subscribed to {}", self.instrument_id);
        Ok(())
    }

    fn on_book_deltas(&mut self, deltas: &OrderBookDeltas) -> Result<()> {
        for record in DeltaRecord::from_deltas(deltas) {
            self.writers.deltas.write(&record)?;
        }
        Ok(())
    }

    fn on_book(&mut self, order_book: &OrderBook) -> Result<()> {
        let record = DepthSnapshotRecord::from_book(order_book, self.depth);
        self.writers.depth10.write(&record)?;
        Ok(())
    }

    fn on_quote(&mut self, quote: &QuoteTick) -> Result<()> {
        self.writers.quotes.write(&QuoteRecord::from(quote))?;
        Ok(())
    }

    fn on_trade(&mut self, tick: &TradeTick) -> Result<()> {
        self.writers.trades.write(&TradeRecord::from(tick))?;
        Ok(())
    }

    fn on_stop(&mut self) -> Result<()> {
        self.writers.flush_all()?;
        log::info!("Recorder flushed output files");
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let trader_id = TraderId::from("HLREC-001");
    let instrument_id = InstrumentId::from(args.instrument.as_str());

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

    let actor = RecorderActor::new(instrument_id, args.out_dir, args.snapshot_interval_ms, args.depth)?;
    node.add_actor(actor)?;

    if let Some(duration_secs) = args.duration_secs {
        let handle = node.handle();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(duration_secs)).await;
            handle.stop();
        });
    }

    node.run().await?;
    Ok(())
}
