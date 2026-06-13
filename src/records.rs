use std::{collections::BTreeMap, str::FromStr};

use anyhow::{Context, Result, bail};
use nautilus_model::{
    data::{OrderBookDeltas, QuoteTick, TradeTick},
    enums::AggressorSide,
    orderbook::OrderBook,
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteRecord {
    pub record_type: &'static str,
    pub instrument_id: String,
    pub bid_price: String,
    pub ask_price: String,
    pub bid_size: String,
    pub ask_size: String,
    pub ts_event: u64,
    pub ts_init: u64,
}

impl From<&QuoteTick> for QuoteRecord {
    fn from(quote: &QuoteTick) -> Self {
        Self {
            record_type: "quote",
            instrument_id: quote.instrument_id.to_string(),
            bid_price: quote.bid_price.to_string(),
            ask_price: quote.ask_price.to_string(),
            bid_size: quote.bid_size.to_string(),
            ask_size: quote.ask_size.to_string(),
            ts_event: quote.ts_event.as_u64(),
            ts_init: quote.ts_init.as_u64(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub record_type: &'static str,
    pub instrument_id: String,
    pub price: String,
    pub size: String,
    pub aggressor_side: String,
    pub trade_id: String,
    pub ts_event: u64,
    pub ts_init: u64,
}

impl From<&TradeTick> for TradeRecord {
    fn from(trade: &TradeTick) -> Self {
        Self {
            record_type: "trade",
            instrument_id: trade.instrument_id.to_string(),
            price: trade.price.to_string(),
            size: trade.size.to_string(),
            aggressor_side: trade.aggressor_side.to_string(),
            trade_id: trade.trade_id.to_string(),
            ts_event: trade.ts_event.as_u64(),
            ts_init: trade.ts_init.as_u64(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaRecord {
    pub record_type: &'static str,
    pub instrument_id: String,
    pub action: String,
    pub side: String,
    pub price: String,
    pub size: String,
    pub order_id: u64,
    pub flags: u8,
    pub sequence: u64,
    pub ts_event: u64,
    pub ts_init: u64,
}

impl DeltaRecord {
    pub fn from_deltas(deltas: &OrderBookDeltas) -> Vec<Self> {
        deltas
            .deltas
            .iter()
            .map(|delta| Self {
                record_type: "book_delta",
                instrument_id: delta.instrument_id.to_string(),
                action: delta.action.to_string(),
                side: delta.order.side.to_string(),
                price: delta.order.price.to_string(),
                size: delta.order.size.to_string(),
                order_id: delta.order.order_id,
                flags: delta.flags,
                sequence: delta.sequence,
                ts_event: delta.ts_event.as_u64(),
                ts_init: delta.ts_init.as_u64(),
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthSnapshotRecord {
    pub record_type: &'static str,
    pub instrument_id: String,
    pub bids: Vec<BookLevelRecord>,
    pub asks: Vec<BookLevelRecord>,
    pub bid_depth: String,
    pub ask_depth: String,
    pub mid: Option<String>,
    pub spread: Option<String>,
    pub microprice: Option<String>,
    pub imbalance: Option<String>,
    pub sequence: u64,
    pub ts_event: u64,
    pub ts_init: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookLevelRecord {
    pub price: String,
    pub size: String,
    pub count: u32,
}

impl DepthSnapshotRecord {
    pub fn from_book(book: &OrderBook, depth: usize) -> Self {
        let bids: Vec<BookLevelRecord> = book
            .bids(Some(depth))
            .map(|level| BookLevelRecord {
                price: level.price.value.to_string(),
                size: level.size_decimal().to_string(),
                count: level.len() as u32,
            })
            .collect();
        let asks: Vec<BookLevelRecord> = book
            .asks(Some(depth))
            .map(|level| BookLevelRecord {
                price: level.price.value.to_string(),
                size: level.size_decimal().to_string(),
                count: level.len() as u32,
            })
            .collect();

        let bid_depth = sum_sizes(&bids);
        let ask_depth = sum_sizes(&asks);
        let best_bid = bids.first();
        let best_ask = asks.first();
        let mid = match (best_bid, best_ask) {
            (Some(bid), Some(ask)) => Some(((dec(&bid.price) + dec(&ask.price)) / Decimal::TWO).to_string()),
            _ => None,
        };
        let spread = match (best_bid, best_ask) {
            (Some(bid), Some(ask)) => Some((dec(&ask.price) - dec(&bid.price)).to_string()),
            _ => None,
        };
        let microprice = match (best_bid, best_ask) {
            (Some(bid), Some(ask)) => {
                let bid_size = dec(&bid.size);
                let ask_size = dec(&ask.size);
                let denom = bid_size + ask_size;
                if denom > Decimal::ZERO {
                    Some(((dec(&ask.price) * bid_size + dec(&bid.price) * ask_size) / denom).to_string())
                } else {
                    None
                }
            }
            _ => None,
        };
        let imbalance = if bid_depth + ask_depth > Decimal::ZERO {
            Some(((bid_depth - ask_depth) / (bid_depth + ask_depth)).to_string())
        } else {
            None
        };

        Self {
            record_type: "depth10",
            instrument_id: book.instrument_id.to_string(),
            bids,
            asks,
            bid_depth: bid_depth.to_string(),
            ask_depth: ask_depth.to_string(),
            mid,
            spread,
            microprice,
            imbalance,
            sequence: book.sequence,
            ts_event: book.ts_last.as_u64(),
            ts_init: now_ns(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalEvent {
    pub record_type: String,
    pub instrument_id: String,
    pub bids: Vec<BookLevelRecord>,
    pub asks: Vec<BookLevelRecord>,
    pub price: Option<String>,
    pub size: Option<String>,
    pub side: Option<String>,
    pub trade_id: Option<String>,
    pub ts_event: u64,
    pub ts_init: u64,
}

pub fn parse_historical_line(source: &str, instrument_id: &str, line: &str) -> Result<Vec<HistoricalEvent>> {
    let value: Value = serde_json::from_str(line).context("invalid JSON line")?;
    match source {
        "tardis" => parse_tardis(instrument_id, &value),
        "hyperliquid-archive" => parse_hyperliquid_archive(instrument_id, &value),
        other => bail!("unsupported source {other}; use tardis or hyperliquid-archive"),
    }
}

fn parse_tardis(instrument_id: &str, value: &Value) -> Result<Vec<HistoricalEvent>> {
    let local_ts = value
        .get("local_timestamp")
        .or_else(|| value.get("timestamp"))
        .and_then(Value::as_str)
        .and_then(parse_timestamp_ns)
        .unwrap_or_else(now_ns);

    let channel = value.get("channel").and_then(Value::as_str).unwrap_or_default();
    if channel.contains("trade") || value.get("price").is_some() && value.get("amount").is_some() {
        let price = string_field(value, &["price", "px"])?;
        let size = string_field(value, &["amount", "sz", "size"])?;
        let side = value.get("side").and_then(Value::as_str).unwrap_or("UNKNOWN").to_string();
        let trade_id = value
            .get("id")
            .or_else(|| value.get("tid"))
            .and_then(|v| v.as_str().map(str::to_string).or_else(|| v.as_u64().map(|x| x.to_string())))
            .unwrap_or_else(|| local_ts.to_string());
        return Ok(vec![HistoricalEvent {
            record_type: "trade".to_string(),
            instrument_id: instrument_id.to_string(),
            bids: vec![],
            asks: vec![],
            price: Some(price),
            size: Some(size),
            side: Some(side),
            trade_id: Some(trade_id),
            ts_event: local_ts,
            ts_init: local_ts,
        }]);
    }

    parse_book_snapshot(instrument_id, value, local_ts)
}

fn parse_hyperliquid_archive(instrument_id: &str, value: &Value) -> Result<Vec<HistoricalEvent>> {
    let ts = value
        .get("time")
        .or_else(|| value.get("timestamp"))
        .and_then(|v| v.as_u64().or_else(|| v.as_i64().map(|x| x as u64)))
        .map(ms_or_ns_to_ns)
        .unwrap_or_else(now_ns);
    parse_book_snapshot(instrument_id, value, ts)
}

fn parse_book_snapshot(instrument_id: &str, value: &Value, ts: u64) -> Result<Vec<HistoricalEvent>> {
    let levels = value.get("levels").context("book record missing levels")?;
    let (bid_values, ask_values) = if let Some(arr) = levels.as_array() {
        (arr.first().cloned().unwrap_or(Value::Array(vec![])), arr.get(1).cloned().unwrap_or(Value::Array(vec![])))
    } else {
        (
            levels.get("bids").cloned().unwrap_or(Value::Array(vec![])),
            levels.get("asks").cloned().unwrap_or(Value::Array(vec![])),
        )
    };

    let bids = parse_levels(&bid_values)?;
    let asks = parse_levels(&ask_values)?;
    Ok(vec![HistoricalEvent {
        record_type: "depth10".to_string(),
        instrument_id: instrument_id.to_string(),
        bids,
        asks,
        price: None,
        size: None,
        side: None,
        trade_id: None,
        ts_event: ts,
        ts_init: ts,
    }])
}

fn parse_levels(value: &Value) -> Result<Vec<BookLevelRecord>> {
    let mut out = Vec::new();
    for level in value.as_array().context("book side is not an array")?.iter().take(10) {
        let price = string_field(level, &["px", "price"])?;
        let size = string_field(level, &["sz", "size", "amount"])?;
        let count = level.get("n").or_else(|| level.get("count")).and_then(Value::as_u64).unwrap_or(1) as u32;
        out.push(BookLevelRecord { price, size, count });
    }
    Ok(out)
}

fn string_field(value: &Value, names: &[&str]) -> Result<String> {
    for name in names {
        if let Some(v) = value.get(*name) {
            if let Some(s) = v.as_str() {
                return Ok(s.to_string());
            }
            if let Some(n) = v.as_f64() {
                return Ok(n.to_string());
            }
            if let Some(n) = v.as_i64() {
                return Ok(n.to_string());
            }
            if let Some(n) = v.as_u64() {
                return Ok(n.to_string());
            }
        }
    }
    bail!("missing any field from {names:?}")
}

fn parse_timestamp_ns(value: &str) -> Option<u64> {
    chrono::DateTime::parse_from_rfc3339(value)
        .ok()
        .and_then(|dt| dt.timestamp_nanos_opt())
        .map(|ns| ns as u64)
}

fn ms_or_ns_to_ns(value: u64) -> u64 {
    if value < 10_000_000_000_000 {
        value * 1_000_000
    } else {
        value
    }
}

fn now_ns() -> u64 {
    chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default() as u64
}

fn sum_sizes(levels: &[BookLevelRecord]) -> Decimal {
    levels.iter().fold(Decimal::ZERO, |acc, level| acc + dec(&level.size))
}

fn dec(value: &str) -> Decimal {
    Decimal::from_str(value).unwrap_or(Decimal::ZERO)
}

pub fn side_to_aggressor(side: &str) -> AggressorSide {
    match side.to_ascii_lowercase().as_str() {
        "buy" | "buyer" | "b" => AggressorSide::Buyer,
        "sell" | "seller" | "s" => AggressorSide::Seller,
        _ => AggressorSide::NoAggressor,
    }
}

pub fn parse_flat_bbo(value: &Value) -> Option<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for key in ["bid_price", "ask_price", "bid_size", "ask_size"] {
        let value = value.get(key)?.as_str()?.to_string();
        out.insert(key.to_string(), value);
    }
    Some(out)
}
