//! Shared types and traits for glitz-quant Rust components.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Bid,
    Ask,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Venue {
    Coinbase,
    Kraken,
    BinanceUs,
    Binance,
    Bybit,
    Hyperliquid,
    Jupiter,
    Drift,
    Paper,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Symbol(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub venue: Venue,
    pub symbol: Symbol,
    pub price: Decimal,
    pub size: Decimal,
    pub side: Side,
    pub ts_exchange_ns: i64,
    pub ts_local_ns: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookLevel {
    pub price: Decimal,
    pub size: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookUpdate {
    pub venue: Venue,
    pub symbol: Symbol,
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
    pub ts_exchange_ns: i64,
    pub ts_local_ns: i64,
    pub is_snapshot: bool,
}
