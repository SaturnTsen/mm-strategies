use std::{
    fs::{File, OpenOptions, create_dir_all},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::Serialize;

pub struct JsonlWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    count: usize,
    flush_every: usize,
}

impl JsonlWriter {
    pub fn new(path: impl AsRef<Path>, flush_every: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            count: 0,
            flush_every,
        })
    }

    pub fn write<T: Serialize>(&mut self, record: &T) -> Result<()> {
        serde_json::to_writer(&mut self.writer, record)
            .with_context(|| format!("serializing {}", self.path.display()))?;
        self.writer.write_all(b"\n")?;
        self.count += 1;
        if self.count % self.flush_every == 0 {
            self.flush()?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush().with_context(|| format!("flushing {}", self.path.display()))
    }
}

pub struct RecordingWriters {
    pub quotes: JsonlWriter,
    pub trades: JsonlWriter,
    pub deltas: JsonlWriter,
    pub depth10: JsonlWriter,
}

impl RecordingWriters {
    pub fn new(out_dir: impl AsRef<Path>) -> Result<Self> {
        let out_dir = out_dir.as_ref();
        Ok(Self {
            quotes: JsonlWriter::new(out_dir.join("quotes.jsonl"), 100)?,
            trades: JsonlWriter::new(out_dir.join("trades.jsonl"), 100)?,
            deltas: JsonlWriter::new(out_dir.join("book_deltas.jsonl"), 100)?,
            depth10: JsonlWriter::new(out_dir.join("depth10.jsonl"), 20)?,
        })
    }

    pub fn flush_all(&mut self) -> Result<()> {
        self.quotes.flush()?;
        self.trades.flush()?;
        self.deltas.flush()?;
        self.depth10.flush()?;
        Ok(())
    }
}
