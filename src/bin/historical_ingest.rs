use std::{
    fs::{File, read_dir},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use hl_market_data_v1::{
    DEFAULT_INSTRUMENT, DEFAULT_OUT_DIR,
    records::parse_historical_line,
    writers::JsonlWriter,
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_parser = ["tardis", "hyperliquid-archive"])]
    source: String,
    #[arg(long, default_value = DEFAULT_INSTRUMENT)]
    instrument: String,
    #[arg(long)]
    input: PathBuf,
    #[arg(long, default_value = DEFAULT_OUT_DIR)]
    out_dir: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut depth10 = JsonlWriter::new(args.out_dir.join("depth10.jsonl"), 100)?;
    let mut trades = JsonlWriter::new(args.out_dir.join("trades.jsonl"), 100)?;

    for file in input_files(&args.input)? {
        if file.extension().and_then(|s| s.to_str()).is_some_and(|ext| matches!(ext, "gz" | "lz4")) {
            bail!(
                "{} is compressed; first V1 expects decompressed JSONL input. Decompress it and rerun.",
                file.display()
            );
        }

        let reader = BufReader::new(File::open(&file).with_context(|| format!("opening {}", file.display()))?);
        for (idx, line) in reader.lines().enumerate() {
            let line = line.with_context(|| format!("reading {}:{idx}", file.display()))?;
            if line.trim().is_empty() {
                continue;
            }
            for event in parse_historical_line(&args.source, &args.instrument, &line)
                .with_context(|| format!("parsing {}:{}", file.display(), idx + 1))?
            {
                match event.record_type.as_str() {
                    "depth10" => depth10.write(&event)?,
                    "trade" => trades.write(&event)?,
                    _ => {}
                }
            }
        }
    }

    depth10.flush()?;
    trades.flush()?;
    Ok(())
}

fn input_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut files = Vec::new();
    for entry in read_dir(path).with_context(|| format!("reading input dir {}", path.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}
