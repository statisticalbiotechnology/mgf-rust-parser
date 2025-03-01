use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::record_batch::RecordBatch;
use arrow_array::RecordBatchIterator; // from arrow_array v51.0.0
use clap::{Parser, ValueEnum, ValueHint};
use indicatif::{ProgressBar, ProgressStyle};
use lance::dataset::{Dataset, WriteMode, WriteParams};

mod read_mgf;
use read_mgf::{MGFConfig, parse_mgf_files};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "lowercase")]
enum WriteModeOption {
    Write,
    Append,
    Overwrite,
}

/// CLI arguments using Clap.
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Parse multiple MGF files (or directories) into Arrow RecordBatches and write to a Lance dataset."
)]
struct Cli {
    /// One or more MGF files or directories to parse. If a directory, recursively search for `.mgf` files.
    #[arg(
        short = 'f',
        long = "file",
        help = "MGF file(s) or directory(ies)",
        value_hint = ValueHint::AnyPath,
        num_args = 1..,
        required = true
    )]
    files: Vec<PathBuf>,

    /// The number of spectra to accumulate per RecordBatch.
    #[arg(long, default_value = "1000", help = "Batch size")]
    batch_size: usize,

    /// Ignore any MGF spectrum with fewer than this many peaks.
    #[arg(long, default_value = "1", help = "Minimum peak count per spectrum")]
    min_peaks: usize,

    /// The size of the bounded channel used in the pipeline.
    #[arg(long, default_value = "8", help = "Channel capacity")]
    channel_capacity: usize,

    /// Output path for the Lance dataset.
    #[arg(
        long = "output-lance",
        value_hint = ValueHint::AnyPath,
        help = "Output Lance dataset path",
        required = true
    )]
    output_lance: PathBuf,

    /// Write mode (case‑insensitive): write, append, or overwrite.
    #[arg(long, default_value = "write", value_enum)]
    write_mode: WriteModeOption,

    /// Optional YAML file defining field prefixes (overrides defaults).
    #[arg(
        long = "fields-config",
        value_hint = ValueHint::AnyPath,
        help = "YAML configuration file for MGF field prefixes",
        required = false
    )]
    fields_config: Option<PathBuf>,
}

/// An adapter that wraps an iterator over RecordBatches and updates a progress bar.
struct ProgressRecordBatchIterator<I> {
    inner: I,
    pb: ProgressBar,
    total: u64,
    start: Instant,
}

impl<I> ProgressRecordBatchIterator<I> {
    fn new(inner: I) -> Self {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ ")
                .template("{spinner:.green} {msg}")
                .expect("Failed to set progress bar template"),
        );
        pb.enable_steady_tick(Duration::from_millis(100));
        Self {
            inner,
            pb,
            total: 0,
            start: Instant::now(),
        }
    }
}

impl<I> Iterator for ProgressRecordBatchIterator<I>
where
    I: Iterator<Item = RecordBatch>,
{
    type Item = RecordBatch;
    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.inner.next()?;
        let rows = batch.num_rows() as u64;
        self.total += rows;
        let elapsed = self.start.elapsed().as_secs_f64();
        let rate = self.total as f64 / elapsed;
        let msg = format!(
            "{} spectra read ({:.2} per second). Skipped: {} due to min_peaks, {} due to errors",
            self.total,
            rate,
            read_mgf::SKIPPED_MIN.load(std::sync::atomic::Ordering::SeqCst),
            read_mgf::SKIPPED_ERR.load(std::sync::atomic::Ordering::SeqCst)
        );
        // Leak the string so it has a 'static lifetime.
        let static_msg: &'static str = Box::leak(msg.into_boxed_str());
        self.pb.set_message(static_msg);
        Some(batch)
    }
}

impl<I> Drop for ProgressRecordBatchIterator<I> {
    fn drop(&mut self) {
        let final_msg = format!(
            "Done: {} spectra read in {:.2} seconds. Skipped: {} due to min_peaks, {} due to errors",
            self.total,
            self.start.elapsed().as_secs_f64(),
            read_mgf::SKIPPED_MIN.load(std::sync::atomic::Ordering::SeqCst),
            read_mgf::SKIPPED_ERR.load(std::sync::atomic::Ordering::SeqCst)
        );
        let leaked: &'static str = Box::leak(final_msg.into_boxed_str());
        self.pb.finish_with_message(leaked);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 1) Parse CLI arguments.
    let cli = Cli::parse();

    // 2) Load YAML configuration if provided; otherwise, use default configuration.
    let config = if let Some(config_path) = cli.fields_config {
        let contents = fs::read_to_string(&config_path)?;
        serde_yaml::from_str::<MGFConfig>(&contents)?
    } else {
        MGFConfig {
            title_prefix: Some("TITLE=".to_string()),
            pepmass_prefix: Some("PEPMASS=".to_string()),
            rtinseconds_prefix: Some("RTINSECONDS=".to_string()),
            charge_prefix: Some("CHARGE=".to_string()),
            scans_prefix: Some("SCANS=".to_string()),
            seq_prefix: Some("SEQ=".to_string()),
        }
    };
    let config = Arc::new(config);

    // 3) Create the MGF iterator from our read_mgf module.
    let base_iter = parse_mgf_files(
        &cli.files,
        cli.batch_size,
        cli.min_peaks,
        cli.channel_capacity,
        Arc::clone(&config),
    )?;

    // 4) Wrap the iterator in a Peekable adapter to peek at the first batch (for obtaining schema)
    let mut peekable_iter = base_iter.peekable();
    let schema = match peekable_iter.peek() {
        Some(batch) => batch.schema().clone(),
        None => {
            eprintln!("No data found to write.");
            return Ok(());
        }
    };

    // 5) Wrap the iterator with our progress adapter.
    let progress_iter = ProgressRecordBatchIterator::new(peekable_iter);

    // 6) Build a RecordBatchIterator (as expected by Lance) from our progress adapter.
    let batch_iter = RecordBatchIterator::new(progress_iter.map(Ok), Arc::clone(&schema));

    // 7) Set write mode according to the CLI argument.
    let write_mode = match cli.write_mode {
        WriteModeOption::Write => WriteMode::Create,
        WriteModeOption::Append => WriteMode::Append,
        WriteModeOption::Overwrite => WriteMode::Overwrite,
    };

    let write_params = WriteParams {
        mode: write_mode,
        ..Default::default()
    };

    // 8) Write the batches to the Lance dataset.
    println!("Writing to Lance dataset at {}", cli.output_lance.display());
    Dataset::write(
        batch_iter,
        &cli.output_lance.to_string_lossy(),
        Some(write_params),
    )
    .await
    .map_err(|e| format!("Failed to write Lance dataset: {e}"))?;
    println!(
        "Successfully wrote Lance dataset at {}",
        cli.output_lance.display()
    );

    // 9) Print the final skipped spectra counts.
    println!(
        "Skipped spectra: {} due to min_peaks, {} due to errors",
        read_mgf::SKIPPED_MIN.load(std::sync::atomic::Ordering::SeqCst),
        read_mgf::SKIPPED_ERR.load(std::sync::atomic::Ordering::SeqCst)
    );

    Ok(())
}
