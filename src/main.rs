use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::record_batch::RecordBatch;
use arrow_array::RecordBatchIterator; // from arrow_array v50.0.0
use clap::{Parser, ValueEnum, ValueHint};
use indicatif::{ProgressBar, ProgressStyle};
use lance::dataset::{Dataset, WriteMode, WriteParams};

mod read_mgf;

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

    /// Output path for the Lance dataset. This argument is required.
    #[arg(
        long = "output-lance",
        value_hint = ValueHint::AnyPath,
        help = "Output Lance dataset path",
        required = true
    )]
    output_lance: PathBuf,

    #[arg(long, default_value = "write", value_enum)]
    write_mode: WriteModeOption,
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

    // 2) Create the MGF iterator from our read_mgf module.
    let base_iter = read_mgf::parse_mgf_files(
        &cli.files,
        cli.batch_size,
        cli.min_peaks,
        cli.channel_capacity,
    )?;

    // 3) Wrap the base iterator in a Peekable adapter to peek at the first batch without consuming it.
    let mut peekable_iter = base_iter.peekable();
    let schema = match peekable_iter.peek() {
        Some(batch) => batch.schema().clone(),
        None => {
            eprintln!("No data found to write.");
            return Ok(());
        }
    };

    // 4) Wrap the iterator with our progress adapter.
    let progress_iter = ProgressRecordBatchIterator::new(peekable_iter);

    // 5) Build a RecordBatchIterator (expected by Lance) from our progress adapter.
    let batch_iter = RecordBatchIterator::new(progress_iter.map(Ok), Arc::clone(&schema));

    let write_mode = match cli.write_mode {
        WriteModeOption::Write => WriteMode::Create,
        WriteModeOption::Append => WriteMode::Append,
        WriteModeOption::Overwrite => WriteMode::Overwrite,
    };

    let write_params = WriteParams {
        mode: write_mode,
        ..Default::default()
    };

    // 6) Write the batches to the Lance dataset.
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

    // 7) Print the final skipped spectra counts.
    println!(
        "Skipped spectra: {} due to min_peaks, {} due to errors",
        read_mgf::SKIPPED_MIN.load(std::sync::atomic::Ordering::SeqCst),
        read_mgf::SKIPPED_ERR.load(std::sync::atomic::Ordering::SeqCst)
    );

    Ok(())
}
