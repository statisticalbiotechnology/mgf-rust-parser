mod read_mgf;

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::record_batch::RecordBatch;
use arrow_array::RecordBatchIterator; // Import from the crate root.
use clap::{Parser, ValueHint};
use indicatif::{ProgressBar, ProgressStyle};
use lance::dataset::{Dataset, WriteMode, WriteParams};

/// A simple CLI parser using Clap.
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Parse multiple MGF files (or directories) into Arrow RecordBatches and optionally write to a Lance dataset."
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

    /// Optional output path for the Lance dataset.
    #[arg(
        long = "output-lance",
        value_hint = ValueHint::AnyPath,
        help = "Output Lance dataset path",
        required = false
    )]
    output_lance: Option<PathBuf>,
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
        let msg = format!("{} spectra read ({:.2} per second)", self.total, rate);
        // Leak the string so it has a 'static lifetime.
        let static_msg: &'static str = Box::leak(msg.into_boxed_str());
        self.pb.set_message(static_msg);
        Some(batch)
    }
}

impl<I> Drop for ProgressRecordBatchIterator<I> {
    fn drop(&mut self) {
        let final_msg = format!(
            "Done: {} spectra read in {:.2} seconds.",
            self.total,
            self.start.elapsed().as_secs_f64()
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
    let mut iter = read_mgf::parse_mgf_files(
        &cli.files,
        cli.batch_size,
        cli.min_peaks,
        cli.channel_capacity,
    )?;

    // 3) If an output path is provided, write batches to a Lance dataset.
    if let Some(lance_path) = cli.output_lance {
        // To obtain the schema, take the first batch.
        if let Some(first_batch) = iter.next() {
            let schema = first_batch.schema();
            // Create an iterator that yields the first batch, then the rest.
            let full_iter = std::iter::once(first_batch).chain(iter);
            // Wrap our iterator with the progress adapter.
            let progress_iter = ProgressRecordBatchIterator::new(full_iter);
            // Create a RecordBatchIterator (from arrow_array) as expected by Dataset::write.
            let batch_iter = RecordBatchIterator::new(progress_iter.map(Ok), Arc::clone(&schema));
            let write_params = WriteParams {
                mode: WriteMode::Create, // Use WriteMode::Append if desired.
                ..Default::default()
            };
            println!("Writing to Lance dataset at {}", lance_path.display());
            Dataset::write(
                batch_iter,
                &lance_path.to_string_lossy(),
                Some(write_params),
            )
            .await
            .map_err(|e| format!("Failed to write Lance dataset: {e}"))?;
            println!(
                "Successfully wrote Lance dataset at {}",
                lance_path.display()
            );
        } else {
            println!("No data to write.");
        }
    } else {
        // Otherwise, simply process all batches (with progress) to count spectra.
        let progress_iter = ProgressRecordBatchIterator::new(iter);
        let total: u64 = progress_iter.map(|b| b.num_rows() as u64).sum();
        println!("Total spectra read: {}", total);
    }

    Ok(())
}
