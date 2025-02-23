mod read_mgf;

use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use arrow::record_batch::RecordBatch;
use clap::{Parser, ValueHint};
use indicatif::{ProgressBar, ProgressStyle};

/// A simple CLI parser using Clap
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Parse multiple MGF files (or directories) into Arrow RecordBatches."
)]
struct Cli {
    /// One or more MGF files or directories to parse. If a directory, we search
    /// recursively for `.mgf` files.
    ///
    /// Example usage:
    ///   --file path1.mgf --file /some/dir/with/mgfs
    ///
    #[arg(
        short = 'f',
        long = "file",
        help = "MGF file(s) or directory(ies)",
        value_hint = ValueHint::AnyPath,
        num_args = 1..,
        required = true
    )]
    files: Vec<PathBuf>,

    /// The number of spectra to accumulate before building a RecordBatch
    #[arg(long, default_value = "1000", help = "Batch size")]
    batch_size: usize,

    /// Ignore any MGF spectrum with fewer than this many peaks
    #[arg(long, default_value = "1", help = "Minimum peak count per spectrum")]
    min_peaks: usize,

    /// The size of the bounded channel used in the pipeline
    #[arg(long, default_value = "8", help = "Channel capacity")]
    channel_capacity: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    // 1) Parse CLI arguments with Clap
    let cli = Cli::parse();

    // 2) Create the MGF iterator from our `read_mgf` module.
    //    This function can handle multiple input paths, building a combined list of all mgf files.
    let iter = read_mgf::parse_mgf_files(
        &cli.files, // the user-provided list of paths
        cli.batch_size,
        cli.min_peaks,
        cli.channel_capacity,
    )?;

    // 3) Process all record batches with a progress bar
    let total_spectra = process_batches_with_progress(iter)?;

    println!("Total spectra read: {total_spectra}");
    Ok(())
}

/// Processes record batches with a spinner-based progress bar.
/// Returns the total number of spectra read across all RecordBatches.
fn process_batches_with_progress<I>(mut iter: I) -> Result<u64, Box<dyn Error>>
where
    I: Iterator<Item = RecordBatch>,
{
    // Create and configure a spinner
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ ")
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.enable_steady_tick(Duration::from_millis(100));

    let start = Instant::now();
    let mut total_spectra: u64 = 0;

    while let Some(batch) = iter.next() {
        let rows = batch.num_rows() as u64;
        total_spectra += rows;
        let elapsed = start.elapsed().as_secs_f64();
        let rate = total_spectra as f64 / elapsed;

        // We must 'leak' the string to get a 'static lifetime for pb.set_message()
        let msg = format!("{total_spectra} spectra read ({rate:.2} per second)");
        let static_msg: &'static str = Box::leak(msg.into_boxed_str());
        pb.set_message(static_msg);
    }

    let final_msg = format!(
        "Done: {total_spectra} spectra read in {:.2} seconds.",
        start.elapsed().as_secs_f64()
    );
    let leaked_final: &'static str = Box::leak(final_msg.into_boxed_str());
    pb.finish_with_message(leaked_final);

    Ok(total_spectra)
}
