mod read_mgf;

use std::env;
use std::error::Error;
use std::time::{Duration, Instant};

use arrow::record_batch::RecordBatch;
use indicatif::{ProgressBar, ProgressStyle};

/// Parse CLI arguments: <mgf_file> [batch_size] [min_peaks] [channel_capacity]
fn parse_cli_args() -> Result<(String, usize, usize, usize), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <mgf_file> [batch_size] [min_peaks] [channel_capacity]",
            args[0]
        );
        std::process::exit(1);
    }

    let mgf_path = args[1].clone();
    let batch_size = if args.len() > 2 {
        args[2]
            .parse::<usize>()
            .map_err(|_| format!("Invalid batch_size: {}", args[2]))?
    } else {
        5 // default
    };
    let min_peaks = if args.len() > 3 {
        args[3]
            .parse::<usize>()
            .map_err(|_| format!("Invalid min_peaks: {}", args[3]))?
    } else {
        3 // default
    };
    let channel_capacity = if args.len() > 4 {
        args[4]
            .parse::<usize>()
            .map_err(|_| format!("Invalid channel_capacity: {}", args[4]))?
    } else {
        8 // default capacity
    };

    Ok((mgf_path, batch_size, min_peaks, channel_capacity))
}

/// Processes the MGF iterator with a progress bar.
/// Returns the total number of spectra read.
fn process_batches_with_progress<I>(mut iter: I) -> Result<u64, Box<dyn Error>>
where
    I: Iterator<Item = RecordBatch>,
{
    // Create and configure a spinner-style progress bar.
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ ")
            .template("{spinner:.green} {msg}")
            .expect("Failed to set progress bar template"),
    );
    pb.enable_steady_tick(Duration::from_millis(100));

    let start = Instant::now();
    let mut total_spectra: u64 = 0;

    // Iterate over batches, updating the progress bar.
    while let Some(batch) = iter.next() {
        let rows = batch.num_rows() as u64;
        total_spectra += rows;
        let elapsed = start.elapsed().as_secs_f64();
        let rate = total_spectra as f64 / elapsed;
        let msg = format!("{} spectra read ({:.2} per second)", total_spectra, rate);
        // Leak the formatted string so it has a 'static lifetime.
        let static_msg: &'static str = Box::leak(msg.into_boxed_str());
        pb.set_message(static_msg);
    }

    let finish_msg = format!(
        "Done: {} spectra read in {:.2} seconds.",
        total_spectra,
        start.elapsed().as_secs_f64()
    );
    let leaked: &'static mut str = Box::leak(finish_msg.into_boxed_str());
    pb.finish_with_message(&*leaked);

    Ok(total_spectra)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command-line arguments.
    let (mgf_path, batch_size, min_peaks, channel_capacity) = parse_cli_args()?;

    // Create the MGF iterator from our module, now with `channel_capacity`.
    let iter = read_mgf::parse_mgf(&mgf_path, batch_size, min_peaks, channel_capacity)?;

    // Process batches with a progress bar
    let total = process_batches_with_progress(iter)?;

    Ok(())
}
