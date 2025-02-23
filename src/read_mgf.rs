use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

use arrow::array::{
    ArrayRef, Float64Array, Float64Builder, Int16Array, Int16Builder, Int32Array, Int32Builder,
    ListBuilder, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;

use crossbeam::channel::{Receiver, Sender, bounded};
use rayon::prelude::*;

/// A minimal struct for storing one MGF spectrum plus metadata.
#[derive(Debug)]
struct MgfSpectrum {
    // Metadata
    title: Option<String>,
    scan_id: Option<i32>,
    pepmass: Option<f64>,
    rtinseconds: Option<f64>,
    charge: Option<i16>,
    seq: Option<String>,
    scans: Option<i32>,

    // Peak arrays
    mz: Vec<f64>,
    intensity: Vec<f64>,
}

/// An iterator that yields `RecordBatch` objects from our pipeline.
pub struct MGFRecordBatchIter {
    rx_recordbatch: Receiver<Option<RecordBatch>>,
    done: bool,
}

/// Return `Some(RecordBatch)` until the channel sends `None`.
impl Iterator for MGFRecordBatchIter {
    type Item = RecordBatch;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.rx_recordbatch.recv().ok()? {
            Some(batch) => Some(batch),
            None => {
                // no more data
                self.done = true;
                None
            }
        }
    }
}

/// Create a pipeline to parse an MGF file in one thread (producer)
/// and build Arrow `RecordBatch` in another thread (consumer),
/// returning an iterator that yields the final `RecordBatch`s.
///
/// - `mgf_path`: the input file
/// - `batch_size`: how many spectra to buffer in one `Vec<MgfSpectrum>` before sending
/// - `min_peaks`: ignore a spectrum if it has fewer than `min_peaks` peaks
/// - `channel_capacity`: the bounded channel size (default 8). This sets how many
///   in-flight batches we allow before the reading thread will block.
pub fn parse_mgf<P: AsRef<Path>>(
    mgf_path: P,
    batch_size: usize,
    min_peaks: usize,
    channel_capacity: usize, // newly added argument, default 8 if user doesn't override
) -> Result<MGFRecordBatchIter, ArrowError> {
    // Build a schema with columns for the metadata and peak arrays
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::Utf8, true),
        Field::new("scan_id", DataType::Int32, true),
        Field::new("pepmass", DataType::Float64, true),
        Field::new("rtinseconds", DataType::Float64, true),
        Field::new("charge", DataType::Int16, true),
        Field::new("seq", DataType::Utf8, true),
        Field::new("scans", DataType::Int32, true),
        Field::new(
            "mz_array",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
        Field::new(
            "intensity_array",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
    ]));

    // Attempt to open the file
    let file = File::open(mgf_path).map_err(|e| ArrowError::ExternalError(Box::new(e)))?;

    // We'll create two channels:
    //  1) "batch_of_spectra" is from the reading thread to the building thread
    //     Bounded by `channel_capacity`
    //  2) "record_batch" is from the building thread to the final iterator,
    //     also with the same or a separate channel capacity (here we set the same).
    let (tx_spectra, rx_spectra) = bounded::<Vec<MgfSpectrum>>(channel_capacity);
    let (tx_recordbatch, rx_recordbatch) = bounded::<Option<RecordBatch>>(channel_capacity);

    // We wrap the BufReader in a Mutex so the reading thread can have exclusive
    // access to it. (We only read in the reading thread, so it's mostly for code clarity.)
    let reader = BufReader::new(file);
    let reader = Arc::new(Mutex::new(reader));

    // Spawn the reading thread
    {
        let reader_clone = Arc::clone(&reader);
        let tx_spectra_clone = tx_spectra.clone();

        thread::spawn(move || {
            // This function reads MGF lines into Vec<MgfSpectrum> and sends them
            if let Err(err) =
                read_mgf_in_thread(reader_clone, batch_size, min_peaks, tx_spectra_clone)
            {
                eprintln!("Error reading mgf: {:?}", err);
            }
        });
    }

    // Spawn the building thread
    {
        let schema_clone = Arc::clone(&schema);
        thread::spawn(move || {
            build_batches_in_thread(rx_spectra, tx_recordbatch, schema_clone);
        });
    }

    // The final MGFRecordBatchIter just yields from "rx_recordbatch"
    Ok(MGFRecordBatchIter {
        rx_recordbatch,
        done: false,
    })
}

// ---------------- READING THREAD ----------------

/// Continuously read the file, building Vec<MgfSpectrum> until `batch_size` or EOF,
/// then send them to `tx_spectra`. If we have leftover spectra smaller than batch_size
/// at EOF, send them as well.
fn read_mgf_in_thread(
    reader: Arc<Mutex<BufReader<File>>>,
    batch_size: usize,
    min_peaks: usize,
    tx_spectra: Sender<Vec<MgfSpectrum>>,
) -> Result<(), ArrowError> {
    let mut buffer = Vec::with_capacity(batch_size);

    loop {
        let spectrum = {
            // Lock the reader for each read operation
            let mut guard = reader.lock().unwrap();
            read_one_spectrum(&mut *guard, min_peaks)?
        };
        match spectrum {
            Some(sp) => {
                buffer.push(sp);
                if buffer.len() == batch_size {
                    tx_spectra.send(buffer).unwrap();
                    buffer = Vec::with_capacity(batch_size);
                }
            }
            None => {
                // EOF or invalid
                if !buffer.is_empty() {
                    tx_spectra.send(buffer).unwrap();
                }
                break;
            }
        }
    }
    // When done, close the channel by dropping tx_spectra
    drop(tx_spectra);

    Ok(())
}

// ---------------- BUILDING THREAD ----------------

/// Receive Vec<MgfSpectrum> from the reading thread,
/// build each as a `RecordBatch`, and send them out.
fn build_batches_in_thread(
    rx_spectra: Receiver<Vec<MgfSpectrum>>,
    tx_recordbatch: Sender<Option<RecordBatch>>,
    schema: Arc<Schema>,
) {
    while let Ok(spectra_batch) = rx_spectra.recv() {
        // We received a batch of MgfSpectrum from the reading thread
        if spectra_batch.is_empty() {
            continue;
        }

        // For strings, we can do from_iter referencing the existing string data
        let opt_title_refs: Vec<Option<&str>> = spectra_batch
            .par_iter()
            .map(|spec| spec.title.as_deref())
            .collect();
        let title_arr = StringArray::from_iter(opt_title_refs);

        let scan_id_vals: Vec<Option<i32>> =
            spectra_batch.par_iter().map(|spec| spec.scan_id).collect();

        let pepmass_vals: Vec<Option<f64>> =
            spectra_batch.par_iter().map(|spec| spec.pepmass).collect();

        let rt_vals: Vec<Option<f64>> = spectra_batch
            .par_iter()
            .map(|spec| spec.rtinseconds)
            .collect();

        let charge_vals: Vec<Option<i16>> =
            spectra_batch.par_iter().map(|spec| spec.charge).collect();

        let opt_seq_refs: Vec<Option<&str>> = spectra_batch
            .par_iter()
            .map(|spec| spec.seq.as_deref())
            .collect();
        let seq_arr = StringArray::from_iter(opt_seq_refs);

        let scans_vals: Vec<Option<i32>> =
            spectra_batch.par_iter().map(|spec| spec.scans).collect();

        // For the "mz" and "intensity" columns, we do clone the Vec<f64>
        let all_mz: Vec<Vec<f64>> = spectra_batch
            .par_iter()
            .map(|spec| spec.mz.clone())
            .collect();

        let all_intens: Vec<Vec<f64>> = spectra_batch
            .par_iter()
            .map(|spec| spec.intensity.clone())
            .collect();

        // Build final arrays
        let scan_id_arr = int32_nullable(&scan_id_vals);
        let pepmass_arr = f64_nullable(&pepmass_vals);
        let rt_arr = f64_nullable(&rt_vals);
        let charge_arr = i16_nullable(&charge_vals);
        let scans_arr = int32_nullable(&scans_vals);
        let mz_list = build_list_array_f64(all_mz);
        let int_list = build_list_array_f64(all_intens);

        // Construct the RecordBatch
        let batch = match RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(title_arr),
                Arc::new(scan_id_arr),
                Arc::new(pepmass_arr),
                Arc::new(rt_arr),
                Arc::new(charge_arr),
                Arc::new(seq_arr),
                Arc::new(scans_arr),
                mz_list,
                int_list,
            ],
        ) {
            Ok(b) => b,
            Err(_) => {
                // if building fails, we can skip or send None
                continue;
            }
        };

        // Send the record batch along
        tx_recordbatch.send(Some(batch)).unwrap();
    }

    // no more input => send a final None to indicate end of stream
    let _ = tx_recordbatch.send(None);
}

// ---------------- The read_one_spectrum function from before ----------------

fn read_one_spectrum(
    reader: &mut BufReader<File>,
    min_peaks: usize,
) -> Result<Option<MgfSpectrum>, ArrowError> {
    // search for "BEGIN IONS"
    loop {
        let mut line = String::new();
        let bytes_read = reader
            .read_line(&mut line)
            .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
        if bytes_read == 0 {
            // EOF
            return Ok(None);
        }
        let trimmed = line.trim();
        if trimmed.starts_with("BEGIN IONS") {
            break;
        }
    }

    let mut spec = MgfSpectrum {
        title: None,
        scan_id: None,
        pepmass: None,
        rtinseconds: None,
        charge: None,
        seq: None,
        scans: None,
        mz: Vec::new(),
        intensity: Vec::new(),
    };

    // Now parse lines until "END IONS"
    loop {
        let mut line = String::new();
        let bytes_read = reader
            .read_line(&mut line)
            .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
        if bytes_read == 0 {
            // EOF in the middle of a block
            break;
        }
        let trimmed = line.trim();
        if trimmed.starts_with("END IONS") {
            break;
        }

        // Check for e.g. TITLE=, PEPMASS=, ...
        if let Some(val) = line.strip_prefix("TITLE=") {
            let s = val.trim().to_string();
            spec.title = Some(s.clone());

            // Attempt to parse out scan=### from the Title
            if let Some(scan_id) = parse_scan_id(&s) {
                spec.scan_id = Some(scan_id);
            }
            continue;
        }
        if let Some(val) = line.strip_prefix("PEPMASS=") {
            let v = val.trim().split_whitespace().next().unwrap_or("");
            if let Ok(pepmass) = v.parse::<f64>() {
                spec.pepmass = Some(pepmass);
            }
            continue;
        }
        if let Some(val) = line.strip_prefix("RTINSECONDS=") {
            if let Ok(rt) = val.trim().parse::<f64>() {
                spec.rtinseconds = Some(rt);
            }
            continue;
        }
        if let Some(val) = line.strip_prefix("CHARGE=") {
            let mut s = val.trim().to_string();
            if s.ends_with('+') {
                s.pop();
            }
            if let Ok(ch) = s.parse::<i16>() {
                spec.charge = Some(ch);
            }
            continue;
        }
        if let Some(val) = line.strip_prefix("SCANS=") {
            if let Ok(sc) = val.trim().parse::<i32>() {
                spec.scans = Some(sc);
            }
            continue;
        }
        if let Some(val) = line.strip_prefix("SEQ=") {
            spec.seq = Some(val.trim().to_string());
            continue;
        }

        // Otherwise assume "mz intensity"
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() >= 2 {
            if let (Ok(mz_val), Ok(int_val)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                spec.mz.push(mz_val);
                spec.intensity.push(int_val);
            }
        }
    }

    if spec.mz.len() < min_peaks {
        Ok(None)
    } else {
        Ok(Some(spec))
    }
}

/// Attempt to parse an i32 from a string containing "scan=###" or "index=###" etc.
fn parse_scan_id(s: &str) -> Option<i32> {
    // If the entire string is just a number, use that
    if let Ok(num) = s.parse::<i32>() {
        return Some(num);
    }
    // Otherwise look for "scan="
    if let Some(idx) = s.find("scan=") {
        let substr = &s[idx + 5..];
        if let Ok(num) = substr.parse::<i32>() {
            return Some(num);
        }
    }
    // Or "index="
    if let Some(idx) = s.find("index=") {
        let substr = &s[idx + 6..];
        if let Ok(num) = substr.parse::<i32>() {
            return Some(num);
        }
    }
    None
}

// -------------- Utility: build arrow arrays --------------

fn build_list_array_f64(data: Vec<Vec<f64>>) -> ArrayRef {
    let val_builder = Float64Builder::new();
    let mut list_builder = ListBuilder::new(val_builder);
    for row in data {
        list_builder.values().append_slice(&row);
        list_builder.append(true);
    }
    Arc::new(list_builder.finish())
}

fn int32_nullable(vals: &[Option<i32>]) -> Int32Array {
    let mut builder = Int32Builder::new();
    for v in vals {
        match v {
            Some(x) => builder.append_value(*x),
            None => builder.append_null(),
        };
    }
    builder.finish()
}

fn i16_nullable(vals: &[Option<i16>]) -> Int16Array {
    let mut builder = Int16Builder::new();
    for v in vals {
        match v {
            Some(x) => builder.append_value(*x),
            None => builder.append_null(),
        };
    }
    builder.finish()
}

fn f64_nullable(vals: &[Option<f64>]) -> Float64Array {
    let mut builder = Float64Builder::new();
    for v in vals {
        match v {
            Some(x) => builder.append_value(*x),
            None => builder.append_null(),
        };
    }
    builder.finish()
}

// -------------- Example main for debugging --------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mgf_path = "tests/test.mgf";
    let batch_size = 5;
    let min_peaks = 3;

    // Default channel capacity = 8
    let channel_capacity = 8;

    // This sets up the parallel pipeline with bounding
    let mut iter = parse_mgf(mgf_path, batch_size, min_peaks, channel_capacity)?;

    // Take the first batch, if any
    if let Some(first_batch) = iter.next() {
        println!("First RecordBatch has {} rows.", first_batch.num_rows());

        // Slice down to row 0..1 so the pretty-printer only prints that single row:
        let single_row = first_batch.slice(0, 1);
        println!("First row from the first batch:");
        println!("{:?}", &single_row);
    } else {
        println!("No batches produced from the MGF file.");
    }

    Ok(())
}
