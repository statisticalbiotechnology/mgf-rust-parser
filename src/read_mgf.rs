use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
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
use serde::Deserialize;
use serde_yaml;
use std::sync::atomic::{AtomicUsize, Ordering};

// Global counters for skipped spectra.
pub static SKIPPED_MIN: AtomicUsize = AtomicUsize::new(0);
pub static SKIPPED_ERR: AtomicUsize = AtomicUsize::new(0);

/// Configuration for parsing MGF fields loaded from a YAML file.
#[derive(Debug, Deserialize)]
pub struct MGFConfig {
    pub title_prefix: String,
    pub pepmass_prefix: String,
    pub rtinseconds_prefix: String,
    pub charge_prefix: String,
    pub scans_prefix: String,
    pub seq_prefix: String,
}

impl Default for MGFConfig {
    fn default() -> Self {
        Self {
            title_prefix: "TITLE=".to_string(),
            pepmass_prefix: "PEPMASS=".to_string(),
            rtinseconds_prefix: "RTINSECONDS=".to_string(),
            charge_prefix: "CHARGE=".to_string(),
            scans_prefix: "SCANS=".to_string(),
            seq_prefix: "SEQ=".to_string(),
        }
    }
}

/// This is the pipeline iterator we yield.
pub struct MGFRecordBatchIter {
    rx_recordbatch: Receiver<Option<RecordBatch>>,
    done: bool,
}

impl Iterator for MGFRecordBatchIter {
    type Item = RecordBatch;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.rx_recordbatch.recv().ok()? {
            Some(batch) => Some(batch),
            None => {
                self.done = true;
                None
            }
        }
    }
}

/// A minimal struct for storing one MGF spectrum + metadata.
#[derive(Debug)]
struct MgfSpectrum {
    filename: String, // which file it came from
    title: Option<String>,
    scan_id: Option<i32>,
    pepmass: Option<f64>,
    rtinseconds: Option<f64>,
    charge: Option<i16>,
    seq: Option<String>,
    scans: Option<i32>,

    // Peak data
    mz: Vec<f64>,
    intensity: Vec<f64>,
}

/// If the user provides multiple paths (files or directories),
/// this function:
/// 1) Gathers all `.mgf` files from them,
/// 2) Spawns the reading and building threads,
/// 3) Returns an iterator for the final RecordBatches.
///
/// Note: The configuration for field prefixes is passed in via the `config` argument.
pub fn parse_mgf_files(
    paths: &[PathBuf],
    batch_size: usize,
    min_peaks: usize,
    channel_capacity: usize,
    fields_config: Arc<MGFConfig>,
) -> Result<MGFRecordBatchIter, ArrowError> {
    // Gather all mgf files.
    let mut mgf_files = Vec::new();
    for p in paths {
        let found = gather_mgf_files(p)?;
        mgf_files.extend(found);
    }
    mgf_files.sort();

    if mgf_files.is_empty() {
        let (_tx, rx) = bounded::<Option<RecordBatch>>(1);
        return Ok(MGFRecordBatchIter {
            rx_recordbatch: rx,
            done: false,
        });
    }

    // Build schema (adding a "filename" column).
    let schema = Arc::new(Schema::new(vec![
        Field::new("filename", DataType::Utf8, false),
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

    // Create channels.
    let (tx_spectra, rx_spectra) = bounded::<Vec<MgfSpectrum>>(channel_capacity);
    let (tx_rb, rx_rb) = bounded::<Option<RecordBatch>>(channel_capacity);

    {
        let files_clone = mgf_files.clone();
        let config_clone = Arc::clone(&fields_config);
        thread::spawn(move || {
            if let Err(e) = read_all_mgfs_in_thread(
                files_clone,
                batch_size,
                min_peaks,
                tx_spectra,
                config_clone,
            ) {
                eprintln!("Error in reading thread: {e:?}");
            }
        });
    }
    {
        let schema_clone = Arc::clone(&schema);
        thread::spawn(move || {
            build_batches_in_thread(rx_spectra, tx_rb, schema_clone);
        });
    }

    Ok(MGFRecordBatchIter {
        rx_recordbatch: rx_rb,
        done: false,
    })
}

/// Recursively gather .mgf files from a file or directory.
fn gather_mgf_files(path: &Path) -> Result<Vec<PathBuf>, ArrowError> {
    let mut results = Vec::new();
    if path.is_dir() {
        for entry in fs::read_dir(path).map_err(|e| ArrowError::ExternalError(Box::new(e)))? {
            let entry = entry.map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
            let p = entry.path();
            if p.is_dir() {
                let sub = gather_mgf_files(&p)?;
                results.extend(sub);
            } else if let Some(ext) = p.extension() {
                if ext.eq_ignore_ascii_case("mgf") {
                    results.push(p);
                }
            }
        }
    } else if path.is_file() {
        if let Some(ext) = path.extension() {
            if ext.eq_ignore_ascii_case("mgf") {
                results.push(path.to_path_buf());
            }
        }
    }
    Ok(results)
}

/// Read all mgf files, parse spectra, batch them, and send them to the building thread.
fn read_all_mgfs_in_thread(
    mgf_files: Vec<PathBuf>,
    batch_size: usize,
    min_peaks: usize,
    tx_spectra: Sender<Vec<MgfSpectrum>>,
    config: Arc<MGFConfig>,
) -> Result<(), ArrowError> {
    let mut buffer = Vec::with_capacity(batch_size);

    for f in mgf_files {
        let file = File::open(&f).map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
        let mut reader = BufReader::new(file);
        let mut line_num = 0;

        loop {
            match read_one_spectrum(&mut reader, min_peaks, &f, &config, &mut line_num) {
                Ok(Some(s)) => {
                    if s.mz.len() < min_peaks {
                        SKIPPED_MIN.fetch_add(1, Ordering::SeqCst);
                        continue;
                    }
                    buffer.push(s);
                    if buffer.len() == batch_size {
                        tx_spectra.send(buffer).unwrap();
                        buffer = Vec::with_capacity(batch_size);
                    }
                }
                Ok(None) => break, // EOF for this file.
                Err(e) => {
                    SKIPPED_ERR.fetch_add(1, Ordering::SeqCst);
                    eprintln!(
                        "Error parsing spectrum in file {} at line {}: {e}",
                        f.display(),
                        line_num
                    );
                    continue;
                }
            }
        }
    }
    if !buffer.is_empty() {
        tx_spectra.send(buffer).unwrap();
    }
    drop(tx_spectra);
    Ok(())
}

/// Build Arrow RecordBatches from each Vec<MgfSpectrum>.
fn build_batches_in_thread(
    rx: Receiver<Vec<MgfSpectrum>>,
    tx: Sender<Option<RecordBatch>>,
    schema: Arc<Schema>,
) {
    while let Ok(batch_of_spectra) = rx.recv() {
        if batch_of_spectra.is_empty() {
            continue;
        }

        let filename_vals: Vec<&str> = batch_of_spectra
            .iter()
            .map(|sp| sp.filename.as_str())
            .collect();
        let title_arr =
            StringArray::from_iter(batch_of_spectra.iter().map(|sp| sp.title.as_deref()));
        let scan_id_vals: Vec<Option<i32>> = batch_of_spectra.iter().map(|sp| sp.scan_id).collect();
        let pepmass_vals: Vec<Option<f64>> = batch_of_spectra.iter().map(|sp| sp.pepmass).collect();
        let rt_vals: Vec<Option<f64>> = batch_of_spectra.iter().map(|sp| sp.rtinseconds).collect();
        let charge_vals: Vec<Option<i16>> = batch_of_spectra.iter().map(|sp| sp.charge).collect();
        let seq_arr = StringArray::from_iter(batch_of_spectra.iter().map(|sp| sp.seq.as_deref()));
        let scans_vals: Vec<Option<i32>> = batch_of_spectra.iter().map(|sp| sp.scans).collect();

        let all_mz: Vec<Vec<f64>> = batch_of_spectra.iter().map(|sp| sp.mz.clone()).collect();
        let all_intens: Vec<Vec<f64>> = batch_of_spectra
            .iter()
            .map(|sp| sp.intensity.clone())
            .collect();

        let filename_arr = StringArray::from(filename_vals);
        let scan_id_arr = int32_nullable(&scan_id_vals);
        let pepmass_arr = f64_nullable(&pepmass_vals);
        let rt_arr = f64_nullable(&rt_vals);
        let charge_arr = i16_nullable(&charge_vals);
        let scans_arr = int32_nullable(&scans_vals);
        let mz_list = build_list_array_f64(all_mz);
        let int_list = build_list_array_f64(all_intens);

        let rb = match RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(filename_arr),
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
                tx.send(None).ok();
                return;
            }
        };

        tx.send(Some(rb)).ok();
    }
    let _ = tx.send(None);
}

/// Read exactly one spectrum from the given reader (for one file).
/// Returns Ok(Some(MgfSpectrum)) if a spectrum is parsed,
/// Ok(None) if EOF is reached.
fn read_one_spectrum(
    reader: &mut BufReader<File>,
    _min_peaks: usize, // Already checked in the reading thread.
    file_path: &Path,
    config: &MGFConfig,
    line_num: &mut usize,
) -> Result<Option<MgfSpectrum>, ArrowError> {
    // Look for "BEGIN IONS"
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
        *line_num += 1;
        if n == 0 {
            return Ok(None); // EOF
        }
        if line.trim_start().starts_with("BEGIN IONS") {
            break;
        }
    }

    let block_start = *line_num;
    let mut sp = MgfSpectrum {
        filename: file_path.display().to_string(),
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

    // Parse lines until "END IONS"
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
        *line_num += 1;
        if n == 0 {
            break; // EOF in the middle of a block.
        }
        let trimmed = line.trim();
        if trimmed.starts_with("END IONS") {
            break;
        }
        if let Some(val) = line.trim_start().strip_prefix(&config.title_prefix) {
            let s = val.trim();
            sp.title = Some(s.to_string());
            if let Some(num) = parse_scan_id(s) {
                sp.scan_id = Some(num);
            }
        } else if let Some(val) = line.trim_start().strip_prefix(&config.pepmass_prefix) {
            let v = val.trim().split_whitespace().next().unwrap_or("");
            if let Ok(p) = v.parse::<f64>() {
                sp.pepmass = Some(p);
            }
        } else if let Some(val) = line.trim_start().strip_prefix(&config.rtinseconds_prefix) {
            if let Ok(r) = val.trim().parse::<f64>() {
                sp.rtinseconds = Some(r);
            }
        } else if let Some(val) = line.trim_start().strip_prefix(&config.charge_prefix) {
            let mut s = val.trim().to_string();
            if s.ends_with('+') {
                s.pop();
            }
            if let Ok(ch) = s.parse::<i16>() {
                sp.charge = Some(ch);
            }
        } else if let Some(val) = line.trim_start().strip_prefix(&config.scans_prefix) {
            if let Ok(sc) = val.trim().parse::<i32>() {
                sp.scans = Some(sc);
            }
        } else if let Some(val) = line.trim_start().strip_prefix(&config.seq_prefix) {
            sp.seq = Some(val.trim().to_string());
        } else {
            // Otherwise, assume "mz intensity"
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                if let (Ok(mz_val), Ok(int_val)) =
                    (parts[0].parse::<f64>(), parts[1].parse::<f64>())
                {
                    sp.mz.push(mz_val);
                    sp.intensity.push(int_val);
                }
            }
        }
    }

    // Check for missing required fields.
    let mut missing_fields = Vec::new();
    if sp.title.is_none() {
        missing_fields.push(&config.title_prefix);
    }
    if sp.pepmass.is_none() {
        missing_fields.push(&config.pepmass_prefix);
    }
    if sp.rtinseconds.is_none() {
        missing_fields.push(&config.rtinseconds_prefix);
    }
    if sp.charge.is_none() {
        missing_fields.push(&config.charge_prefix);
    }
    if sp.scans.is_none() {
        missing_fields.push(&config.scans_prefix);
    }
    if sp.seq.is_none() {
        missing_fields.push(&config.seq_prefix);
    }
    if !missing_fields.is_empty() {
        println!(
            "Warning: In file {} near line {}: missing fields: {:?}",
            file_path.display(),
            block_start,
            missing_fields
        );
    }
    Ok(Some(sp))
}

/// Parse a scan ID from a string containing "scan=###" or "index=###".
fn parse_scan_id(s: &str) -> Option<i32> {
    if let Ok(num) = s.parse::<i32>() {
        return Some(num);
    }
    if let Some(idx) = s.find("scan=") {
        let substr = &s[idx + 5..];
        if let Ok(num) = substr.parse::<i32>() {
            return Some(num);
        }
    }
    if let Some(idx) = s.find("index=") {
        let substr = &s[idx + 6..];
        if let Ok(num) = substr.parse::<i32>() {
            return Some(num);
        }
    }
    None
}

/// Utility functions to build Arrow arrays.
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
    let mut b = Int32Builder::new();
    for v in vals {
        match v {
            Some(x) => b.append_value(*x),
            None => b.append_null(),
        };
    }
    b.finish()
}

fn i16_nullable(vals: &[Option<i16>]) -> Int16Array {
    let mut b = Int16Builder::new();
    for v in vals {
        match v {
            Some(x) => b.append_value(*x),
            None => b.append_null(),
        };
    }
    b.finish()
}

fn f64_nullable(vals: &[Option<f64>]) -> Float64Array {
    let mut b = Float64Builder::new();
    for v in vals {
        match v {
            Some(x) => b.append_value(*x),
            None => b.append_null(),
        };
    }
    b.finish()
}

//-------------- Example main for debugging --------------//

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path0 = args.get(1).cloned().unwrap_or_else(|| ".".to_string());
    let path1 = PathBuf::from(path0);

    let batch_size = 5;
    let min_peaks = 3;
    let channel_capacity = 8;

    let config = MGFConfig {
        title_prefix: "TITLE=".to_string(),
        pepmass_prefix: "PEPMASS=".to_string(),
        rtinseconds_prefix: "RTINSECONDS=".to_string(),
        charge_prefix: "CHARGE=".to_string(),
        scans_prefix: "SCANS=".to_string(),
        seq_prefix: "SEQ=".to_string(),
    };

    let config = Arc::new(config);

    let mut it = parse_mgf_files(&[path1], batch_size, min_peaks, channel_capacity, config)?;
    if let Some(rb) = it.next() {
        println!("Got first RecordBatch with {} rows.", rb.num_rows());
        if rb.num_rows() > 0 {
            println!("First row:\n{:?}", rb.slice(0, 1));
        }
    } else {
        println!("No data found.");
    }
    println!(
        "Skipped spectra: {} due to min_peaks, {} due to errors",
        SKIPPED_MIN.load(Ordering::SeqCst),
        SKIPPED_ERR.load(Ordering::SeqCst)
    );
    Ok(())
}
