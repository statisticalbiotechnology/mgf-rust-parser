use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Array, Float64Builder, Int16Array, Int16Builder, Int32Array, Int32Builder,
    ListBuilder, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;

use rayon::prelude::*; // Import Rayon’s parallel iterator traits

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

/// This iterator yields [`RecordBatch`]es from an MGF file.
pub struct MGFRecordBatchIter {
    reader: BufReader<File>,
    batch_size: usize,
    min_peaks: usize,
    buffer: Vec<MgfSpectrum>,
    done: bool,
    schema: Arc<Schema>,
}

/// Create an iterator that parses an MGF file at `mgf_path`,
/// grouping up to `batch_size` spectra per batch,
/// and ignoring spectra with fewer than `min_peaks`.
pub fn parse_mgf<P: AsRef<Path>>(
    mgf_path: P,
    batch_size: usize,
    min_peaks: usize,
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

    Ok(MGFRecordBatchIter {
        reader: BufReader::new(file),
        batch_size,
        min_peaks,
        buffer: Vec::new(),
        done: false,
        schema,
    })
}

impl Iterator for MGFRecordBatchIter {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        self.buffer.clear();

        // Read spectra until we fill a batch or hit EOF.
        while self.buffer.len() < self.batch_size {
            match read_one_spectrum(&mut self.reader, self.min_peaks) {
                Ok(Some(spectrum)) => self.buffer.push(spectrum),
                Ok(None) => {
                    self.done = true;
                    break;
                }
                Err(_) => continue, // skip parse errors
            }
        }
        if self.buffer.is_empty() {
            return None;
        }

        let row_count = self.buffer.len();

        // Build each column in parallel.
        let title_vals: Vec<Option<String>> = self
            .buffer
            .par_iter()
            .map(|spec| spec.title.clone())
            .collect();
        let scan_id_vals: Vec<Option<i32>> =
            self.buffer.par_iter().map(|spec| spec.scan_id).collect();
        let pepmass_vals: Vec<Option<f64>> =
            self.buffer.par_iter().map(|spec| spec.pepmass).collect();
        let rt_vals: Vec<Option<f64>> = self
            .buffer
            .par_iter()
            .map(|spec| spec.rtinseconds)
            .collect();
        let charge_vals: Vec<Option<i16>> =
            self.buffer.par_iter().map(|spec| spec.charge).collect();
        let seq_vals: Vec<Option<String>> = self
            .buffer
            .par_iter()
            .map(|spec| spec.seq.clone())
            .collect();
        let scans_vals: Vec<Option<i32>> = self.buffer.par_iter().map(|spec| spec.scans).collect();
        let all_mz: Vec<Vec<f64>> = self.buffer.par_iter().map(|spec| spec.mz.clone()).collect();
        let all_intens: Vec<Vec<f64>> = self
            .buffer
            .par_iter()
            .map(|spec| spec.intensity.clone())
            .collect();

        // Build Arrow arrays sequentially (builders are not thread-safe).
        let title_arr = StringArray::from(title_vals);
        let scan_id_arr = int32_nullable(&scan_id_vals);
        let pepmass_arr = f64_nullable(&pepmass_vals);
        let rt_arr = f64_nullable(&rt_vals);
        let charge_arr = i16_nullable(&charge_vals);
        let seq_arr = StringArray::from(seq_vals);
        let scans_arr = int32_nullable(&scans_vals);
        let mz_list = build_list_array_f64(all_mz);
        let int_list = build_list_array_f64(all_intens);

        // Construct the RecordBatch.
        let batch = RecordBatch::try_new(
            self.schema.clone(),
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
        )
        .ok()?;

        Some(batch)
    }
}

/// Read lines until we find `BEGIN IONS`, then parse until `END IONS`.
/// Return an `MgfSpectrum` if we get enough peaks. Otherwise None (EOF or invalid).
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
            if let Some(sc) = parse_scan_id(&s) {
                spec.scan_id = Some(sc);
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

/// Attempt to parse an i32 from a line containing "scan=###" or "index=###" etc.
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

/// Build a ListArray<f64> from `Vec<Vec<f64>>`.
fn build_list_array_f64(data: Vec<Vec<f64>>) -> ArrayRef {
    let val_builder = Float64Builder::new();
    let mut list_builder = ListBuilder::new(val_builder);
    for row in data {
        list_builder.values().append_slice(&row);
        list_builder.append(true);
    }
    Arc::new(list_builder.finish())
}

/// Build a nullable Int32Array from a slice of Option<i32>.
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

/// Build a nullable i16 array from a slice of Option<i16>.
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

/// Build a nullable f64 array from a slice of Option<f64>.
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

/// Example main function that prints the first batch.
// use arrow::util::pretty::print_batches;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mgf_path = "tests/test.mgf";
    let batch_size = 5;
    let min_peaks = 3;

    let mut iter = parse_mgf(mgf_path, batch_size, min_peaks)?;

    // Take the first batch, if any
    if let Some(first_batch) = iter.next() {
        println!("First RecordBatch has {} rows.", first_batch.num_rows());

        // Slice down to row 0..1 so the pretty-printer only prints that single row:
        let single_row = first_batch.slice(0, 1);
        println!("First row from the first batch:");
        println!("{:?}", &single_row)
    } else {
        println!("No batches produced from the MGF file.");
    }

    Ok(())
}
