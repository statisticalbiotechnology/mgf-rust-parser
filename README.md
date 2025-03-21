# mgf‑rust‑parser

mgf‑rust‑parser is a Rust‑based tool that parses mass spectrometry MGF files into Apache Arrow RecordBatches and (optionally) writes the resulting data to a Lance dataset.

## Installation and Build

Clone the repository and build in release mode:

  `git clone git@github.com:statisticalbiotechnology/mgf-rust-parser.git`

  `cd mgf-rust-parser`

  `cargo build --release`

## Features

- **CLI Tool:**  
  The command‑line interface (compiled as `mgf2lance`) lets you parse one or more MGF files (or directories) and write the data into a Lance dataset.  

  **Example usage:**  

  ```bash
  mgf2lance  --file /path/to/mgfs --output-lance output.lance 
  --batch-size 1000 --min-peaks 1

- The `--write-mode` option accepts one of the following values:

  - `create` — Create a new dataset (fails if the dataset exists).
  - `append` — Append to an existing dataset.
  - `overwrite` — Overwrite the dataset as a new version (old versions are preserved for time‑travel).

- `--fields-config` lets you customize the field prefixes for parsing.
  For example, your YAML file may look like this:

  ```bash
  title_prefix: "TITLE="
  pepmass_prefix: "PEPMASS="
  rtinseconds_prefix: "RTINSECONDS="
  charge_prefix: "CHARGE="
  scans_prefix: "SCANS="
  seq_prefix: null

- **MGF Parsing Module:**  
  The MGF parsing module `read_mgf::parse_mgf_files` returns an iterator yielding Apache Arrow RecordBatches. This module can be used directly in other Rust projects. The RecordBatches include the fields: pepmass, rtinseconds, charge, seq (sequence), mz_array,intensity_array among others.

- **Read resulting dataset with Python**
  The resulting Lance dataset can be easily loaded in Python using the Lance API (tested with pylance 0.10.10 on pip).

  `dataset = lance.dataset("output.lance")`
