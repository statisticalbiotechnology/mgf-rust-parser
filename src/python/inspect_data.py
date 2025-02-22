import lance
import pyarrow as pa
import pandas as pd


def main():
    # Open an existing Lance dataset.
    # Replace "diffusiondb_train.lance" with the path to your Lance dataset.
    _dataset = lance.dataset(
        "/Users/alfred/Documents/Code/Rust/mgf-rust-parser/data/test.lance/my_table.lance"
    )

    # Read the entire dataset as an Arrow Table.
    table = _dataset.to_table()

    # Option 1: Pretty-print using Arrow's own formatting.
    print("=== Arrow Table ===")
    print(table)

    # Option 2: Convert to a Pandas DataFrame for prettier output.
    df = table.to_pandas()
    print("\n=== Pandas DataFrame ===")
    print(df)


if __name__ == "__main__":
    main()
