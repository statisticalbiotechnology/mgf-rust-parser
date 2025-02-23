import lance
import pandas as pd


def main():
    # Open the Lance dataset
    dataset = lance.dataset("data/output.lance")

    # Read the first 5 rows as an Arrow Table
    table = dataset.to_table(limit=5)

    # Convert to Pandas DataFrame with Arrow types preserved
    df = table.to_pandas(types_mapper=pd.ArrowDtype)

    # Print DataFrame with array lengths

    print(df.head(5))

    # Ensure that mz_array and intensity_array are treated as lists
    df["mz_array"] = df["mz_array"].apply(
        lambda x: list(x) if hasattr(x, "__iter__") else None
    )
    df["intensity_array"] = df["intensity_array"].apply(
        lambda x: list(x) if hasattr(x, "__iter__") else None
    )

    # Compute array lengths
    df["mz_array_length"] = df["mz_array"].apply(
        lambda x: len(x) if isinstance(x, list) else None
    )
    df["intensity_array_length"] = df["intensity_array"].apply(
        lambda x: len(x) if isinstance(x, list) else None
    )

    print("\n=== Array Lengths ===")
    print(df[["mz_array_length", "intensity_array_length"]])


if __name__ == "__main__":
    main()
