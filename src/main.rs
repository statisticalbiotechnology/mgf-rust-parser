use arrow::error::ArrowError;
use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, Int32Array, RecordBatch, RecordBatchReader};
use arrow_schema::{DataType, Field, Schema};
use lancedb::connect;
use std::sync::Arc;
use tokio;

/// A custom record batch reader that wraps an iterator yielding
/// `Result<RecordBatch, ArrowError>`.
struct MyRecordBatchReader<I> {
    iter: I,
    schema: Arc<Schema>,
}

impl<I> MyRecordBatchReader<I> {
    fn new(iter: I, schema: Arc<Schema>) -> Self {
        Self { iter, schema }
    }
}

impl<I> Iterator for MyRecordBatchReader<I>
where
    I: Iterator<Item = Result<RecordBatch, ArrowError>>,
{
    type Item = Result<RecordBatch, ArrowError>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

/// Implement the public trait `arrow_array::RecordBatchReader` for our custom reader.
impl<I> arrow_array::RecordBatchReader for MyRecordBatchReader<I>
where
    I: Iterator<Item = Result<RecordBatch, ArrowError>>,
{
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to a local LanceDB database (the directory will be created if needed).
    let db = connect("data/test.lance").execute().await?;

    // Define a simple schema with two columns: an integer "id" and a "vector" column
    // which is a fixed-size list of 128 Float32 values.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            // FixedSizeList expects its inner field as an Arc<Field>.
            // We create a Box<Field> and convert it using `.into()`.
            DataType::FixedSizeList(
                Box::new(Field::new("item", DataType::Float32, true)).into(),
                128,
            ),
            true,
        ),
    ]));

    // Create an Int32 array for the "id" column with 3 rows.
    let id_array = Arc::new(Int32Array::from(vec![1, 2, 3]));

    // Create a FixedSizeList array for the "vector" column.
    // Each row is a vector of 128 ones.
    let vector_array = Arc::new(
        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            (0..3).map(|_| Some(vec![Some(1.0_f32); 128])),
            128,
        ),
    );

    // Build a RecordBatch with the arrays and schema.
    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, vector_array])?;

    // Create an iterator that yields our single RecordBatch.
    let batches = std::iter::once(Ok(batch));

    // Wrap the iterator in our custom RecordBatchReader.
    let my_reader = MyRecordBatchReader::new(batches, schema.clone());

    // Create a table named "my_table" in the database with our test data.
    db.create_table("my_table", Box::new(my_reader))
        .execute()
        .await?;

    println!("Table 'my_table' created successfully with 3 rows.");
    Ok(())
}
