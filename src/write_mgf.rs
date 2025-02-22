use arrow::error::ArrowError;
use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchReader};
use arrow_schema::{DataType, Field, Schema};
use lancedb::connect;
use std::error::Error;
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

/// Implement the public trait `RecordBatchReader` (re-exported by arrow_array)
/// for our custom reader.
impl<I> RecordBatchReader for MyRecordBatchReader<I>
where
    I: Iterator<Item = Result<RecordBatch, ArrowError>>,
{
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Define a simple schema with two columns: "id" and "vector".
    // "id" is an Int32 field, and "vector" is a FixedSizeList of 128 Float32 values.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                // The inner field must be provided as an Arc<Field>.
                Box::new(Field::new("item", DataType::Float32, true)).into(),
                128,
            ),
            true,
        ),
    ]));

    // Create dummy data for 3 rows.
    let id_array = Arc::new(Int32Array::from(vec![1, 2, 3]));
    let vector_array = Arc::new(
        // Each row is a vector of 128 ones.
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

    // Connect to a local LanceDB dataset.
    let db = connect("data/test.lance").execute().await?;

    // Write the dummy batch into a table named "dummy_table".
    db.create_table("dummy_table", Box::new(my_reader))
        .execute()
        .await?;

    println!("Dummy batch has been written to table 'dummy_table'.");
    Ok(())
}
