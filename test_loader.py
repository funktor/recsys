import gcsfs
import pyarrow.parquet as pq

def count_rows_in_gcs_parquet(parquet_path:str):
    """
    Counts the total number of rows across multiple Parquet files in a GCS bucket path.

    Args:
        bucket_path (str): The Google Cloud Storage path (e.g., "gs://your-bucket/your-folder/").

    Returns:
        int: The total number of rows.
    """
    # Initialize the GCSFileSystem
    fs = gcsfs.GCSFileSystem()
    
    # Use pyarrow to open the dataset without reading the actual data
    # parquet_path is assumed to be in the following format: gs://[bucket-name]/**/*.parquet
    parquet_paths = parquet_path.split("/")
    parquet_paths = parquet_paths[2:-1]
    parquet_dir = "/".join(parquet_paths)

    dataset = pq.ParquetDataset(parquet_dir, filesystem=fs)
    
    # Sum the row counts from the metadata of each fragment (file)
    total_rows = sum(fragment.count_rows() for fragment in dataset.fragments)
    return total_rows


datasets_gcs_path = "gs://r6-ae-dev-adperf-adintelligence-data/amondal/parquet_dataset_ml_32m"
ratings_train_path = f"{datasets_gcs_path}/train/*.parquet"
num_train_data = count_rows_in_gcs_parquet(ratings_train_path)
print(num_train_data)