import os

from torch.utils.data import DataLoader

from claudino_gpt.persistence.lazy_parquet_dataset import LazyParquetDataset


def load_parquet_data(
    source_path: str,
    features_column_name: str,
    label_column_name: str,
    batch_size: int,
    max_seq_length: int,
    data_loading_workers_count: int
) -> DataLoader:
    """
    Load a partition from multiple Parquet files lazily.

    This function returns a DataLoader that will load data from Parquet files
    on-demand, one file at a time, without loading the entire dataset into memory.

    Args:
        source_path (str): Base path to the dataset.
        features_column_name (str): Name of the input features column.
        label_column_name (str): Name of the label column.
        batch_size (int): Batch size for the DataLoader.
        max_seq_length (int): Expected sequence length for casting.

    Returns:
        DataLoader: A PyTorch DataLoader for lazy loading.
    """    # List all .parquet files in the partition directory
    file_paths = [
        os.path.join(source_path, f)
        for f in os.listdir(source_path)
        if f.endswith('.parquet')
    ]

    if not file_paths:
        raise FileNotFoundError(f"No .parquet files found in {source_path}")

    # Create the lazy dataset
    lazy_dataset = LazyParquetDataset(
        file_paths=file_paths,
        features_column_name=features_column_name,
        label_column_name=label_column_name,
        max_seq_length=max_seq_length,
    )

    # Create and return the DataLoader
    dataset_loader = DataLoader(
        lazy_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=data_loading_workers_count
    )

    return dataset_loader # type: ignore

