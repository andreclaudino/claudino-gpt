import os

from torch.utils.data import DataLoader

from claudino_gpt.persistence.lazy_parquet_dataset import (
    LazyParquetDataset,
    RepeatedOrInfiniteDataset
)


def load_parquet_data(
    source_path: str,
    features_column_name: str,
    label_column_name: str,
    batch_size: int,
    max_seq_length: int,
    data_loading_workers_count: int,
    infinite: bool = False,
    epochs: int = 1,
) -> DataLoader:
    """
    Load a partition from multiple Parquet files lazily.

    Args:
        source_path (str): Base path to the dataset.
        features_column_name (str): Name of the input features column.
        label_column_name (str): Name of the label column.
        batch_size (int): Batch size for the DataLoader.
        max_seq_length (int): Expected sequence length for casting.
        data_loading_workers_count (int): Number of worker processes.
        infinite (bool): If True, repeat data indefinitely (ignores `epochs`).
        epochs (int): Number of times to repeat the dataset (ignored if `infinite=True`).

    Returns:
        DataLoader: A PyTorch DataLoader for lazy loading.
    """
    file_paths = [
        os.path.join(source_path, f)
        for f in os.listdir(source_path)
        if f.endswith('.parquet')
    ]

    if not file_paths:
        raise FileNotFoundError(f"No .parquet files found in {source_path}")

    lazy_dataset = LazyParquetDataset(
        file_paths=file_paths,
        features_column_name=features_column_name,
        label_column_name=label_column_name,
        max_seq_length=max_seq_length,
    )

    # Aplica repetição ou modo infinito
    if infinite or epochs > 1:
        dataset = RepeatedOrInfiniteDataset(
            base_dataset=lazy_dataset,
            epochs=epochs,
            infinite=infinite
        )
    else:
        dataset = lazy_dataset

    dataset_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Mantido como False (já que é lazy e ordenado)
        pin_memory=True,
        num_workers=data_loading_workers_count
    )

    return dataset_loader  # type: ignore

