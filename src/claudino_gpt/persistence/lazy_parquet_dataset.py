from typing import Any, Iterable, Tuple

from torch.utils.data import Dataset
from tqdm import tqdm
import polars as pl


class LazyParquetDataset(Dataset):
    """
    A PyTorch Dataset that lazily loads data from multiple Parquet files.

    It does not load all files into memory at once. Instead, it loads a file
    only when data from that file is requested.
    """

    def __init__(
        self,
        file_paths: Iterable[str],
        features_column_name: str,
        label_column_name: str,
        max_seq_length: int,
    ):
        """
        Initialize the dataset.

        Args:
            file_paths (List[str]): List of paths to Parquet files.
            features_column_name (str): Name of the column containing input features.
            label_column_name (str): Name of the column containing labels.
            max_seq_length (int): The expected sequence length for casting.
        """
        self.file_paths = file_paths
        self.features_column_name = features_column_name
        self.label_column_name = label_column_name
        self.max_seq_length = max_seq_length

        # Pre-calculate the starting index and length for each file
        self._file_index_map = []  # List of (start_idx, end_idx, file_path)
        self._total_length = 0

        for file_path in tqdm(self.file_paths, desc="Tota files loaded"):
            # Get the number of rows in the file without loading its content
            # Polars can read metadata quickly
            current_file = pl.scan_parquet(file_path)
            num_rows = current_file.select(pl.len()).collect().item()
            start_idx = self._total_length
            end_idx = start_idx + num_rows
            self._file_index_map.append((start_idx, end_idx, file_path))
            self._total_length = end_idx

    def __len__(self) -> int:
        return self._total_length

    def _find_file_for_index(self, global_index: int) -> Tuple[int, str]:
        """
        Find which file contains the data for the given global index.

        Returns:
            Tuple[int, str]: The local index within the file and the file path.
        """
        for start_idx, end_idx, file_path in self._file_index_map:
            if start_idx <= global_index < end_idx:
                local_index = global_index - start_idx
                return local_index, file_path
        raise IndexError(f"Global index {global_index} out of range.")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get a single data sample.

        Args:
            index (int): Global index of the sample.

        Returns:
            Tuple[Any, Any]: A tuple (features, label).
        """
        local_index, file_path = self._find_file_for_index(index)

        # Load only the specific row from the specific file
        # We use `pl.read_parquet` with `n_rows` and `skip_rows` for efficiency
        # Alternatively, `pl.scan_parquet().slice(local_index, 1).collect()`
        df = pl.read_parquet(
            file_path,
            n_rows=1,
            row_index_offset=local_index
        )

        # Cast the features column
        df = df.with_columns(
            pl.col(self.features_column_name).cast(pl.Array(pl.Int64, width=self.max_seq_length))
        )

        # Extract the single row as tensors
        features = df[self.features_column_name].to_torch()
        label = df[self.label_column_name].to_torch()

        # Since we loaded one row, features and label are tensors of shape (1, ...)
        # We squeeze to remove the batch dimension of 1
        return features.squeeze(0), label.squeeze(0)


class RepeatedOrInfiniteDataset(Dataset):
    """
    Wrapper around LazyParquetDataset to support:
      - Finite repetition (epochs > 1)
      - Infinite repetition (infinite=True)
    """

    def __init__(
        self,
        base_dataset: LazyParquetDataset,
        epochs: int = 1,
        infinite: bool = False,
    ):
        if infinite and epochs != 1:
            # Quando infinite=True, epochs é ignorado
            pass
        self.base_dataset = base_dataset
        self.epochs = epochs
        self.infinite = infinite
        self._base_len = len(base_dataset)

        if self.infinite:
            # Para DataLoader funcionar com sampler, damos um "comprimento fictício"
            # suficientemente grande. Usamos um valor arbitrário alto.
            # NOTA: Em modo infinito, __len__ não deve ser usado para controlar epochs!
            self._effective_length = 10**12  # 1 trilhão — "infinito" para efeitos práticos
        else:
            self._effective_length = self._base_len * self.epochs

    def __len__(self) -> int:
        return self._effective_length

    def __getitem__(self, index: int):
        if self.infinite:
            # Mapeia índice para o dataset base com módulo
            base_index = index % self._base_len
        else:
            if index >= self._effective_length:
                raise IndexError(f"Index {index} out of range for dataset of length {self._effective_length}")
            base_index = index % self._base_len
        return self.base_dataset[base_index]