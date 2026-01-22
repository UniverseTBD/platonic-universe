from abc import ABC, abstractmethod
from typing import Iterable, Callable, Optional, List, Set


class DatasetAdapter(ABC):
    """
    Minimal dataset adapter interface.

    Implementations should encapsulate all dataset-specific loading and column
    transformations so experiments only need to call `prepare(processor, modes, filterfun)`.
    """

    AVAILABLE_PHYSICAL_PARAMS: Set[str] = set()

    def __init__(self, hf_ds: str, comp_mode: str):
        self.hf_ds = hf_ds
        self.comp_mode = comp_mode

    @classmethod
    def list_physical_params(cls) -> List[str]:
        """Return list of physical parameters available in this dataset."""
        return sorted(cls.AVAILABLE_PHYSICAL_PARAMS)

    @abstractmethod
    def load(self) -> None:
        """Load any external resources required by this adapter (if any)."""
        raise NotImplementedError

    @abstractmethod
    def prepare(
        self, 
        processor: Callable, 
        modes: Iterable[str], 
        filterfun: Callable,
        physical_params: Optional[List[str]] = None,
    ):
        """
        Return a preprocessed `datasets.Dataset` ready for iteration.

        The adapter should:
        - load/concatenate/rename columns as needed for this dataset
        - call .filter(filterfun) when streaming is used
        - call .map(processor)
        - optionally preserve physical_params columns for downstream analysis

        Args:
            processor: callable returned by model_adapter.get_preprocessor(modes)
            modes: sequence of mode names used by experiments (e.g. ['hsc', 'jwst'])
            filterfun: callable used to filter streaming datasets
            physical_params: optional list of column names to preserve for analysis

        Returns:
            datasets.Dataset
        """
        raise NotImplementedError
