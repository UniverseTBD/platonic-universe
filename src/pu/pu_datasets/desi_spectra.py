from typing import Callable, Iterable
from datasets import load_dataset
from pu.pu_datasets.base import DatasetAdapter
from pu.pu_datasets.registry import register_dataset


class DESISpectraAdapter(DatasetAdapter):
    """Adapter for DESI that provides raw spectra for on-the-fly SpecFormer embedding.

    Unlike :class:`DESIAdapter` (which loads pre-computed specformer embeddings),
    this adapter passes raw spectrum flux arrays through to the SpecFormer
    preprocessor so embeddings can be generated from scratch.
    """

    def load(self) -> None:
        return None

    def prepare(self, processor: Callable, modes: Iterable[str], filterfun: Callable):
        ds = (
            load_dataset(self.hf_ds, split="train", streaming=True)
            .select_columns(["spectrum"])
            .filter(filterfun)
            .map(processor)
            .remove_columns(["spectrum"])
        )
        return ds


register_dataset("desi_spectra", DESISpectraAdapter)
