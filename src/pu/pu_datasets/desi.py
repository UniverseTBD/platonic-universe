from typing import Callable, Iterable
from datasets import load_dataset, concatenate_datasets
from pu.pu_datasets.base import DatasetAdapter
from pu.pu_datasets.registry import register_dataset

class DESIAdapter(DatasetAdapter):
    """Adapter for the DESI case that concatenates specformer embeddings."""

    def load(self) -> None:
        # No external resources required for this adapter.
        return None

    def prepare(self, processor: Callable, modes: Iterable[str], filterfun: Callable):
        ds = (
            concatenate_datasets(
                (
                    load_dataset(self.hf_ds, split="train", streaming=True),
                    load_dataset("Smith42/specformer_desi", split="train", streaming=True),
                ),
                axis=1,
            )
            .rename_column("image", "hsc_image")
            .select_columns(["hsc_image", "embeddings"])
            .filter(filterfun)
            .map(processor)
            .remove_columns(["hsc_image"])
        )
        # See hf_crossmatched.py: required when streaming=False so .map() output
        # is returned as torch tensors instead of Python lists.
        if hasattr(ds, "with_format"):
            ds = ds.with_format("torch")
        return ds

# Register adapter
register_dataset("desi", DESIAdapter)
