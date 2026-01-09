from typing import Callable, Iterable, Optional, List
from datasets import load_dataset, concatenate_datasets
from pu.pu_datasets.base import DatasetAdapter
from pu.pu_datasets.registry import register_dataset


class SDSSAdapter(DatasetAdapter):
    """Adapter for the SDSS case that concatenates an external SDSS dataset."""

    def load(self) -> None:
        # No external resources required for this adapter.
        return None

    def prepare(
        self, 
        processor: Callable, 
        modes: Iterable[str], 
        filterfun: Callable,
        physical_params: Optional[List[str]] = None,
    ):
        base_columns = ["hsc_image", "embedding"]
        if physical_params:
            base_columns.extend(physical_params)
        
        ds = (
            concatenate_datasets(
                (
                    load_dataset(self.hf_ds, split="train", streaming=True),
                    load_dataset("Shashwat20/SDSS_Interpolated", split="train", streaming=True),
                ),
                axis=1,
            )
            .rename_column("image", "hsc_image")
            .select_columns(base_columns)
            .filter(filterfun)
        )
        
        if physical_params:
            def processor_with_params(example):
                result = processor(example)
                for param in physical_params:
                    if param in example:
                        result[param] = example[param]
                return result
            ds = ds.map(processor_with_params)
        else:
            ds = ds.map(processor)
        
        ds = ds.remove_columns(["hsc_image"])
        return ds


register_dataset("sdss", SDSSAdapter)
