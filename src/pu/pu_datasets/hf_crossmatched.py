from typing import Callable, Iterable, Optional, List
from datasets import load_dataset
from pu.pu_datasets.base import DatasetAdapter
from pu.pu_datasets.registry import register_dataset


class HFCrossmatchedAdapter(DatasetAdapter):
    """Adapter for the generic HF crossmatched datasets (used for jwst, legacysurvey)."""

    def load(self) -> None:
        return None

    def prepare(
        self, 
        processor: Callable, 
        modes: Iterable[str], 
        filterfun: Callable,
        physical_params: Optional[List[str]] = None,
    ):
        """
        Prepare a streaming dataset that selects image columns, filters, maps and removes raw images.
        
        Args:
            processor: Preprocessing callable
            modes: List of modes (e.g., ['hsc', 'jwst'])
            filterfun: Filter function
            physical_params: Optional list of physical parameter columns to preserve
        """
        if self.comp_mode not in ("jwst", "legacysurvey"):
            raise NotImplementedError(
                f"HFCrossmatchedAdapter does not support comp_mode '{self.comp_mode}'"
            )
        
        # Determine columns to select
        image_columns = [f"{mode}_image" for mode in modes]
        columns_to_select = list(image_columns)
        
        if physical_params:
            columns_to_select.extend(physical_params)
        
        # Build the dataset pipeline
        ds = load_dataset(self.hf_ds, split="train", streaming=True)
        
        # Select columns (only those that exist)
        ds = ds.select_columns(columns_to_select)
        ds = ds.filter(filterfun)
        
        # Create a wrapper processor that preserves physical params
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
        
        # Remove only the image columns
        ds = ds.remove_columns(image_columns)
        
        return ds


# Register this adapter for both aliases that use the HF crossmatched flow.
for alias in ("jwst", "legacysurvey"):
    register_dataset(alias, HFCrossmatchedAdapter)
