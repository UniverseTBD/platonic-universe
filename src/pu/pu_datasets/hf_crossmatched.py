from typing import Callable, Iterable
from datasets import load_dataset
from pu.pu_datasets.base import DatasetAdapter
from pu.pu_datasets.registry import register_dataset

class HFCrossmatchedAdapter(DatasetAdapter):
    """Adapter for the generic HF crossmatched datasets (used for jwst, legacysurvey)."""

    def load(self) -> None:
        # No external resources required for this adapter.
        return None

    def prepare(self, processor: Callable, modes: Iterable[str], filterfun: Callable):
        """Prepare a streaming dataset that selects image columns, filters, maps and removes raw images."""
        if (self.comp_mode == "jwst") or (self.comp_mode == "legacysurvey"):
            ds = (
                load_dataset(self.hf_ds, split="train", streaming=True)
                .select_columns([f"{mode}_image" for mode in modes])
                .filter(filterfun)
                .map(processor)
                .remove_columns([f"{mode}_image" for mode in modes])
            )
            return ds
        else:
            raise NotImplementedError(
                f"HFCrossmatchedAdapter does not support comp_mode '{self.comp_mode}'"
            )

# Register this adapter for both aliases that use the HF crossmatched flow.
for alias in ("jwst", "legacysurvey"):
    register_dataset(alias, HFCrossmatchedAdapter)
