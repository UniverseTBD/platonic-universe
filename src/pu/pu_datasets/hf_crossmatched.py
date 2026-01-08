import os
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
        """Prepare the dataset, optionally using streaming if enabled."""
        if (self.comp_mode == "jwst") or (self.comp_mode == "legacysurvey"):
            streaming = True
            if os.environ.get("PU_NO_STREAMING", "").lower() in ("1", "true", "yes"):
                streaming = False
            if os.environ.get("PU_STREAMING", "").lower() in ("0", "false", "no"):
                streaming = False
            ds = (
                load_dataset(self.hf_ds, split="train", streaming=streaming)
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
