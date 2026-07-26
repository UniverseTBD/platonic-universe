from typing import Callable, Iterable

from datasets import load_dataset

from pu.pu_datasets.base import DatasetAdapter
from pu.pu_datasets.registry import register_dataset

# Maps physics parameter name → dataset column name for Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2.
CATALOG_COLUMNS = {
    "redshift": "lephare_photozs",
    "mag_g":    "mag_model_hsc-g",
    "mag_r":    "mag_model_hsc-r",
    "mass":     "lp_mass",
    "sSFR":     "lp_ssfr",
}


class CosmosWebAdapter(DatasetAdapter):
    """Adapter for Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2.

    comp_mode selects the telescope band: "hsc" or "jwst".
    Image column convention: {comp_mode}_images.
    """

    def load(self) -> None:
        return None

    def prepare(self, processor: Callable, modes: Iterable[str], filterfun: Callable):
        image_cols = [f"{mode}_images" for mode in modes]
        ds = (
            load_dataset(self.hf_ds, split="train", streaming=True)
            .select_columns(image_cols)
            .filter(filterfun)
            .map(processor)
            .remove_columns(image_cols)
        )
        if hasattr(ds, "with_format"):
            ds = ds.with_format("torch")
        return ds


register_dataset("cosmosweb", CosmosWebAdapter)
