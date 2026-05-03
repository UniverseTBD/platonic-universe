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
        # Cosmosweb stores per-band images under the plural keys
        # `<mode>_images`, but the rest of the pipeline (notably
        # `pu.preprocess.PreprocessTransform`) reads the singular
        # `<mode>_image`. Rename on the way in so downstream code is
        # survey-agnostic.
        plural_cols = [f"{mode}_images" for mode in list(modes)]
        rename_map = {f"{mode}_images": f"{mode}_image" for mode in list(modes)}
        ds = (
            load_dataset(self.hf_ds, split="train", streaming=True)
            .select_columns(plural_cols)
            .rename_columns(rename_map)
            .filter(filterfun)
            .map(processor)
            .remove_columns(list(rename_map.values()))
        )
        if hasattr(ds, "with_format"):
            ds = ds.with_format("torch")
        return ds


register_dataset("cosmosweb", CosmosWebAdapter)
