"""
Dataset adapter for Smith42/galaxies (v2.0).

This adapter loads galaxy images with their bundled metadata for physics
validation tests.  Unlike the crossmatched adapters (which compare two
modalities), this one provides single-modality images plus physical
property labels.

Usage with the v2.0 revision which includes metadata columns directly:
    galaxies = load_dataset("Smith42/galaxies", revision="v2.0", streaming=True)
"""

from typing import Callable, Iterable

from datasets import load_dataset

from pu.pu_datasets.base import DatasetAdapter
from pu.pu_datasets.registry import register_dataset


# Metadata columns that we want to keep through the pipeline for physics tests.
# These are the columns from Smith42/galaxies_metadata that ship with v2.0.
METADATA_COLUMNS = [
    "dr8_id",
    # Morphology
    "smooth-or-featured_smooth_fraction",
    "smooth-or-featured_featured-or-disk_fraction",
    "smooth-or-featured_artifact_fraction",
    "has-spiral-arms_yes_fraction",
    "has-spiral-arms_no_fraction",
    "bar_strong_fraction",
    "bar_weak_fraction",
    "bar_no_fraction",
    "bulge-size_dominant_fraction",
    "bulge-size_large_fraction",
    "bulge-size_moderate_fraction",
    "bulge-size_small_fraction",
    "bulge-size_none_fraction",
    "disk-edge-on_yes_fraction",
    "disk-edge-on_no_fraction",
    "merging_merger_fraction",
    "merging_major-disturbance_fraction",
    "merging_minor-disturbance_fraction",
    "merging_none_fraction",
    # Photometry
    "mag_r",
    "mag_g",
    "mag_z",
    "u_minus_r",
    # Structure
    "sersic_n",
    "sersic_ba",
    "petro_th50",
    "petro_th90",
    "elpetro_ba",
    "elpetro_theta_r",
    # Physical
    "elpetro_mass_log",
    "redshift",
    # Star formation
    "total_sfr_median",
    "total_ssfr_median",
]


class GalaxiesAdapter(DatasetAdapter):
    """Adapter for Smith42/galaxies v2.0 with bundled metadata."""

    def load(self) -> None:
        return None

    def prepare(
        self,
        processor: Callable,
        modes: Iterable[str],
        filterfun: Callable,
        split: str = "test",
        max_samples: int | None = None,
    ):
        """Prepare a streaming dataset of galaxy images with metadata.

        Args:
            processor: Image preprocessor callable
            modes: Not used (single modality), kept for interface compat
            filterfun: Row filter callable
            split: Which split to use ("test", "validation", "train")
            max_samples: If set, take only the first N samples (for speed)

        Returns:
            HuggingFace dataset ready for iteration
        """
        ds = load_dataset(
            self.hf_ds,
            revision="v2.0",
            split=split,
            streaming=True,
        )

        if max_samples is not None:
            ds = ds.take(max_samples)

        ds = (
            ds
            .filter(filterfun)
            .map(processor)
        )

        return ds


register_dataset("galaxies", GalaxiesAdapter)
