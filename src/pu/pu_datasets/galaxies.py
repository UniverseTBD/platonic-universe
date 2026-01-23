"""Dataset adapter for Smith42/galaxies - 8.5M galaxy cutouts from DESI Legacy Survey DR8."""
from typing import Callable, Iterable, Optional, List
from datasets import load_dataset
from pu.pu_datasets.base import DatasetAdapter
from pu.pu_datasets.registry import register_dataset


class GalaxiesAdapter(DatasetAdapter):
    """Adapter for Smith42/galaxies. Use revision v2.0 to include metadata."""

    AVAILABLE_PHYSICAL_PARAMS = {
        # Identifiers
        'dr8_id', 'ra', 'dec', 'brickid', 'objid', 'file_name', 'iauname',
        # Morphology (Galaxy Zoo)
        'smooth-or-featured_smooth_fraction', 'smooth-or-featured_featured-or-disk_fraction',
        'smooth-or-featured_artifact_fraction', 'disk-edge-on_yes_fraction', 'disk-edge-on_no_fraction',
        'has-spiral-arms_yes_fraction', 'has-spiral-arms_no_fraction',
        'bar_strong_fraction', 'bar_weak_fraction', 'bar_no_fraction',
        'bulge-size_dominant_fraction', 'bulge-size_large_fraction', 'bulge-size_moderate_fraction',
        'bulge-size_small_fraction', 'bulge-size_none_fraction',
        'how-rounded_round_fraction', 'how-rounded_in-between_fraction', 'how-rounded_cigar-shaped_fraction',
        'edge-on-bulge_boxy_fraction', 'edge-on-bulge_none_fraction', 'edge-on-bulge_rounded_fraction',
        'spiral-winding_tight_fraction', 'spiral-winding_medium_fraction', 'spiral-winding_loose_fraction',
        'spiral-arm-count_1_fraction', 'spiral-arm-count_2_fraction', 'spiral-arm-count_3_fraction',
        'spiral-arm-count_4_fraction', 'spiral-arm-count_more-than-4_fraction', 'spiral-arm-count_cant-tell_fraction',
        'merging_none_fraction', 'merging_minor-disturbance_fraction', 'merging_major-disturbance_fraction', 'merging_merger_fraction',
        # Petrosian
        'est_petro_th50', 'petro_theta', 'petro_th50', 'petro_th90', 'petro_phi50', 'petro_phi90', 'petro_ba50', 'petro_ba90',
        'elpetro_ba', 'elpetro_phi', 'elpetro_flux_r', 'elpetro_theta_r', 'elpetro_mass', 'elpetro_mass_log',
        'elpetro_absmag_f', 'elpetro_absmag_n', 'elpetro_absmag_u', 'elpetro_absmag_g', 'elpetro_absmag_r', 'elpetro_absmag_i', 'elpetro_absmag_z',
        # Sersic
        'sersic_n', 'sersic_ba', 'sersic_phi',
        'sersic_nmgy_f', 'sersic_nmgy_n', 'sersic_nmgy_u', 'sersic_nmgy_g', 'sersic_nmgy_r', 'sersic_nmgy_i', 'sersic_nmgy_z',
        # Magnitudes
        'mag_r_desi', 'mag_g_desi', 'mag_z_desi',
        'mag_f', 'mag_n', 'mag_u', 'mag_g', 'mag_r', 'mag_i', 'mag_z', 'u_minus_r',
        # Redshifts
        'redshift', 'redshift_nsa', 'redshift_ossy', 'photo_z', 'photo_zerr', 'spec_z',
        # OSSY
        'dr7objid_ossy', 'ra_ossy', 'dec_ossy', 'log_l_oiii', 'fwhm', 'e_fwhm', 'equiv_width',
        'log_l_ha', 'log_m_bh', 'upper_e_log_m_bh', 'lower_e_log_m_bh', 'log_bolometric_l',
        # ALFALFA
        'ra_alf', 'dec_alf', 'W50', 'sigW', 'W20', 'HIflux', 'sigflux', 'SNR', 'RMS', 'Dist', 'sigDist', 'logMH', 'siglogMH',
        # JHU SFR
        'ra_jhu', 'dec_jhu',
        'fibre_sfr_avg', 'fibre_sfr_entropy', 'fibre_sfr_median', 'fibre_sfr_mode', 'fibre_sfr_p16', 'fibre_sfr_p2p5', 'fibre_sfr_p84', 'fibre_sfr_p97p5',
        'fibre_ssfr_avg', 'fibre_ssfr_entropy', 'fibre_ssfr_median', 'fibre_ssfr_mode', 'fibre_ssfr_p16', 'fibre_ssfr_p2p5', 'fibre_ssfr_p84', 'fibre_ssfr_p97p5',
        'total_ssfr_avg', 'total_ssfr_entropy', 'total_ssfr_flag', 'total_ssfr_median', 'total_ssfr_mode', 'total_ssfr_p16', 'total_ssfr_p2p5', 'total_ssfr_p84', 'total_ssfr_p97p5',
        'total_sfr_avg', 'total_sfr_entropy', 'total_sfr_flag', 'total_sfr_median', 'total_sfr_mode', 'total_sfr_p16', 'total_sfr_p2p5', 'total_sfr_p84', 'total_sfr_p97p5',
        # Photo-z
        'photoz_id', 'ra_photoz', 'dec_photoz', 'mag_abs_g_photoz', 'mag_abs_r_photoz', 'mag_abs_z_photoz',
        'mass_inf_photoz', 'mass_med_photoz', 'mass_sup_photoz', 'sfr_inf_photoz', 'sfr_sup_photoz',
        'ssfr_inf_photoz', 'ssfr_med_photoz', 'ssfr_sup_photoz', 'sky_separation_arcsec_from_photoz',
        'est_petro_th50_kpc',
    }

    def load(self) -> None:
        pass

    def prepare(
        self,
        processor: Callable,
        modes: Iterable[str],
        filterfun: Callable,
        physical_params: Optional[List[str]] = None,
    ):
        ds = (
            load_dataset(self.hf_ds, split="train", streaming=True, revision="v2.0")
            .rename_column("image", "galaxies_image")
            .select_columns(["galaxies_image"] + (physical_params or []))
            .filter(filterfun)
        )

        if physical_params:
            def processor_with_params(example):
                result = processor(example)
                for p in physical_params:
                    if p in example:
                        result[p] = example[p]
                return result
            ds = ds.map(processor_with_params)
        else:
            ds = ds.map(processor)

        return ds.remove_columns(["galaxies_image"])


register_dataset("galaxies", GalaxiesAdapter)
