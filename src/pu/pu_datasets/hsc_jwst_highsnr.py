from typing import Callable, Iterable, Optional, List
import numpy as np
from datasets import load_dataset
from pu.pu_datasets.base import DatasetAdapter
from pu.pu_datasets.registry import register_dataset

class HSCJWSTHighSNRAdapter(DatasetAdapter):
    """
    Adapter for the Ashodkh/hsc-jwst-images-high-snr dataset.
    
    This dataset stores images as flat float32 lists with separate shape columns,
    and includes rich physical parameters from LePhare SED fitting.
    """

    AVAILABLE_PHYSICAL_PARAMS = {
        # Redshifts
        'specz', 'photoz', 'lephare_photozs',
        # LePhare SED fitting results  
        'lp_mass', 'lp_sfr', 'lp_ssfr', 'lp_age', 'lp_type',
        'lp_zpdf_l68', 'lp_zpdf_u68',
        # Morphology
        'disk_radius', 'bulge_radius',
        # Magnitudes (HSC)
        'mag_model_hsc-g', 'mag_model_hsc-r', 'mag_model_hsc-i',
        'mag_model_hsc-z', 'mag_model_hsc-y', 'mag_model_cfht-u',
        # Magnitudes (JWST)
        'mag_model_f115w', 'mag_model_f150w', 'mag_model_f277w', 'mag_model_f444w',
        # Coordinates
        'ra', 'dec',
    }

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
        Prepare the streaming dataset, reshaping flat image arrays back to 3D.
        """
	if physical_params:
	    invalid = set(physical_params) - self.AVAILABLE_PHYSICAL_PARAMS
	    if invalid:
	        raise ValueError(
		    f"Unknown physical params: {invalid}. "
		    f"Available: {sorted(self.AVAILABLE_PHYSICAL_PARAMS)}"
		)

        columns_to_select = ['hsc_image', 'hsc_shape', 'jwst_image', 'jwst_shape']
	if physical_params:
	    columns_to_select.extend(physical_params)

	ds = load_dataset(self.hf_ds, split="train", streaming=True)
	ds = select_columns(columns_to_select)
	ds = ds.filter(filterfun)

	def reshape_and_process(example):
	    hsc_arr = np.array(example['hsc_image'], dtype=np.float32)
            hsc_shape = tuple(example['hsc_shape'])
            hsc_img = hsc_arr.reshape(hsc_shape)

            jwst_arr = np.array(example['jwst_image'], dtype=np.float32)
            jwst_shape = tuple(example['jwst_shape'])
            jwst_img = jwst_arr.reshape(jwst_shape)

            reformatted = {
	        'hsc_image': {'flux': hsc_img},
	        'jwst_image': {'flux': jwst_img},
	    }

	    result = processor(reformatted)

	    if physical_params:
                for param in physical_params:
		    if param in example:
		        result[param] = example[param]

            return result
	
	ds = ds.map(reshape_and_process)
	ds = ds.remove_columns(['hsc_image', 'hsc_shape', 'jwst_image', 'jwst_shape'])

        return ds

register_dataset("hsc_jwst_highsnr", HSCJWSTHighSNRAdapter)
