import os
from pathlib import Path
from typing import Optional, Dict, Any

import nibabel as nib
import ants
import pyrobex
import numpy as np


def optimal_brain_registration(
    mri_path: str,
    template_path: str,
    output_dir: str,
    do_n4: bool = True,
    do_skullstrip: bool = True,
    save_warped: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Perform an optimal 3D brain MRI registration pipeline.

    Pipeline:
      1. Load MRI and template
      2. Optional N4 bias-field correction
      3. Optional skull-strip via PyRobex (masking)
      4. Multi-stage registration: Rigid -> Affine -> SyN(non-linear)
      5. Apply final transforms to the full MRI and (optionally) segmentation

    Returns a dictionary with keys: `warped_path`, `transforms` (list of transform paths/strings),
    and the `ants_registration` dict returned by ANTs for the final stage.
    """
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Load images via nibabel then convert to ANTs images
    mri_nii = nib.load(mri_path)
    mri_data = mri_nii.get_fdata()
    template_nii = nib.load(template_path)
    template_data = template_nii.get_fdata()

    # Create ANTs images
    ants_mri = ants.from_numpy(mri_data)
    ants_template = ants.from_numpy(template_data)

    # N4 correction
    if do_n4:
        if verbose:
            print("Running N4 bias field correction...")
        ants_mri = ants.n4_bias_field_correction(ants_mri)

    # Skull-strip / masking
    if do_skullstrip:
        if verbose:
            print("Extracting brain mask with PyRobex...")
        mask = pyrobex.extract_brain(ants_mri.numpy())
        # Ensure boolean mask
        mask = (mask > 0).astype(np.uint8)
        masked_np = ants_mri.numpy() * mask
        ants_mri = ants.from_numpy(masked_np)

    # Stage 1: Rigid
    if verbose:
        print("Running rigid registration...")
    rigid = ants.registration(fixed=ants_template, moving=ants_mri, type_of_transform='Rigid')

    # Stage 2: Affine (initialize with rigid)
    if verbose:
        print("Running affine registration...")
    affine = ants.registration(fixed=ants_template, moving=ants_mri, initial_transform=rigid['fwdtransforms'], type_of_transform='Affine')

    # Stage 3: SyN (non-linear) initialized with affine
    if verbose:
        print("Running SyN (non-linear) registration...")
    syn = ants.registration(fixed=ants_template, moving=ants_mri, initial_transform=affine['fwdtransforms'], type_of_transform='SyN')

    # Apply final transforms to original (possibly N4-corrected) MRI
    final_transform_list = syn.get('fwdtransforms', [])

    # Save transform file paths (they may already be filepaths returned by ANTs)
    transforms_saved = []
    for i, t in enumerate(final_transform_list):
        try:
            # t is often a string filepath; if not, write to disk
            if isinstance(t, str) and os.path.exists(t):
                transforms_saved.append(t)
            else:
                fp = outp / f'transform_stage_final_{i}.mat'
                ants.write_transform(t, str(fp))
                transforms_saved.append(str(fp))
        except Exception:
            # fallback: store repr
            transforms_saved.append(str(t))

    # Apply transforms to the (optionally N4) ants_mri; use the template as fixed
    warped = ants.apply_transforms(fixed=ants_template, moving=ants_mri, transformlist=final_transform_list, interpolator='linear')

    warped_np = warped.numpy() if hasattr(warped, 'numpy') else np.asarray(warped)

    warped_path = None
    if save_warped:
        warped_nii = nib.Nifti1Image(warped_np, mri_nii.affine, mri_nii.header)
        warped_path = str(outp / 'mri_warped_to_template.nii.gz')
        nib.save(warped_nii, warped_path)

    return {
        'warped_path': warped_path,
        'transforms': transforms_saved,
        'ants_registration': syn,
    }
