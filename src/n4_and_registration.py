import nibabel as nib
import numpy as np
import ants
import pyrobex

# Step 1: Load the MRI using nibabel
mri_path = "path/to/mri.nii.gz"
mri = nib.load(mri_path)
mri_data = mri.get_fdata()

# Step 2: Perform N4 intensity inhomogeneity correction using ANTs
n4 = ants.n4_bias_field_correction(mri)
n4_data = n4['output'].get_fdata()

# Step 3: Extract a brain mask from the original MRI using PyRobex
brain_mask = pyrobex.extract_brain(n4_data)

# Step 4: Extract a brain mask from a template MRI using PyRobex
template_path = "path/to/template.nii.gz"
template = nib.load(template_path)
template_data = template.get_fdata()
template_mask = pyrobex.extract_brain(template_data)

# Step 5: Align the mask from step 4 to the mask from step 3 using ANTs
fixed = ants.from_numpy(brain_mask)
moving = ants.from_numpy(template_mask)
transform = ants.registration(
    fixed=fixed, moving=moving, type_of_transform='Rigid')
registered_mask = transform['warpedmovout'].numpy()

# Step 6: Apply the best transformation to the MRI obtained in step 2
aligned_data = ants.apply_transforms(
    fixed=mri,
    moving=n4['output'],
    transformlist=transform['fwdtransforms'],
    interpolator='linear',
    default_value=0
)

# Save the aligned data to a file using nibabel
aligned_image = nib.Nifti1Image(aligned_data.numpy(), mri.affine, mri.header)
nib.save(aligned_image, "path/to/aligned_mri.nii.gz")
