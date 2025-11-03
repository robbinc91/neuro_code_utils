import nibabel as nib
import ants
import pyrobex


def load_mri(mri_path):
    # Step 1: Load the MRI using nibabel
    mri = nib.load(mri_path)
    return mri


def n4_correction(mri):
    # Step 2: Perform N4 intensity inhomogeneity correction over the loaded MRI, using ANTs
    n4 = ants.n4_bias_field_correction(mri)
    return n4['output']


def extract_brain_mask(mri):
    # Step 3: Extract a brain mask from the original MRI using PyRobex
    mask = pyrobex.extract_brain(mri.get_fdata())
    return mask


def extract_template_mask(template_path):
    # Step 4: Extract a brain mask from a template MRI using PyRobex
    template = nib.load(template_path)
    mask = pyrobex.extract_brain(template.get_fdata())
    return mask


def rigid_registration(template_mask, target_mask, output_transform_path):
    # Step 5: Align the mask from the step 4 to the mask from the step 5 using ANTs (rigid transformation).
    # Save the best transformation to a file.
    fixed = ants.from_numpy(template_mask)
    moving = ants.from_numpy(target_mask)
    transform = ants.registration(
        fixed=fixed, moving=moving, type_of_transform='Rigid')
    transform['fwdtransforms'][0].to_file(output_transform_path)
    return transform


def apply_rigid_transform(target_image, transform_path, output_path):
    # Step 6: Apply the best transformation to the MRI obtained in step 2. Save the result image.
    transform = ants.read_transform(transform_path)
    warped = ants.apply_transforms(
        fixed=target_image, moving=target_image, transformlist=transform)
    nib.save(warped, output_path)


def apply_inverse_transform(target_image, transform_path, output_path):
    # Step 7: Create a function that permits to apply the inverse transformation of step 5.
    transform = ants.read_transform(transform_path)
    inverse_transform = transform.invert()
    warped = ants.apply_transforms(
        fixed=target_image, moving=target_image, transformlist=inverse_transform)
    nib.save(warped, output_path)
