"""SimpleITK-based brain MRI registration (no ANTs, no PyRobex).

This module provides `simple_brain_registration` which performs an
N4 bias-field correction (optional), a simple skull-strip based on Otsu
thresholding and connected components (optional), and a 3-stage registration:
Rigid -> Affine -> BSpline. It uses SimpleITK for all transforms and IO.
"""
from pathlib import Path
from typing import Dict, Any

import numpy as np
import nibabel as nib
import SimpleITK as sitk


def _otsu_brain_mask(sitk_img: sitk.Image, closing_radius: int = 5) -> sitk.Image:
    mask = sitk.OtsuThreshold(sitk_img, 0, 1)
    mask = sitk.BinaryMorphologicalClosing(mask, [closing_radius] * sitk_img.GetDimension())
    cc = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    if stats.GetNumberOfLabels() == 0:
        return mask
    # pick largest component
    largest_label = max(stats.GetLabels(), key=lambda l: stats.GetNumberOfPixels(l))
    brain_mask = sitk.Equal(cc, largest_label)
    brain_mask = sitk.Cast(brain_mask, sitk.sitkUInt8)
    return brain_mask


def _register(fixed: sitk.Image, moving: sitk.Image, transform, metric='Mattes', verbose=False):
    registration = sitk.ImageRegistrationMethod()

    if metric == 'Mattes':
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    else:
        registration.SetMetricAsMeanSquares()

    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetInitialTransform(transform, inPlace=False)
    final_transform = registration.Execute(sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32))
    if verbose:
        print(f"Final metric value: {registration.GetMetricValue()}")
    return final_transform


def simple_brain_registration(
    mri_path: str,
    template_path: str,
    output_dir: str,
    do_n4: bool = True,
    do_skullstrip: bool = True,
    save_warped: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Register `mri_path` to `template_path` using SimpleITK.

    Returns a dict with `warped_path` and `transforms` (saved transform file paths).
    """
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Read via SimpleITK for convenience
    fixed = sitk.ReadImage(str(template_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(mri_path), sitk.sitkFloat32)

    working_moving = moving

    transforms = []

    # N4 bias correction
    if do_n4:
        if verbose:
            print("Running N4 bias field correction (SimpleITK)...")
        mask = _otsu_brain_mask(working_moving)
        working_moving = sitk.N4BiasFieldCorrection(working_moving, mask)

    # Skull-strip via Otsu + largest component
    brain_mask = None
    if do_skullstrip:
        if verbose:
            print("Computing brain mask via Otsu (no PyRobex)...")
        brain_mask = _otsu_brain_mask(working_moving)
        working_moving = sitk.Mask(working_moving, brain_mask)

    # Stage 1: Rigid
    if verbose:
        print("Stage 1: Rigid registration...")
    initial_transform = sitk.CenteredTransformInitializer(fixed, working_moving, sitk.VersorRigid3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    rigid_tform = _register(fixed, working_moving, initial_transform, verbose=verbose)
    rigid_path = str(outp / 'transform_rigid.tfm')
    sitk.WriteTransform(rigid_tform, rigid_path)
    transforms.append(rigid_path)

    # Resample moving by rigid to speed up next stage
    moving_rigid = sitk.Resample(working_moving, fixed, rigid_tform, sitk.sitkLinear, 0.0, moving.GetPixelID())

    # Stage 2: Affine
    if verbose:
        print("Stage 2: Affine registration...")
    affine_init = sitk.AffineTransform(3)
    # initialize affine from rigid
    affine_init.SetMatrix(rigid_tform.GetMatrix())
    affine_tform = _register(fixed, moving_rigid, affine_init, verbose=verbose)
    affine_path = str(outp / 'transform_affine.tfm')
    sitk.WriteTransform(affine_tform, affine_path)
    transforms.append(affine_path)

    moving_affine = sitk.Resample(working_moving, fixed, affine_tform, sitk.sitkLinear, 0.0, moving.GetPixelID())

    # Stage 3: BSpline non-linear
    if verbose:
        print("Stage 3: BSpline registration (non-linear)...")
    grid_physical_spacing = [50.0, 50.0, 50.0]
    image_physical_size = [size*spacing for size, spacing in zip(fixed.GetSize(), fixed.GetSpacing())]
    mesh_size = [max(int(image_physical_size[i]/grid_physical_spacing[i]+0.5), 1) for i in range(3)]
    bspline_transform = sitk.BSplineTransformInitializer(image1=fixed, transformDomainMeshSize=mesh_size, order=3)

    # Use registration with BSpline transform
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100, maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000, costFunctionConvergenceFactor=1e+7)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(bspline_transform, inPlace=False)

    bspline_tform = registration.Execute(sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving_affine, sitk.sitkFloat32))
    bspline_path = str(outp / 'transform_bspline.tfm')
    sitk.WriteTransform(bspline_tform, bspline_path)
    transforms.append(bspline_path)

    # Composite transform: apply transforms in order (BSpline, Affine, Rigid)
    composite = sitk.Transform(3, sitk.sitkComposite)
    composite.AddTransform(bspline_tform)
    composite.AddTransform(affine_tform)
    composite.AddTransform(rigid_tform)

    # Apply to original moving image (not masked) to produce final warped image
    final_warped = sitk.Resample(sitk.ReadImage(str(mri_path), sitk.sitkFloat32), fixed, composite, sitk.sitkLinear, 0.0, moving.GetPixelID())

    warped_path = None
    if save_warped:
        warped_path = str(outp / 'mri_simple_warped.nii.gz')
        sitk.WriteImage(final_warped, warped_path)

    return {
        'warped_path': warped_path,
        'transforms': transforms,
    }
