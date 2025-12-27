import os
import argparse
from pathlib import Path
import nibabel as nib

from MRIDataAugmenter import MRIDataAugmenter


def list_nifti_files(root_dir, recursive=False):
    p = Path(root_dir)
    if recursive:
        patterns = ("**/*.nii", "**/*.nii.gz")
    else:
        patterns = ("*.nii", "*.nii.gz")
    files = []
    for pat in patterns:
        files.extend([str(x) for x in p.glob(pat)])
    return sorted(files)


def batch_augment(input_dir, output_dir, n_augment=5, recursive=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_nifti_files(input_dir, recursive=recursive)
    if not files:
        print(f"No NIfTI files found in {input_dir}")
        return 0

    augmenter = MRIDataAugmenter()
    created = 0
    for fpath in files:
        img = nib.load(fpath)
        affine = img.affine
        hdr = img.header
        stem = Path(fpath).stem
        for i in range(n_augment):
            data = augmenter.augment_image(fpath)
            out_name = f"{stem}_aug{i+1}.nii.gz"
            out_path = output_dir / out_name
            out_img = nib.Nifti1Image(data.astype(img.get_data_dtype()), affine, hdr)
            nib.save(out_img, str(out_path))
            created += 1

    print(f"Created {created} augmented images in {output_dir}")
    return created


def _cli():
    parser = argparse.ArgumentParser(description="Batch augment NIfTI files using MRIDataAugmenter")
    parser.add_argument("input_dir", help="Input directory containing NIfTI files")
    parser.add_argument("output_dir", help="Output directory for augmented files")
    parser.add_argument("-n", "--n-augment", type=int, default=5, help="Augmentations per file")
    parser.add_argument("-r", "--recursive", action="store_true", help="Search directories recursively")
    args = parser.parse_args()
    batch_augment(args.input_dir, args.output_dir, args.n_augment, args.recursive)


if __name__ == "__main__":
    _cli()
