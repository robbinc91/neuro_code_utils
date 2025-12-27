from setuptools import setup, find_packages

setup(
    name="neuro_code_utils",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "nibabel",
    ],
    entry_points={
        "console_scripts": [
            "neuro-batch-augment=src.batch_augment:_cli",
        ]
    },
)
