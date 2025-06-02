from setuptools import setup, find_packages

setup(
    name='wsitools',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'dask[array]',
        'scikit-image',
        'ome-types',
        'zarr',
        'dask-image',
        'scipy',
        'napari',
        'magicgui',
        'qtpy',
        'distributed',  # for dask.distributed
    ],
    description='Tools for analysing whole slide imaging datasets',
    author='Michael Haley',
    author_email='mrmichaelhaley@gmail.com',
    url='https://github.com/dr-michael-haley/wsi-tools',
)