from setuptools import setup, find_packages

setup(
    name='DSMC',
    version='0.1',
    author='Antti Isosalo',
    author_email='antti.isosalo@oulu.fi',
    packages=find_packages(),
    include_package_data=True,
    license='LICENSE',
    long_description=open('README.md').read(),
    project_urls={
    }, install_requires=['pandas==0.22.0', 'numpy==1.19.2', 'scipy==1.5.2', 'sklearn==0.24.2', 'torch==1.1.0', 'torchvision==0.3', 'h5py==2.7.1', 'imageio==2.4.1', 'opencv-python==3.4.2.17', 'qhoptim==1.1.0', 'tqdm==4.64.1', 'termcolor==1.1.0', 'tensorboardx==1.4']
)
