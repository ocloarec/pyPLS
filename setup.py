from setuptools import setup, find_packages

setup(
    name='pyPLS',
    version='0.2.0',
    packages=find_packages(),
    url='',
    license='MIT',
    author='Olivier Cloarec',
    author_email='ocloarec@mac.com',
    description='A package for Projection on Latent Structures',
    requires=['numpy', 'pandas', 'scipy']
)
