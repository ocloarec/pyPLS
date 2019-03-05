from setuptools import setup, find_packages

setup(
    name='pyPLS',
    version='0.1.1',
    packages=find_packages(),
    url='',
    license='MIT',
    author='Olivier Cloarec',
    author_email='ocloarec@korrigan.co.uk',
    description='A package for Projection on Latent Structures',
    requires=['numpy', 'pandas', 'scipy']
)
