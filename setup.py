from setuptools import setup, find_packages

setup(
    name='lca_birdy',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['pandas', 'openpyxl'],
    author='Birdy',
    description='Lightweight LCA tool using Excel + IPCC 2021',
    license='MIT',
    include_package_data=True,
)