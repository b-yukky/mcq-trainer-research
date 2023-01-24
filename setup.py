from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'transformers==4.25.1',
    'tqdm==4.64.1',
    'pytorch-lightning>=1.7.0',
    'pandas>=1.4.0',
    'numpy==1.23.5',
    'scikit-learn>=1.0',
    'scipy>=1.7.0',
    'datasets==2.8.0',
    'typing_extensions==4.4.0'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Missing distractors generation for MCQ'
)
