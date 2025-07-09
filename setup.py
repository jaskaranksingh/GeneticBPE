from setuptools import setup, find_packages

setup(
    name="genetic_bpe",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "requests>=2.26.0",
        "pytest>=6.2.5",
        "scikit-learn>=0.24.2",
        "tqdm>=4.62.0",
    ],
    author="Jaskaran",
    description="GeneticBPE: A motif-preserving tokenizer for miRNA sequences",
    python_requires=">=3.7",
) 