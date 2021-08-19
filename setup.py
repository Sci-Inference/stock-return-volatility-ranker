import os
from setuptools import setup

required = [
    'numpy',
    'pandas',
    'pyarrow',
    'scipy',
    'Sphinx',
    'rinohtype',
    'nbsphinx',
    'pytest-dependency',
    'scipy',
    'matplotlib',
    'sklearn',
    'joblib'
]

with open("README.md", "r") as fh:
    long_description = fh.read() 

setup(
    name='time_series_transform',
    version='1.1.3',
    description = 'Implementation of various learning to rank neural network',
    packages=[
        'stock_rv_ranker',
        'stock_rv_ranker/layers',
        'stock_rv_ranker/losses',
        'stock_rv_ranker/metrics',
        'stock_rv_ranker/utils',
        'stock_rv_ranker/test',
        'stock-rv_ranker/model'
        ],
    license='MIT',
    author_email = 'kuanlun.chiang@outlook.com',
    url = 'https://github.com/Sci-Inference/stock-return-volatility-ranker',
    project_urls = {
        'Source Code' : 'https://github.com/Sci-Inference/stock-return-volatility-ranker',
        # 'Documentation' : "https://allen-chiang.github.io/Time-Series-Transformer/"
    },
    # download_url ='https://github.com/allen-chiang/Time-Series-Transformer/archive/1.1.2.tar.gz',
    keywords = ['time series','stock', 'machine learning', 'deep learning'],
    install_requires = required,
    author = 'Sci-Inference',
    long_description= long_description,
    long_description_content_type='text/markdown',
    classifiers=[
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',      
    'Programming Language :: Python',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: MIT License', 
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
