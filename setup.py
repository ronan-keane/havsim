
"""
@author: rlk268@cornell.edu
"""

import setuptools

with open('README.readme', 'r') as fh: 
    long_description = fh.read()
    
setuptools.setup(
        name = 'havsim',
        version = '0.0.1',
        author = 'ronan-keane',
        author_email = 'rlk268@cornell.edu',
        description='A differentiable traffic simulator for calibration of traffic models and optimization of AV behavior',
        long_description = long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/ronan-keane/havsim',
        classifiers = [
                'Programming Language :: Python :: 3',
                'License :: Apache 2.0',
                'Operating System :: Windows/Linux/Mac'
        ],
        install_requires=['palettable', 'tensorflow', 'tensorflow-probability', 'scipy', 'networkx'],
        python_requires = '>=3.10',
        packages = setuptools.find_packages(exclude = ['Jax','nlopt','scripts'])
        )
