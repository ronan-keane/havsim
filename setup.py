
import setuptools

with open('README.md', 'r') as fh: 
    long_description = fh.read()
    
setuptools.setup(
        name = 'havsim',
        version = '0.0.1',
        author = 'ronan-keane',
        author_email = 'rkeane@umich.edu',
        description='Traffic simulator implemented in python with an advanced parametric model of human driving.',
        long_description = long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/ronan-keane/havsim',
        classifiers = [
                'Programming Language :: Python :: 3',
                'License :: Apache 2.0',
                'Operating System :: Windows/Linux/Mac'
        ],
        install_requires=['palettable', 'scipy', 'bayesian-optimization', 'tqdm', 'matplotlib'],
        python_requires = '>=3.10',
        packages = setuptools.find_packages(exclude = ['Jax','nlopt','scripts', 'tensorflow', 'tensorflow-probability', 'networkx'])
        )
