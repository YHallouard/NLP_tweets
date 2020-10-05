from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

required = ['matplotlib',
            'numpy',
            'pandas',
            'notebook',
            'jupyter',
            'tensorflow>=2.2.0',
            'nltk',
            'sklearn',
            'gensim',
            'seaborn'
            ]

__version__ = 'init'
exec(open('nlp_tweets/version.py').read())

setup(
    name='nlp_tweets',
    packages=find_packages(),
    version=__version__,
    description="NLP model for tweets classification",
    author='Yann HALLOUARD',
    long_description=long_description,
    long_description_content_type="text/markdown",
    setup_requires=['pytest-runner', 'wheel', 'flake8'],
    tests_require=['pytest', 'pytest-cov', 'treon', 'coverage', 'coverage-badge'],
    install_requires=required,
    license='MIT\
    ',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
