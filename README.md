NLP_tweets
==============================

NLP model for tweets classification

## Getting Started

### Installation process
```
python setup.py install
```

### Install Table of Content extension for jupyter notebook

- Ensure `jupyter_contrib_nbextensions`` package is installed (it is in the dev requirements file)
- Launch the following commands:
    - ``jupyter contrib nbextension install --user``
    - ``jupyter nbextension enable toc2/main``
- To export your notebook in HTML with the table of contents:
    - ``jupyter nbconvert YOUR_NOTEBOOK.ipynb --template toc2``


## Software dependencies
- Python 3.6, 3.7

Project Organization
------------
```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a date (for ordering),
│                         the creator's initials, and a short `_` delimited description, e.g.
│                         `2020_06_01-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── version.py     <- Source code version
│
└── setup.cfg          <- Setup configuration
```


## Latest releases
## Build and Test
## Contribute
