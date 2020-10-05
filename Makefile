init-repo: ## initialize the repo
	python setup.py install

update-package: ## install the functions
	pip uninstall nlp-tweets -y
	python setup.py install
