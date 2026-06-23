.PHONY: install data check notebooks clean

install:
	pip install -e ".[all]"

data:
	python scripts/download_zenodo_data.py --extract

check:
	python scripts/00_check_setup.py

notebooks:
	bash scripts/run_notebooks.sh

clean:
	rm -rf __pycache__ breastgnn/__pycache__ .pytest_cache
