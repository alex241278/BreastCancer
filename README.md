# BreastGNN — self-contained reproducibility package

BreastGNN is a modular graph neural network pipeline for molecular subtype classification in breast cancer using expression data and prior biological graphs. 

The data source configured for this package is:

```text
DOI: 10.5281/zenodo.19476488
Zenodo record id: 19476488
```

## Quick start

Create the environment:

```bash
conda env create -f environment.yml
conda activate breastgnn
pip install -e ".[all]"
```

For non-conda installations:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[all]"
```

Use Python 3.10 or 3.11. The package metadata intentionally excludes Python 3.13 because several scientific and graph-learning dependencies may not yet provide stable wheels for that version.

For GPU runs, install the PyTorch and PyTorch Geometric builds that match your CUDA version before running the notebooks.

Download and arrange the data:

```bash
python scripts/download_zenodo_data.py --extract
python scripts/00_check_setup.py
```

Then run the notebooks in order:

```text
0_Main_Pipeline.ipynb
1_Results_Bootstrap.ipynb
2_Ejes_REFINED_AXES.ipynb
3_Ejes_Figura_residuales.ipynb
4_Lista_genes_por_eje.ipynb
5_Bioinformatics_Benchmarks.ipynb
6_Benchmarks_SOTA.ipynb
```

Or execute them from the shell:

```bash
bash scripts/run_notebooks.sh
```

## Required data files

The core pipeline expects:

```text
data/processed/expr_combat_corrected.csv
data/processed/metadata_combined.csv
```

The download script attempts to find these files automatically inside the Zenodo files. If Zenodo stores them under slightly different names, the script searches common alternatives and copies them into the expected layout. If automatic detection fails, manually copy or rename the files to the two paths above.

## Optional external graph resources

The graph builder can use local copies of:

```text
data/external/omnipath_interactions.tsv
data/external/HuRI.tsv
data/external/HuRI.psi
```

If these files are absent, `breastgnn.graph` can still try to download OmniPath and HuRI resources during graph construction. Keeping local copies is preferable for offline and reproducible reruns.

## Useful commands

```bash
make install     # pip install -e .[all]
make data        # download/extract Zenodo data
make check       # verify required local layout
make smoke       # import package modules and check notebook metadata
make notebooks   # execute notebooks in order
make clean       # remove Python caches
```

## Environment variables

All important paths can be overridden without editing code:

```bash
export BREASTGNN_DATA_DIR=/path/to/processed_data
export BREASTGNN_EXPR_CSV=/path/to/expr_combat_corrected.csv
export BREASTGNN_META_CSV=/path/to/metadata_combined.csv
export BREASTGNN_RAW_DATA_DIR=/path/to/raw_zenodo_files
export BREASTGNN_EXTERNAL_DATA_DIR=/path/to/external_graph_files
export BREASTGNN_CACHE_ROOT=/path/to/cache
export BREASTGNN_ARTIFACTS_ROOT=/path/to/artifacts_ablation
```

## Main code changes relative to the previous package

1. `breastgnn/config.py` now resolves all paths from the repository root and stores caches/artifacts locally.
2. `scripts/download_zenodo_data.py` downloads files from Zenodo record `19476488`, verifies MD5 checksums when available, extracts archives and prepares the expected data layout.
3. `scripts/00_check_setup.py` validates that the required processed CSV files exist before running notebooks.
4. `breastgnn/data.py` now raises a clear `FileNotFoundError` with setup instructions if required input files are missing.
5. `breastgnn/graph_cache.py` uses `CFG.PIPELINE_CACHE_DIR` rather than creating a cache inside the processed-data directory.
6. `breastgnn/benchmarks.py` now auto-registers package-level objects needed by fixed-prior controls, so notebooks 5 and 6 no longer fail because `make_prior_for_cfg` or `AblationConfig` were not passed manually.
7. Notebooks 5 and 6 explicitly register `make_prior_for_cfg` and `AblationConfig` in the benchmark runtime.
8. All notebooks were cleared of previous outputs and absolute local paths.


## Validation checklist

Before launching the full workflow, run:

```bash
python scripts/01_smoke_test.py
python scripts/00_check_setup.py
```

The smoke test checks that core package modules import correctly and that notebooks do not contain stored outputs, execution counts, or obvious absolute local paths. The setup check verifies that the required processed CSV files are available.

## Reproducibility notes

The package follows standard computational reproducibility practices: relative paths, explicit environment files, stable data DOI, checksum manifest, notebook execution order and local cache/artifact directories. The file `REFERENCES.bib` includes methodological and software references relevant to FAIR/reproducible workflows, HuRI, OmniPath, PyTorch and PyTorch Geometric.

## Citation

Use the Zenodo DOI for the data:

```text
10.5281/zenodo.19476488
```

A software citation file is provided in `CITATION.cff` and points to the public repository URL.
