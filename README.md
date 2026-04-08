# BreastGNN – Modular Package

GNN-based breast cancer molecular subtype classification using HuRI + OmniPath
protein-protein interaction networks as graph priors.

## Important
Data must be downloaded from Zenodo: 10.5281/zenodo.19476386

## Structure

```
breastgnn_package/
├── breastgnn/                  # Python package (17 modules)
│   ├── __init__.py             # Package root
│   ├── config.py               # BreastGNNConfig dataclass (~80 hyperparameters)
│   ├── utils.py                # Seeds, memory, formatting
│   ├── data.py                 # Data loading, gene prep, splits, scaling, DataLoaders
│   ├── graph.py                # OmniPath + HuRI graph construction, regulator features
│   ├── model.py                # ResGATBlock, SignedResGATBlock, ImprovedSharedGraphGNN, HybridGNNTabular
│   ├── losses.py               # FocalLoss, AUC, compute_metrics_full
│   ├── training.py             # predict_proba, train_one_epoch, train_graph_learning, finetune_pruned
│   ├── pruning.py              # export_pruned_graph, evaluate_keep_ratios
│   ├── stability.py            # Jaccard edge-set utilities, stability_edge_sets
│   ├── ablation.py             # AblationConfig, build/save/load, run_single_seed, run_ablation
│   ├── axes.py                 # Biological axes definitions, scores, OVR analysis, plotting
│   ├── enrichment.py           # gProfiler, Enrichr, KEGG map overlay
│   ├── visualization.py        # Graph plots, dotplot, barplot, components grid
│   ├── bootstrap.py            # Bootstrap CI for classification metrics and AUC
│   ├── benchmarks.py           # PAM50, LightGBM, CatBoost, GCN, GraphSAGE, SOTA
│   └── postprocessing.py       # OVR thresholds, gene importance (Grad×Input)
│
├── notebooks/                  # Clean notebooks that import from breastgnn
│   ├── 0_Main_Pipeline.ipynb           # Full training pipeline
│   ├── 1_Results_Bootstrap.ipynb       # Evaluation + bootstrap CI
│   ├── 2_Axes_Refined.ipynb            # Biological axes analysis
│   ├── 3_Axes_Figure.ipynb             # Enrichment + KEGG overlay
│   ├── 4_Gene_Lists_by_Axis.ipynb      # Supplementary gene tables
│   ├── 5_Bioinformatics_Benchmarks.ipynb  # Tabular + fixed-prior benchmarks
│   └── 6_SOTA_Benchmarks.ipynb         # PAM50, GCN, GraphSAGE, SOTA
│── create data/                  # donwload data
│   ├── 1.-Download_Microarrays_Compact # Download Microarrys GEOS
│   ├── 6.-Download_RNASEQ_Union-Copy1  # Download RNASEQ GEOS
│   ├── 7.-Integracion_buena            # Integration dataa
│
└── README.md
```



## Quick Start

```python
# In any notebook:
import sys
sys.path.insert(0, "..")  # if notebooks/ is a subdirectory

from breastgnn.config import CFG
from breastgnn.data import load_expression_and_metadata, prepare_genes, encode_labels
from breastgnn.graph import build_graph
from breastgnn.ablation import run_ablation

# Override config
CFG.DATA_DIR = Path("./my_data")
CFG.LR = 1e-3

# Load data
X_df, y_str, cohort = load_expression_and_metadata(CFG.EXPR_CSV, CFG.META_CSV)
X_df_kegg, genes_kegg = prepare_genes(X_df)
y, classes, label_map = encode_labels(y_str)

# Build graph
edge_index, edge_weight, edge_type = build_graph(genes_kegg, CFG.DATA_DIR)

# ... etc.
```

## Key Design Decisions

1. **`config.py`**: All ~80 hyperparameters are centralised in `BreastGNNConfig`. 
   A singleton `CFG` is importable from any module. Override attributes before calling functions.

2. **No global state**: Functions in `data`, `graph`, `losses`, `training`, `pruning` 
   receive all inputs as parameters — no `globals()` lookups.

3. **`ablation.py`**: Contains `run_single_seed` and `run_ablation` which still rely on 
   some globals (the full training pipeline state). These are the "runner" functions that 
   orchestrate the entire flow and are designed to be called from notebooks that have 
   set up the necessary variables.

4. **`benchmarks.py`**: Contains the benchmark code from notebooks 5 and 6. These functions 
   use `_require_globals()` to validate that the necessary data is available before running.

## Dependencies

- Python 3.10+
- PyTorch
- scikit-learn
- pandas, numpy, matplotlib, networkx
- mygene (for HuRI SYMBOL→ENSG mapping)
- gprofiler-official and/or gseapy (for enrichment)
- scipy, statsmodels (for axes OVR analysis)
