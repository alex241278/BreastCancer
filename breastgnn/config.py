"""
Centralised configuration: paths, hyperparameters, and novelty toggles.

Usage in notebooks::

    from breastgnn.config import CFG
    CFG.LR = 1e-3          # override any default
    CFG.DATA_DIR = Path("./my_data")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional

import torch

from .utils import get_device


@dataclass
class BreastGNNConfig:
    """All hyper-parameters in a single, overridable object."""

    # ── Paths ──────────────────────────────────────────────
    DATA_DIR: Path = Path("../Listados_tumores/output_combat/").resolve()
    EXPR_CSV: Optional[Path] = None  # set in __post_init__
    META_CSV: Optional[Path] = None

    SAMPLE_COL: str = "sample"
    COHORT_COL: str = "batch"
    LABEL_COL: str = "label"
    PIPELINE_CACHE_DIR: Path = (DATA_DIR / "pipeline_cache").resolve()

    # ── Splits ─────────────────────────────────────────────
    SEED: int = 42
    TEST_SIZE: float = 0.20
    VAL_SIZE: float = 0.20

    # ── Scaling ────────────────────────────────────────────
    USE_QUANTILE: bool = False
    SCALE_MODE: str = "standard"  # "standard" | "minmax_-1_1" | "none"

    # ── Interactome (HuRI) ─────────────────────────────────
    USE_HURI: bool = True
    HURI_DATASETS: Tuple[str, ...] = ("HuRI",)
    HURI_CACHE_DIR: Path        = (DATA_DIR / "interactome_cache").resolve()
    USE_HURI_CONFIDENCE: bool = True
    HURI_MIN_SCORE: float = 0.0
    HURI_DEFAULT_WEIGHT: float = 1.0
    FORCE_REBUILD_HURI_CACHE: bool = True

    # ── OmniPath ───────────────────────────────────────────
    USE_OMNIPATH: bool = True
    OMNIPATH_CACHE_DIR: Path = (DATA_DIR / "omnipath_cache").resolve()
    OMNIPATH_URL: str        = (
        "https://omnipathdb.org/interactions"
        "?datasets=omnipath&genesymbols=1&directed=1&signed=1&format=tsv"
    )

    # ── Graph construction ─────────────────────────────────
    ADD_SELF_LOOPS_IN_GRAPH: bool = False
    CONNECTED_ONLY: bool = True

    # ── Regulator features (X_graph) ──────────────────────
    ADD_REGULATOR_FEATURES: bool = True
    REG_STATS: Tuple[str, ...] = ("mean", "std", "max")
    REG_MIN_GENES: int = 5
    REG_MAX_REGULATORS: Optional[int] = None

    # ── Training (Phase 1 – graph learning) ────────────────
    BATCH_SIZE: int = 20
    ACCUM_STEPS: int = 8
    PATIENCE: int = 30
    FOCAL_GAMMA: float = 2.0

    LR: float = 2e-3
    WEIGHT_DECAY: float = 1e-4

    # ── Gates / sparsity ──────────────────────────────────
    GATE_TAU_START: float = 2.0
    GATE_TAU_END: float = 0.7
    EDGE_L1_PER_EDGE: float = 1e-5

    # ── GNN backbone ──────────────────────────────────────
    HIDDEN: int = 64
    NUM_LAYERS: int = 3
    DROPOUT: float = 0.10
    NUM_HEADS: int = 4       # GAT heads
    POOL_HEADS: int = 4      # pooling heads
    XGRAPH_DROPOUT: float = 0.00

    # ── Block / bypass ────────────────────────────────────
    KEEP_SELF_LOOPS: bool = False
    BLOCK_USE_SELF: bool = True
    BLOCK_RESIDUAL: bool = True

    # ── Loss / pruning ────────────────────────────────────
    AUX_LAMBDA: float = 0.10
    KEEP_MIN: float = 0.004

    # ── Data augmentation (Phase 2) ──────────────────────
    USE_MIXUP: bool = True
    MIXUP_ALPHA: float = 0.15
    MIXUP_P: float = 0.30

    # ── Epochs ────────────────────────────────────────────
    EPOCHS1: int = 90        # Phase 1 (graph learning)
    FT_EPOCHS_A: int = 5     # Phase 2A (Xh=0)
    FT_EPOCHS_B: int = 80    # Phase 2B (Xh=orig)

    # ── Novelty toggles ──────────────────────────────────
    EDGE_TYPE_GATING: bool = True
    SAMPLE_COND_GATING: bool = True
    SAMPLE_COND_MODE: str = "per_type"

    SIGNED_CHANNELS: bool = False
    SIGNED_CHANNELS_MODE: str = "type_only"

    ADD_CONNECTIVITY_PENALTY: bool = True
    CONNECTIVITY_LAMBDA: float = 0.002
    CONNECTIVITY_MIN_DEG: float = 0.05
    CONNECTIVITY_USE_ABS: bool = True

    DO_PRETRAIN: bool = False
    DO_STABILITY_SELECTION: bool = True

    # ── Stability selection ──────────────────────────────
    STAB_RUNS: int = 8
    STAB_EPOCHS: int = 60
    STAB_KEEP_FINAL: float = 0.004
    STAB_FREQ_THR: float = 0.5

    # ── Hybrid model (Phase 2) ──────────────────────────
    USE_HYBRID_MODEL: bool = True
    HYBRID_TAB_LAYERS: int = 2
    HYBRID_TAB_DROPOUT: float = 0.20
    HYBRID_FUSION_DROPOUT: float = 0.20
    HYBRID_BLEND_LOGITS: bool = False
    HYBRID_BLEND_INIT: float = 0.70

    # ── Artifacts ─────────────────────────────────────────
    ARTIFACTS_ROOT: str = "./artifacts_ablation"
    SAVE_ALL_CONFIGS: bool = True
    RUN_FINAL_AFTER_ABLATION: bool = True
    FINAL_SELECT_METRIC: str = "macro_f1"

    # ── Device ────────────────────────────────────────────
    DEVICE: Optional[torch.device] = None

    def __post_init__(self):
        if self.EXPR_CSV is None:
            self.EXPR_CSV = self.DATA_DIR / "expr_combat_corrected.csv"
        if self.META_CSV is None:
            self.META_CSV = self.DATA_DIR / "metadata_combined.csv"
        if self.DEVICE is None:
            self.DEVICE = get_device()


# Singleton – import this in notebooks
CFG = BreastGNNConfig()
