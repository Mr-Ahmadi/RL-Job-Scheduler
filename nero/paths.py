"""Every filesystem location the project reads or writes, in one place.

Paths are absolute, resolved from this file, so any entry point works whatever
the current directory is.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# inputs
DATA = ROOT / "data"
GAVEL_TABLE = DATA / "physical_all.json"        # Gavel co-location throughput table
TEST_SETS = DATA / "saved_job_sets"        # the 20 held-out test sets

# trained models
MODELS = ROOT / "models"
INNER = MODELS / "job_scheduling" / "ppo"                   # inner placement agent
SECONDARY = INNER / "secondary"                             # duplicate-placement agent
PRIMARY = INNER / "primary"                                 # frozen copy; never written
OUTER = MODELS / "subset_selector" / "ppo"                  # outer duplication agent
HEADS = MODELS / "job_scheduling" / "learned_topk"          # reward model, V, Q
ONLINE = MODELS / "job_scheduling" / "online"               # online-learning output

# outputs
RESULTS = ROOT / "results"
SCORES = RESULTS / "job_scheduling"        # evaluation scores and curve data
OUTER_CURVES = RESULTS / "subset_selector"
PAPER = ROOT / "paper"
FIGURES = PAPER / "images"
