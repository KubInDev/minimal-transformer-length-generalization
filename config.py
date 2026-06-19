"""
config.py
"""

# Experiment
TRAIN_LEN = 50
TEST_LENS = [20, 30, 40, 50, 60, 80, 100]
N_TRAIN = 20_000
N_TEST = 3_000
N_SEEDS = 5
SEEDS = [42, 7, 13, 99, 2024]

# PE types to run. Options: "learned" | "sin" | "none"
PE_TYPES = ["learned", "sin", "none"]

# Tasks to run. Must match keys in data/generators.py AKA. TASK_REGISTRY.
TASKS = ["First-Last", "A*B*", "Parity", "Contains-11", "Even-Pairs"]

# Model
D_MODEL = 32
NHEAD = 2  # even - avoids PyTorch nested-tensor warning
DIM_FF = 64
MAX_LEN = 200 # headroom for out of distribution
DROPOUT = 0.0 # intentionally zero - studying raw inductive bias

# Training
EPOCHS = 80
BATCH_SIZE = 128
LR = 1e-3
MIN_TRAIN_ACC = 0.99 # early-stop threshold (convergence)
PATIENCE = 15  # epochs without improvement before early stop (plateau)
WARMUP_EPOCHS = 20       # plateau detection starts after this epoch

# Output
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
RESULTS_FILE = "results.pkl"
