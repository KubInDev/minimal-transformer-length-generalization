# Minimal Transformer - Length Generalisation Experiments

A minimal PyTorch transformer trained on synthetic formal-language classification tasks to study how positional encoding affects out-of-distribution (OOD) length generalisation.

## Requirements
- Python 3.10+
- PyTorch 2.0+
- NumPy, Matplotlib

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate # Windows
# source .venv/bin/activate # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

## Running experiments

All experiments are launched from `main.py` in the project root.

### Full run (train + plot)

Trains all task × PE-type combinations across 5 seeds, saves results, and generates plots:

```bash
python main.py
```

### Common flags

| Flag | Description | Default |
|---|---|---|
| `--tasks` | Tasks to run | all five |
| `--pe` | Positional encodings (`learned`, `sin`, `none`) | all three |
| `--seeds N` | Number of random seeds | `5` |
| `--no-save` | Skip saving results to disk | off |

Example - run only two tasks with learned PE across 3 seeds:

```bash
python main.py --tasks First-Last "A*B*" --pe learned --seeds 3
```

### Modes (mutually exclusive)

| Flag | What it does |
|---|---|
| `--plot-only` | Regenerate plots from a previously saved `results/results.pkl` |
| `--ablations-only` | CLS vs last-token readout ablation on First-Last |
| `--attention-only` | Visualise attention heatmaps for each task |
| `--attention-extended` | Extended mechanistic attention analysis |
| `--multi-length` | Train on variable-length sequences, test OOD |
| `--cot` | Chain-of-thought auxiliary heads vs standard model |

Examples:

```bash
# Re-plot without re-training
python main.py --plot-only

# Readout ablation with learned and sinusoidal PE
python main.py --ablations-only --pe learned sin

# CoT experiment with 3 seeds
python main.py --cot --seeds 3
```

### Standalone scripts

Individual experiments can also be run directly:

```bash
python -m experiments.run_length_gen          # length generalisation sweep
python -m experiments.run_ablations           # readout ablation
python -m experiments.run_cot                 # CoT vs standard
python -m experiments.run_multi_length_train  # variable-length training
python plot_only.py                           # quick re-plot shortcut
```

## Tasks

| Name | Description |
|---|---|
| `First-Last` | Label 1 if `seq[0] == seq[-1]` |
| `A*B*` | Label 1 if all 0s precede all 1s |
| `Parity` | Label = sum of bits mod 2 |
| `Contains-11` | Label 1 if any adjacent 1-1 pair exists |
| `Even-Pairs` | Label 1 if the count of 1s is even |

## Output

| Path | Contents |
|---|---|
| `results/results.pkl` | Serialised experiment results |
| `plots/` | Generated figures (PNG) |

Key hyperparameters are in `config.py` (model size, training length, seeds, etc.).