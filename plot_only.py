"""
plot_only.py
Extra file for plot generation only for an existing trained model
"""
from utils import load_results
from analysis.plots import plot_length_generalization, plot_training_curves, print_summary_table
import config

results = load_results()
print_summary_table(results)
plot_length_generalization(results, save_dir=config.PLOTS_DIR)
plot_training_curves(results, save_dir=config.PLOTS_DIR)