# ==============================================================================
# Script Name: generate_figure7_sensitivity.py
# Description: Generates the Tornado Chart for Figure 7 in the manuscript.
#              Reads data from the dynamically generated sensitivity_results.csv.
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 1. Define robust paths matching your local setup
# We use the current working directory and point to "monte_carlo_outputs"
BASE_DIR = Path.cwd()
INPUT_FILE = BASE_DIR / "monte_carlo_outputs" / "sensitivity_results.csv"
OUTPUT_DIR = BASE_DIR / "monte_carlo_outputs"
OUTPUT_FILE = OUTPUT_DIR / "figure6_rcpi_sensitivity_computer_hardware.png"

# 2. Check if the input file exists
if not INPUT_FILE.exists():
    print(f"Error: Could not find '{INPUT_FILE}'.")
    print("Please run 'circular_return_simulation.py' first to generate the data.")
    exit()

# 3. Load actual sensitivity results
df = pd.read_csv(INPUT_FILE)

sector = "Computer hardware refurbishment"

# Select sector and relevant parameters
params_to_show = [
    "virgin_material_cost_per_unit",
    "recovery_yield_mean",
    "processing_cost",
    "transport_cost",
    "rho_high"
]

labels = {
    "virgin_material_cost_per_unit": "Virgin Material Cost",
    "recovery_yield_mean": "Recovery Yield",
    "processing_cost": "Processing Cost",
    "transport_cost": "Transport Cost",
    "rho_high": "Return Probability"
}

plot_df = df[
    (df["Sector"] == sector) &
    (df["Parameter"].isin(params_to_show))
].copy()

# Add formatted labels
plot_df["Parameter_Label"] = plot_df["Parameter"].map(labels)

# Separate negative and positive variations
minus_df = plot_df[plot_df["Direction"] == "minus_20_percent"]
plus_df = plot_df[plot_df["Direction"] == "plus_20_percent"]

# Merge for plotting
merged = minus_df[["Parameter", "Parameter_Label", "Delta_RCPI"]].merge(
    plus_df[["Parameter", "Delta_RCPI"]],
    on="Parameter",
    suffixes=("_minus", "_plus")
)

# Sort by the absolute maximum impact
merged["max_abs"] = merged[["Delta_RCPI_minus", "Delta_RCPI_plus"]].abs().max(axis=1)
merged = merged.sort_values("max_abs", ascending=True)

# 4. Plotting
fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(merged))

ax.barh(
    y_pos,
    merged["Delta_RCPI_minus"],
    align="center",
    color="coral",
    label="-20% variation"
)

ax.barh(
    y_pos,
    merged["Delta_RCPI_plus"],
    align="center",
    color="teal",
    label="+20% variation"
)

# Formatting the chart
ax.set_yticks(y_pos)
ax.set_yticklabels(merged["Parameter_Label"])
ax.set_xlabel("Change in mean RCPI")
ax.set_title("One-at-a-time sensitivity analysis of RCPI\nComputer hardware refurbishment")
ax.axvline(0, linewidth=1, color="black")
ax.legend()

# 5. Save and Show
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
print(f"Success! Figure saved to: {OUTPUT_FILE}")
plt.show()
