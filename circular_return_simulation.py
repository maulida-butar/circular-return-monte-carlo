# ==============================================================================
# Script Name: circular_return_simulation.py
# Description: A reproducible Monte Carlo framework for evaluating product 
#              returns, cost performance (RCPI), and circularity (MRR, VMD) 
#              in Circular Supply Chains.
# ==============================================================================
# ============================================================
# Monte Carlo Simulation for Product Returns in Circular Supply Chains
# Outputs:
# 1. monte_carlo_summary.csv
# 2. monte_carlo_raw_draws.csv
# 3. sensitivity_results.csv
# 4. RCPI, MRR, VMD, and CO2 plots
# ============================================================

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Global settings
# ============================================================

N_ITER = 10_000
PERIODS = 9
RANDOM_SEED = 42

OUTPUT_DIR = Path("monte_carlo_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Sector parameters
# Replace these values later if you obtain real firm-level data.
# These are case-informed archetype parameters.
# ============================================================

SECTOR_PARAMS = {
    "Aircraft manufacturing": {
        "demand_min": 16,
        "demand_max": 35,
        "rho_high": 0.60,
        "rho_low": 0.06,
        "beta_a": 7.0,
        "beta_b": 3.0,
        "sort_capacity": 25,
        "process_capacity": 18,
        "mass_per_unit": 8.0,
        "reintegration_rate": 0.85,
        "can_substitute_new_production": True,
        "collection_cost": 40.0,
        "holding_cost": 75.0,
        "sort_test_cost": 180.0,
        "transport_cost": 80.0,
        "processing_cost": 450.0,
        "disposal_cost": 60.0,
        "new_mfg_cost": 2800.0,
        "virgin_material_cost_per_unit": 350.0,
        "secondary_market_credit": 80.0,
        "ef_virgin": 4.0,
        "ef_recovery": 1.5,
        "ef_reverse_per_return": 0.7,
    },

    "Telecommunications": {
        "demand_min": 95,
        "demand_max": 105,
        "rho_high": 0.95,
        "rho_low": 0.095,
        "beta_a": 3.0,
        "beta_b": 5.0,
        "sort_capacity": 95,
        "process_capacity": 85,
        "mass_per_unit": 1.5,
        "reintegration_rate": 0.10,
        "can_substitute_new_production": False,
        "collection_cost": 8.0,
        "holding_cost": 4.0,
        "sort_test_cost": 12.0,
        "transport_cost": 10.0,
        "processing_cost": 28.0,
        "disposal_cost": 14.0,
        "new_mfg_cost": 0.0,
        "virgin_material_cost_per_unit": 0.0,
        "secondary_market_credit": 4.0,
        "ef_virgin": 2.8,
        "ef_recovery": 1.4,
        "ef_reverse_per_return": 0.25,
    },

    "Computer hardware refurbishment": {
        "demand_min": 25,
        "demand_max": 150,
        "rho_high": 0.90,
        "rho_low": 0.09,
        "beta_a": 7.0,
        "beta_b": 2.5,
        "sort_capacity": 120,
        "process_capacity": 100,
        "mass_per_unit": 2.2,
        "reintegration_rate": 0.80,
        "can_substitute_new_production": True,
        "collection_cost": 5.0,
        "holding_cost": 3.0,
        "sort_test_cost": 9.0,
        "transport_cost": 6.0,
        "processing_cost": 22.0,
        "disposal_cost": 8.0,
        "new_mfg_cost": 120.0,
        "virgin_material_cost_per_unit": 18.0,
        "secondary_market_credit": 8.0,
        "ef_virgin": 3.5,
        "ef_recovery": 1.2,
        "ef_reverse_per_return": 0.18,
    },

    "General retail": {
        "demand_min": 350,
        "demand_max": 650,
        "rho_high": 0.85,
        "rho_low": 0.085,
        "beta_a": 2.0,
        "beta_b": 6.0,
        "sort_capacity": 400,
        "process_capacity": 300,
        "mass_per_unit": 1.0,
        "reintegration_rate": 0.05,
        "can_substitute_new_production": False,
        "collection_cost": 3.0,
        "holding_cost": 1.5,
        "sort_test_cost": 5.0,
        "transport_cost": 3.5,
        "processing_cost": 9.0,
        "disposal_cost": 3.0,
        "new_mfg_cost": 0.0,
        "virgin_material_cost_per_unit": 0.0,
        "secondary_market_credit": 1.0,
        "ef_virgin": 1.4,
        "ef_recovery": 0.9,
        "ef_reverse_per_return": 0.10,
    },

    "Carpet manufacturing": {
        "demand_min": 20,
        "demand_max": 100,
        "rho_high": 0.70,
        "rho_low": 0.07,
        "beta_a": 6.0,
        "beta_b": 3.0,
        "sort_capacity": 85,
        "process_capacity": 80,
        "mass_per_unit": 12.0,
        "reintegration_rate": 0.75,
        "can_substitute_new_production": True,
        "collection_cost": 5.0,
        "holding_cost": 2.5,
        "sort_test_cost": 6.0,
        "transport_cost": 4.0,
        "processing_cost": 11.0,
        "disposal_cost": 4.0,
        "new_mfg_cost": 85.0,
        "virgin_material_cost_per_unit": 8.0,
        "secondary_market_credit": 4.0,
        "ef_virgin": 2.2,
        "ef_recovery": 0.8,
        "ef_reverse_per_return": 0.15,
    },
}


# ============================================================
# 3. Helper functions
# ============================================================

def summarize_array(x, prefix):
    """Return mean, median, SD, and empirical 95% confidence interval."""
    return {
        f"Mean_{prefix}": float(np.mean(x)),
        f"Median_{prefix}": float(np.median(x)),
        f"SD_{prefix}": float(np.std(x, ddof=1)),
        f"{prefix}_CI_2.5": float(np.percentile(x, 2.5)),
        f"{prefix}_CI_97.5": float(np.percentile(x, 97.5)),
    }


def classify_typology(row):
    """
    Return Absorptive Capacity Typology.
    Uses RCPI, confidence interval, probability of positive RCPI, MRR, and VMD.
    """
    mean_rcpi = row["Mean_RCPI"]
    lower_ci = row["RCPI_CI_2.5"]
    upper_ci = row["RCPI_CI_97.5"]
    p_positive = row["P_RCPI_Positive"]
    mean_mrr = row["Mean_MRR"]
    mean_vmd = row["Mean_VMD"]

    if (
        mean_rcpi > 0.10
        and lower_ci > 0
        and p_positive >= 0.95
        and (mean_mrr >= 0.45 or mean_vmd >= 0.25)
    ):
        return "Sunny"

    if mean_rcpi > 0 and p_positive >= 0.75:
        return "Cloudy"

    if lower_ci <= 0 <= upper_ci:
        return "Foggy"

    if mean_rcpi < 0 and upper_ci < 0:
        return "Rainy"

    return "Foggy"


def adjust_beta_mean(params, factor):
    """
    Adjust the mean of the Beta recovery-yield distribution.
    Keeps concentration constant.
    """
    a = params["beta_a"]
    b = params["beta_b"]
    concentration = a + b
    old_mean = a / concentration
    new_mean = np.clip(old_mean * factor, 0.01, 0.99)

    params["beta_a"] = new_mean * concentration
    params["beta_b"] = (1.0 - new_mean) * concentration

    return params


# ============================================================
# 4. Core simulation model
# ============================================================

def simulate_condition(params, demand, rho, rng):
    """
    Simulate one return condition, either high-return or low-return.

    Returns:
    TC, MRR, VMD, CO2_Avoided
    """
    n_iter, periods = demand.shape

    total_cost = np.zeros(n_iter)
    mrr_num = np.zeros(n_iter)
    mrr_den = np.zeros(n_iter)
    vmd_num = np.zeros(n_iter)
    vmd_den = np.zeros(n_iter)
    co2_avoided = np.zeros(n_iter)

    reverse_inventory = np.zeros(n_iter)
    forward_inventory = np.zeros(n_iter)

    for t in range(periods):
        D = demand[:, t].astype(float)

        # Stochastic return quantity
        returns = rng.binomial(D.astype(int), rho).astype(float)

        # Stochastic recovery yield, bounded between 0 and 1
        recovery_yield = rng.beta(
            params["beta_a"],
            params["beta_b"],
            size=n_iter
        )

        # Reverse inventory and capacity logic
        available_reverse = reverse_inventory + returns
        sorted_qty = np.minimum(available_reverse, params["sort_capacity"])
        processed_qty = np.minimum(sorted_qty, params["process_capacity"])

        recovered_units = recovery_yield * processed_qty
        unrecovered_units = np.maximum(0, processed_qty - recovered_units)

        reverse_inventory = available_reverse - sorted_qty

        # Material calculations
        returned_mass = returns * params["mass_per_unit"]
        recovered_mass = recovered_units * params["mass_per_unit"]

        if params["can_substitute_new_production"]:
            recovered_units_used = np.minimum(
                recovered_units * params["reintegration_rate"],
                D
            )
            new_units = np.maximum(0, D - recovered_units_used)
            virgin_material_units = new_units
        else:
            recovered_units_used = recovered_units * params["reintegration_rate"]
            new_units = np.zeros(n_iter)
            virgin_material_units = np.zeros(n_iter)

        recovered_mass_used = recovered_units_used * params["mass_per_unit"]
        virgin_material_demand_mass = D * params["mass_per_unit"]

        # Cost components
        collection_cost = returns * params["collection_cost"]
        holding_cost = (
            reverse_inventory + np.maximum(forward_inventory, 0)
        ) * params["holding_cost"]
        sort_test_cost = sorted_qty * params["sort_test_cost"]
        transport_cost = (
            returns + processed_qty + unrecovered_units
        ) * params["transport_cost"]
        processing_cost = processed_qty * params["processing_cost"]
        disposal_cost = unrecovered_units * params["disposal_cost"]
        manufacturing_cost = new_units * params["new_mfg_cost"]
        virgin_material_cost = (
            virgin_material_units * params["virgin_material_cost_per_unit"]
        )
        secondary_market_credit = recovered_units * params["secondary_market_credit"]

        period_cost = (
            collection_cost
            + holding_cost
            + sort_test_cost
            + transport_cost
            + processing_cost
            + disposal_cost
            + manufacturing_cost
            + virgin_material_cost
            - secondary_market_credit
        )

        total_cost += period_cost

        # Circularity metrics
        mrr_num += recovered_mass
        mrr_den += returned_mass

        vmd_num += recovered_mass_used
        vmd_den += virgin_material_demand_mass

        co2_avoided += (
            (params["ef_virgin"] - params["ef_recovery"]) * recovered_mass_used
            - params["ef_reverse_per_return"] * returns
        )

        # Forward inventory balance
        forward_inventory = np.maximum(
            0,
            forward_inventory + new_units + recovered_units_used - D
        )

    mrr = np.divide(
        mrr_num,
        mrr_den,
        out=np.zeros(n_iter),
        where=mrr_den > 0
    )

    vmd = np.divide(
        vmd_num,
        vmd_den,
        out=np.zeros(n_iter),
        where=vmd_den > 0
    )

    return {
        "TC": total_cost,
        "MRR": mrr,
        "VMD": vmd,
        "CO2_Avoided": co2_avoided,
    }


def run_monte_carlo(
    sector_params,
    n_iter=N_ITER,
    periods=PERIODS,
    seed=RANDOM_SEED
):
    """
    Run Monte Carlo simulation for all sectors.
    """
    master_rng = np.random.default_rng(seed)

    summary_rows = []
    raw_rows = []
    draws = {}

    for sector, params in sector_params.items():
        sector_seed = int(master_rng.integers(0, 2**32 - 1))
        rng = np.random.default_rng(sector_seed)

        # Common demand for high and low scenarios
        # This isolates the effect of return intensity.
        demand = rng.integers(
            params["demand_min"],
            params["demand_max"] + 1,
            size=(n_iter, periods)
        )

        high = simulate_condition(params, demand, params["rho_high"], rng)
        low = simulate_condition(params, demand, params["rho_low"], rng)

        rcpi = (low["TC"] - high["TC"]) / low["TC"]

        row = {
            "Sector": sector,
            "Mean_TC_High": float(np.mean(high["TC"])),
            "Mean_TC_Low": float(np.mean(low["TC"])),
            **summarize_array(rcpi, "RCPI"),
            "P_RCPI_Positive": float(np.mean(rcpi > 0)),
            "Mean_MRR": float(np.mean(high["MRR"])),
            "MRR_CI_2.5": float(np.percentile(high["MRR"], 2.5)),
            "MRR_CI_97.5": float(np.percentile(high["MRR"], 97.5)),
            "Mean_VMD": float(np.mean(high["VMD"])),
            "VMD_CI_2.5": float(np.percentile(high["VMD"], 2.5)),
            "VMD_CI_97.5": float(np.percentile(high["VMD"], 97.5)),
            "Mean_CO2_Avoided": float(np.mean(high["CO2_Avoided"])),
            "CO2_CI_2.5": float(np.percentile(high["CO2_Avoided"], 2.5)),
            "CO2_CI_97.5": float(np.percentile(high["CO2_Avoided"], 97.5)),
        }

        summary_rows.append(row)

        draws[sector] = {
            "TC_High": high["TC"],
            "TC_Low": low["TC"],
            "RCPI": rcpi,
            "MRR": high["MRR"],
            "VMD": high["VMD"],
            "CO2_Avoided": high["CO2_Avoided"],
        }

        for i in range(n_iter):
            raw_rows.append({
                "Sector": sector,
                "Iteration": i + 1,
                "TC_High": high["TC"][i],
                "TC_Low": low["TC"][i],
                "RCPI": rcpi[i],
                "MRR": high["MRR"][i],
                "VMD": high["VMD"][i],
                "CO2_Avoided": high["CO2_Avoided"][i],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df["Typology"] = summary_df.apply(classify_typology, axis=1)

    raw_df = pd.DataFrame(raw_rows)

    return summary_df, raw_df, draws


# ============================================================
# 5. Sensitivity analysis
# ============================================================

def run_sensitivity_analysis(
    sector_params,
    baseline_summary,
    n_iter=2_000,
    periods=PERIODS,
    seed=RANDOM_SEED,
    perturbation=0.20
):
    """
    One-at-a-time sensitivity analysis.
    Each parameter is increased and decreased by 20%.
    """
    sensitivity_targets = [
        "rho_high",
        "recovery_yield_mean",
        "sort_test_cost",
        "transport_cost",
        "processing_cost",
        "disposal_cost",
        "new_mfg_cost",
        "virgin_material_cost_per_unit",
        "process_capacity",
        "ef_reverse_per_return",
    ]

    rows = []

    baseline_lookup = baseline_summary.set_index("Sector")

    for sector in sector_params.keys():
        base_rcpi = baseline_lookup.loc[sector, "Mean_RCPI"]
        base_mrr = baseline_lookup.loc[sector, "Mean_MRR"]
        base_vmd = baseline_lookup.loc[sector, "Mean_VMD"]
        base_co2 = baseline_lookup.loc[sector, "Mean_CO2_Avoided"]

        for target in sensitivity_targets:
            for direction, factor in [("minus_20_percent", 1 - perturbation),
                                      ("plus_20_percent", 1 + perturbation)]:

                modified_params = copy.deepcopy(sector_params)

                if target == "recovery_yield_mean":
                    modified_params[sector] = adjust_beta_mean(
                        modified_params[sector],
                        factor
                    )
                else:
                    modified_params[sector][target] *= factor

                    if target in ["rho_high"]:
                        modified_params[sector][target] = float(
                            np.clip(modified_params[sector][target], 0.001, 0.999)
                        )

                    if target in ["process_capacity"]:
                        modified_params[sector][target] = max(
                            1,
                            int(round(modified_params[sector][target]))
                        )

                temp_summary, _, _ = run_monte_carlo(
                    modified_params,
                    n_iter=n_iter,
                    periods=periods,
                    seed=seed
                )

                sector_result = temp_summary[temp_summary["Sector"] == sector].iloc[0]

                rows.append({
                    "Sector": sector,
                    "Parameter": target,
                    "Direction": direction,
                    "Mean_RCPI": sector_result["Mean_RCPI"],
                    "Delta_RCPI": sector_result["Mean_RCPI"] - base_rcpi,
                    "Mean_MRR": sector_result["Mean_MRR"],
                    "Delta_MRR": sector_result["Mean_MRR"] - base_mrr,
                    "Mean_VMD": sector_result["Mean_VMD"],
                    "Delta_VMD": sector_result["Mean_VMD"] - base_vmd,
                    "Mean_CO2_Avoided": sector_result["Mean_CO2_Avoided"],
                    "Delta_CO2_Avoided": (
                        sector_result["Mean_CO2_Avoided"] - base_co2
                    ),
                    "Typology": sector_result["Typology"],
                })

    return pd.DataFrame(rows)


# ============================================================
# 6. Plotting functions
# ============================================================

def plot_rcpi_boxplot(raw_df):
    plt.figure(figsize=(11, 6))

    sectors = raw_df["Sector"].unique()
    data = [
        raw_df.loc[raw_df["Sector"] == sector, "RCPI"].values
        for sector in sectors
    ]

    plt.boxplot(data, tick_labels=sectors, showmeans=True)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("RCPI")
    plt.title("Distribution of RCPI across sectors")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_rcpi_boxplot.png", dpi=300)
    plt.close()


def plot_mean_rcpi(summary_df):
    plt.figure(figsize=(11, 6))

    x = np.arange(len(summary_df))
    means = summary_df["Mean_RCPI"].values
    lower = means - summary_df["RCPI_CI_2.5"].values
    upper = summary_df["RCPI_CI_97.5"].values - means

    plt.bar(x, means)
    plt.errorbar(x, means, yerr=[lower, upper], fmt="none", capsize=5)
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xticks(x, summary_df["Sector"], rotation=30, ha="right")
    plt.ylabel("Mean RCPI")
    plt.title("Mean RCPI with 95% confidence intervals")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_mean_rcpi_ci.png", dpi=300)
    plt.close()


def plot_circularity(summary_df):
    circularity_df = summary_df[["Sector", "Mean_MRR", "Mean_VMD"]].set_index("Sector")

    ax = circularity_df.plot(kind="bar", figsize=(11, 6))
    ax.set_ylabel("Indicator value")
    ax.set_title("Circularity performance: MRR and VMD")
    ax.legend(["Material Recovery Rate", "Virgin Material Displacement"])
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_mrr_vmd.png", dpi=300)
    plt.close()


def plot_co2(summary_df):
    plt.figure(figsize=(11, 6))

    x = np.arange(len(summary_df))
    means = summary_df["Mean_CO2_Avoided"].values
    lower = means - summary_df["CO2_CI_2.5"].values
    upper = summary_df["CO2_CI_97.5"].values - means

    plt.bar(x, means)
    plt.errorbar(x, means, yerr=[lower, upper], fmt="none", capsize=5)
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xticks(x, summary_df["Sector"], rotation=30, ha="right")
    plt.ylabel("CO2 equivalent avoided")
    plt.title("Mean CO2 equivalent avoided with 95% confidence intervals")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_co2_avoided.png", dpi=300)
    plt.close()


def plot_sensitivity_tornado(sensitivity_df, sector):
    temp = sensitivity_df[sensitivity_df["Sector"] == sector].copy()

    temp["Abs_Delta_RCPI"] = temp["Delta_RCPI"].abs()
    temp = temp.sort_values("Abs_Delta_RCPI", ascending=True)

    labels = temp["Parameter"] + " " + temp["Direction"]

    plt.figure(figsize=(10, 7))
    plt.barh(labels, temp["Delta_RCPI"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Change in Mean RCPI")
    plt.title(f"Sensitivity analysis for {sector}")
    plt.tight_layout()

    safe_name = sector.lower().replace(" ", "_").replace("/", "_")
    plt.savefig(OUTPUT_DIR / f"figure_sensitivity_{safe_name}.png", dpi=300)
    plt.close()


# ============================================================
# 7. Run everything
# ============================================================

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    summary_df, raw_df, draws = run_monte_carlo(
        SECTOR_PARAMS,
        n_iter=N_ITER,
        periods=PERIODS,
        seed=RANDOM_SEED
    )

    sensitivity_df = run_sensitivity_analysis(
        SECTOR_PARAMS,
        baseline_summary=summary_df,
        n_iter=2_000,
        periods=PERIODS,
        seed=RANDOM_SEED,
        perturbation=0.20
    )

    # Save outputs
    summary_df.to_csv(OUTPUT_DIR / "monte_carlo_summary.csv", index=False)
    raw_df.to_csv(OUTPUT_DIR / "monte_carlo_raw_draws.csv", index=False)
    sensitivity_df.to_csv(OUTPUT_DIR / "sensitivity_results.csv", index=False)

    # Save figures
    plot_rcpi_boxplot(raw_df)
    plot_mean_rcpi(summary_df)
    plot_circularity(summary_df)
    plot_co2(summary_df)

    for sector_name in SECTOR_PARAMS.keys():
        plot_sensitivity_tornado(sensitivity_df, sector_name)

    # Print main results
    print("\nMONTE CARLO SUMMARY")
    print(summary_df.to_string(index=False))

    print("\nSENSITIVITY ANALYSIS SUMMARY")
    print(
        sensitivity_df.sort_values(["Sector", "Parameter", "Direction"])
        .to_string(index=False)
    )

    print(f"\nFiles saved in: {OUTPUT_DIR.resolve()}")
