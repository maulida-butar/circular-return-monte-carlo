"""Revised circular-return model 2.0.0-rc1. See MODEL_NOTES.md for assumptions."""
import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import beta, binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_VERSION = "2.0.0-rc1"
N_ITER = 10_000
PERIODS = 9
RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "monte_carlo_outputs"

# Numerical archetype inputs are retained from v1.0.
# can_substitute_new_production is retained as the legacy name of the flag
# selecting inclusion of forward production/procurement COSTS.
# ef_virgin: kg CO2e/kg virgin material; ef_recovery: kg CO2e/kg INPUT processed.
# These emission units and the sale of surplus recovered output are explicit
# scenario assumptions, not source-validated empirical estimates.

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

# ---------- Revised accounting and simulation implementation ----------

COST_COMPONENTS = ('Collection', 'Holding', 'Sort_Test', 'Transport',
                   'Processing', 'Disposal', 'Manufacturing', 'Virgin_Material',
                   'External_Sales_Credit')
SENSITIVITY_TARGETS = ('rho_high', 'recovery_yield_mean', 'sort_test_cost',
                       'transport_cost', 'processing_cost', 'disposal_cost',
                       'new_mfg_cost', 'virgin_material_cost_per_unit',
                       'process_capacity', 'ef_reverse_per_return')


def summarize_array(values, prefix):
    """Outcome percentiles describe simulation variation, not a mean CI."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {f'{name}_{prefix}': np.nan for name in ('Mean', 'Median', 'SD')} | {
            f'{prefix}_PI_2.5': np.nan, f'{prefix}_PI_97.5': np.nan,
            f'{prefix}_MCSE_Mean': np.nan, f'{prefix}_Valid_N': 0}
    sd = float(np.std(x, ddof=1)) if x.size > 1 else np.nan
    return {f'Mean_{prefix}': float(np.mean(x)),
            f'Median_{prefix}': float(np.median(x)), f'SD_{prefix}': sd,
            f'{prefix}_PI_2.5': float(np.quantile(x, .025)),
            f'{prefix}_PI_97.5': float(np.quantile(x, .975)),
            f'{prefix}_MCSE_Mean': sd / np.sqrt(x.size),
            f'{prefix}_Valid_N': int(x.size)}


def classify_typology(row):
    """Sequential rules of manuscript Table 5, conditional on defined RCPI."""
    if row.get('RCPI_Invalid_N', 0) > 0:
        return 'Not assessable'
    mu, lo, hi = (row['Mean_RCPI'], row['RCPI_PI_2.5'], row['RCPI_PI_97.5'])
    p, mrr, vmd, co2 = (row['P_RCPI_Positive'], row['Mean_MRR'],
                         row['Mean_VMD'], row['Mean_CO2_Avoided'])
    if not np.isfinite([mu, lo, hi, p, mrr, vmd, co2]).all():
        return 'Not assessable'
    if mu > .10 and lo > 0 and p >= .95 and mrr >= .50 and vmd >= .30 and co2 > 0:
        return 'Sunny'
    if mu < 0 and hi < 0 and p < .40 and (vmd < .10 or co2 <= 0):
        return 'Rainy'
    if mu > 0 and p >= .60 and (mrr >= .50 or vmd >= .30 or co2 > 0):
        return 'Cloudy'
    return 'Foggy'


def validate_params(p):
    if not 0 <= p['rho_low'] <= p['rho_high'] <= 1:
        raise ValueError('Return probabilities must satisfy 0 <= low <= high <= 1.')
    if not 0 <= p['reintegration_rate'] <= 1:
        raise ValueError('Reintegration rate must be between zero and one.')
    if not 0 <= p['demand_min'] <= p['demand_max']:
        raise ValueError('Invalid demand bounds.')
    for key in ('demand_min', 'demand_max', 'sort_capacity', 'process_capacity'):
        if p[key] != int(p[key]) or p[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer.')
    for key in ('beta_a', 'beta_b', 'mass_per_unit'):
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    for key, value in p.items():
        if isinstance(value, (int, float)) and (not np.isfinite(value) or value < 0):
            raise ValueError(f'Invalid parameter {key}.')


class SharedRandomInputs:
    """Independent base uniforms; inverse CDF coupling across counterfactuals.

    Named sector streams and replication-major arrays preserve prefixes when N
    changes. Cost/capacity changes cannot change demand, return, or yield draws.
    """
    def __init__(self, sector, n_iter, periods, seed):
        if n_iter < 2 or periods < 1 or seed < 0:
            raise ValueError('Require N >= 2, periods >= 1, and seed >= 0.')
        name_seed = int.from_bytes(hashlib.sha256(sector.encode()).digest()[:8], 'big')
        uniforms = []
        for stream in range(3):
            rng = np.random.default_rng(np.random.SeedSequence([seed, name_seed, stream]))
            uniforms.append(np.clip(rng.random((n_iter, periods)),
                                    np.finfo(float).eps, 1 - np.finfo(float).eps))
        self.demand_u, self.return_u, self.yield_u = uniforms
        self.demand_cache, self.return_cache, self.yield_cache = {}, {}, {}

    def sample(self, p, rho):
        bounds = (p['demand_min'], p['demand_max'])
        if bounds not in self.demand_cache:
            self.demand_cache[bounds] = (bounds[0] + np.floor(
                self.demand_u * (bounds[1] - bounds[0] + 1))).astype(int)
        demand = self.demand_cache[bounds]
        return_key = (*bounds, rho)
        if return_key not in self.return_cache:
            self.return_cache[return_key] = binom.ppf(self.return_u, demand, rho)
        yield_key = (p['beta_a'], p['beta_b'])
        if yield_key not in self.yield_cache:
            self.yield_cache[yield_key] = beta.ppf(self.yield_u, *yield_key)
        return demand, self.return_cache[return_key], self.yield_cache[yield_key]


def simulate_condition(params, demand, rho, rng=None, *, returns=None,
                       recovery_yields=None):
    """Finite-horizon flows, with two queues and explicit recovery destinations.

    kg/unit converts all product flows to mass. ef_virgin is kg CO2e/kg
    virgin material displaced; ef_recovery is kg CO2e/kg INPUT processed.
    Both are illustrative assumptions, not newly measured emission factors.
    Unsorted and sorted terminal stocks are carried, with no salvage value or
    post-horizon processing/disposal cost. Non-used recovered output is sold
    in the same period; its sale receives no virgin-displacement carbon credit.
    """
    validate_params(params)
    demand = np.asarray(demand, dtype=float)
    if demand.ndim != 2 or np.any(~np.isfinite(demand)) or np.any(demand < 0):
        raise ValueError('Demand must be a finite nonnegative N-by-T matrix.')
    if np.any(demand != np.floor(demand)) or not 0 <= rho <= 1:
        raise ValueError('Demand must be integral and rho must lie in [0, 1].')
    if (returns is None) != (recovery_yields is None):
        raise ValueError('Supply both return and yield trajectories, or neither.')
    if returns is None and rng is None:
        raise ValueError('Supply random input trajectories or a random generator.')
    if returns is not None:
        returns, recovery_yields = np.asarray(returns), np.asarray(recovery_yields)
        if returns.shape != demand.shape or recovery_yields.shape != demand.shape:
            raise ValueError('Input trajectory shapes must match demand.')
    n_iter, periods = demand.shape
    totals = {key: np.zeros(n_iter) for key in (
        'Returned_Units', 'Sorted_Units', 'Processed_Units', 'Recovered_Units',
        'Recovered_Used_Units', 'Sold_Units', 'Disposed_Units', 'New_Units',
        'Demand_Units', 'Avoided_Virgin_Emissions', 'Recovery_Emissions',
        'Reverse_Emissions')}
    costs = {key: np.zeros(n_iter) for key in COST_COMPONENTS}
    unsorted, sorted_stock = np.zeros(n_iter), np.zeros(n_iter)
    max_balance_error = np.zeros(n_iter)
    mass = params['mass_per_unit']
    for t in range(periods):
        D = demand[:, t]
        if returns is None:
            R = np.asarray(rng.binomial(D.astype(int), rho), dtype=float)
            Y = np.asarray(rng.beta(params['beta_a'], params['beta_b'], size=n_iter))
        else:
            R, Y = returns[:, t], recovery_yields[:, t]
        if (np.any(~np.isfinite(R)) or np.any(~np.isfinite(Y)) or np.any(R < 0)
                or np.any(R > D) or np.any(Y < 0) or np.any(Y > 1)):
            raise ValueError('Invalid return or recovery-yield trajectory.')
        previous_stock = unsorted + sorted_stock
        available_unsorted = unsorted + R
        q_sort = np.minimum(available_unsorted, params['sort_capacity'])
        unsorted = available_unsorted - q_sort
        available_sorted = sorted_stock + q_sort
        q_proc = np.minimum(available_sorted, params['process_capacity'])
        sorted_stock = available_sorted - q_proc
        q_rec = Y * q_proc
        q_dispose = q_proc - q_rec
        q_used = np.minimum(q_rec * params['reintegration_rate'], D)
        q_sold = q_rec - q_used
        # Reference demand is served for all archetypes. The legacy boolean
        # controls which forward costs are INCLUDED in the accounting boundary.
        q_new = D - q_used
        residual = previous_stock + R - unsorted - sorted_stock - q_used - q_sold - q_dispose
        max_balance_error = np.maximum(max_balance_error, np.abs(residual))
        if np.any(np.abs(residual) > 1e-8):
            raise ArithmeticError('Period material balance failed.')
        flows = {'Returned_Units': R, 'Sorted_Units': q_sort,
                 'Processed_Units': q_proc, 'Recovered_Units': q_rec,
                 'Recovered_Used_Units': q_used, 'Sold_Units': q_sold,
                 'Disposed_Units': q_dispose, 'New_Units': q_new, 'Demand_Units': D,
                 'Avoided_Virgin_Emissions': params['ef_virgin'] * q_used * mass,
                 'Recovery_Emissions': params['ef_recovery'] * q_proc * mass,
                 'Reverse_Emissions': params['ef_reverse_per_return'] * R}
        for key, value in flows.items():
            totals[key] += value
        costs['Collection'] += R * params['collection_cost']
        costs['Holding'] += (unsorted + sorted_stock) * params['holding_cost']
        costs['Sort_Test'] += q_sort * params['sort_test_cost']
        costs['Transport'] += (R + q_proc + q_dispose) * params['transport_cost']
        costs['Processing'] += q_proc * params['processing_cost']
        costs['Disposal'] += q_dispose * params['disposal_cost']
        if params['can_substitute_new_production']:
            costs['Manufacturing'] += q_new * params['new_mfg_cost']
            costs['Virgin_Material'] += q_new * params['virgin_material_cost_per_unit']
        costs['External_Sales_Credit'] += q_sold * params['secondary_market_credit']
    total_cost = sum(costs[k] for k in COST_COMPONENTS[:-1]) - costs['External_Sales_Credit']
    def ratio(numerator, denominator):
        return np.divide(numerator, denominator, out=np.full(n_iter, np.nan), where=denominator > 0)
    final_residual = (totals['Returned_Units'] - totals['Recovered_Used_Units']
                      - totals['Sold_Units'] - totals['Disposed_Units'] - unsorted - sorted_stock)
    if np.max(np.abs(final_residual)) > 1e-8:
        raise ArithmeticError('Horizon material balance failed.')
    carbon = totals['Avoided_Virgin_Emissions'] - totals['Recovery_Emissions'] - totals['Reverse_Emissions']
    return dict(totals, **{f'Cost_{k}': v for k, v in costs.items()}, TC=total_cost,
                MRR=ratio(totals['Recovered_Units'], totals['Returned_Units']),
                VMD=ratio(totals['Recovered_Used_Units'], totals['Demand_Units']),
                CO2_Avoided=carbon,
                CO2_per_kg_Demand=ratio(carbon, totals['Demand_Units'] * mass),
                Terminal_Unsorted_Units=unsorted, Terminal_Sorted_Units=sorted_stock,
                Material_Balance_Error=np.maximum(max_balance_error, np.abs(final_residual)),
                Demand_Balance_Error=np.abs(totals['Demand_Units'] - totals['New_Units']
                                            - totals['Recovered_Used_Units']))


def evaluate_sector(sector, p, random_inputs):
    validate_params(p)
    conditions = []
    for rho in (p['rho_high'], p['rho_low']):
        D, R, Y = random_inputs.sample(p, rho)
        conditions.append(simulate_condition(p, D, rho, returns=R, recovery_yields=Y))
    high, low = conditions
    n_iter = len(high['TC'])
    valid = (low['TC'] > 0) & np.isfinite(low['TC']) & np.isfinite(high['TC'])
    rcpi = np.divide(low['TC'] - high['TC'], low['TC'],
                     out=np.full(n_iter, np.nan), where=valid)
    row = {'Sector': sector, 'N_Replications': n_iter,
           'Cost_Boundary': ('forward_and_recovery' if p['can_substitute_new_production']
                             else 'recovery_operations_only'),
           'Mean_TC_High': float(high['TC'].mean()), 'Mean_TC_Low': float(low['TC'].mean()),
           **summarize_array(rcpi, 'RCPI'),
           'RCPI_Invalid_N': int((~valid).sum()),
           'P_RCPI_Positive': float(np.mean(rcpi[valid] > 0)) if valid.any() else np.nan}
    row.update(summarize_array(high['TC'] - low['TC'], 'Delta_TC'))
    row.update(summarize_array(high['CO2_Avoided'] - low['CO2_Avoided'], 'Delta_CO2_Avoided'))
    for label, result in [('High', high), ('Low', low)]:
        for indicator in ('MRR', 'VMD', 'CO2_Avoided', 'CO2_per_kg_Demand'):
            prefix = indicator if label == 'High' else f'{indicator}_Low'
            row.update(summarize_array(result[indicator], prefix))
        for key in ('Returned_Units', 'Processed_Units', 'Recovered_Used_Units',
                    'Sold_Units', 'Disposed_Units', 'Terminal_Unsorted_Units',
                    'Terminal_Sorted_Units'):
            row[f'Mean_{key}_{label}'] = float(result[key].mean())
        row[f'Max_Material_Balance_Error_{label}'] = float(result['Material_Balance_Error'].max())
        row[f'Max_Demand_Balance_Error_{label}'] = float(result['Demand_Balance_Error'].max())
    row['Typology'] = classify_typology(row)
    raw = {'Sector': np.repeat(sector, n_iter), 'Iteration': np.arange(1, n_iter + 1),
           'TC_High': high['TC'], 'TC_Low': low['TC'], 'RCPI': rcpi,
           'Delta_TC': high['TC'] - low['TC'],
           'Delta_CO2_Avoided': high['CO2_Avoided'] - low['CO2_Avoided']}
    for label, result in [('High', high), ('Low', low)]:
        for key, value in result.items():
            if key == 'TC' or key.startswith('Cost_'):
                continue
            raw[key if label == 'High' and key in ('MRR', 'VMD', 'CO2_Avoided')
                else f'{key}_{label}'] = value
    return row, raw, {'High': high, 'Low': low, 'RCPI': rcpi}


def run_monte_carlo(sector_params, n_iter=N_ITER, periods=PERIODS, seed=RANDOM_SEED):
    rows, frames, draws = [], [], {}
    for sector, p in sector_params.items():
        row, raw, results = evaluate_sector(sector, p, SharedRandomInputs(sector, n_iter, periods, seed))
        rows.append(row)
        frames.append(pd.DataFrame(raw))
        draws[sector] = results
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True), draws


def adjust_beta_mean(params, factor):
    concentration = params['beta_a'] + params['beta_b']
    mean = np.clip(params['beta_a'] / concentration * factor, 1e-6, 1 - 1e-6)
    params['beta_a'], params['beta_b'] = mean * concentration, (1 - mean) * concentration
    return params


def run_sensitivity_analysis(sector_params, baseline_summary=None, n_iter=N_ITER,
                             periods=PERIODS, seed=RANDOM_SEED, perturbation=.20):
    """OAT effects use matched baseline draws; an old summary is never subtracted.

    baseline_summary is accepted for compatibility. The reference is recomputed
    at n_iter using the same random inputs as each perturbation. All mean deltas
    and MCSEs refer to paired replication differences, not outcome intervals.
    """
    if not 0 < perturbation < 1:
        raise ValueError('Perturbation must lie between zero and one.')
    rows = []
    indicators = ('RCPI', 'MRR', 'VMD', 'CO2_Avoided')
    for sector, original in sector_params.items():
        inputs = SharedRandomInputs(sector, n_iter, periods, seed)
        base, base_raw, _ = evaluate_sector(sector, original, inputs)
        for target in SENSITIVITY_TARGETS:
            original_value = (original['beta_a'] / (original['beta_a'] + original['beta_b'])
                              if target == 'recovery_yield_mean' else original[target])
            for direction, factor in [(f'minus_{100 * perturbation:g}_percent', 1 - perturbation),
                                       (f'plus_{100 * perturbation:g}_percent', 1 + perturbation)]:
                p = copy.deepcopy(original)
                if target == 'recovery_yield_mean':
                    adjust_beta_mean(p, factor)
                    actual_value = p['beta_a'] / (p['beta_a'] + p['beta_b'])
                else:
                    actual_value = original_value * factor
                    if target == 'rho_high':
                        actual_value = float(np.clip(actual_value, p['rho_low'], 1.0))
                    if target == 'process_capacity':
                        actual_value = max(0, int(round(actual_value)))
                    p[target] = actual_value
                summary, raw, _ = evaluate_sector(sector, p, inputs)
                changed = bool(not np.isclose(actual_value, original_value, atol=0, rtol=1e-12))
                row = {'Sector': sector, 'Parameter': target, 'Direction': direction,
                       'Baseline_Value': original_value, 'Perturbed_Value': actual_value,
                       'Requested_Change_Percent': (factor - 1) * 100,
                       'Actual_Change_Percent': ((actual_value / original_value - 1) * 100
                                                 if original_value != 0 else np.nan),
                       'Parameter_Changed': changed,
                       'Clipped_Or_Rounded': not np.isclose(actual_value, original_value * factor),
                       'N_Replications': n_iter, 'Baseline_Typology': base['Typology'],
                       'Typology': summary['Typology'],
                       'P_RCPI_Positive': summary['P_RCPI_Positive'],
                       'RCPI_PI_2.5': summary['RCPI_PI_2.5'],
                       'RCPI_PI_97.5': summary['RCPI_PI_97.5']}
                for indicator in indicators:
                    row[f'Baseline_Mean_{indicator}'] = base[f'Mean_{indicator}']
                    row[f'Mean_{indicator}'] = summary[f'Mean_{indicator}']
                    difference = np.asarray(raw[indicator]) - np.asarray(base_raw[indicator])
                    stats = summarize_array(difference, 'difference')
                    row[f'Delta_{indicator}'] = stats['Mean_difference']
                    row[f'Delta_{indicator}_MCSE'] = stats['difference_MCSE_Mean']
                    row[f'Delta_{indicator}_Valid_N'] = stats['difference_Valid_N']
                rows.append(row)
    return pd.DataFrame(rows)


# ---------- Figures and reproducibility files ----------

SHORT_NAMES = {'Aircraft manufacturing': 'Aircraft', 'Telecommunications': 'Telecom',
               'Computer hardware refurbishment': 'Computer hardware',
               'General retail': 'Retail', 'Carpet manufacturing': 'Carpet'}


def save_figure(fig, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f'{stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'{stem}.svg', bbox_inches='tight')
    plt.close(fig)


def plot_sensitivity_tornado(sensitivity_df, sector, output_dir=OUTPUT_DIR):
    """All ten analyzed parameters, paired effects and actual clipped changes."""
    labels = {'rho_high': 'Return probability', 'recovery_yield_mean': 'Recovery yield',
              'sort_test_cost': 'Sorting and testing cost', 'transport_cost': 'Transport cost',
              'processing_cost': 'Processing cost', 'disposal_cost': 'Disposal cost',
              'new_mfg_cost': 'New manufacturing cost',
              'virgin_material_cost_per_unit': 'Virgin material cost',
              'process_capacity': 'Processing capacity',
              'ef_reverse_per_return': 'Reverse logistics emission factor'}
    temp = sensitivity_df[sensitivity_df.Sector == sector].copy()
    minus = temp[temp.Direction.str.startswith('minus_')].set_index('Parameter')
    plus = temp[temp.Direction.str.startswith('plus_')].set_index('Parameter')
    if minus.empty or set(minus.index) != set(plus.index):
        raise ValueError(f'Incomplete sensitivity pairs for {sector}.')
    order = pd.concat([minus.Delta_RCPI.abs(), plus.Delta_RCPI.abs()], axis=1).max(axis=1).sort_values().index
    minus, plus = minus.loc[order], plus.loc[order]
    names = []
    for parameter in order:
        name = labels.get(parameter, parameter)
        m, p = minus.loc[parameter], plus.loc[parameter]
        if not m.Parameter_Changed and not p.Parameter_Changed:
            name += ' (unchanged at zero)'
        elif m.Clipped_Or_Rounded or p.Clipped_Or_Rounded:
            name += f' ({m.Actual_Change_Percent:+.1f}% / {p.Actual_Change_Percent:+.1f}%)'
        names.append(name)
    fig, ax = plt.subplots(figsize=(11, 6.8))
    y = np.arange(len(order))
    for frame, shift, color, label in [(minus, -.18, '#c46842', 'Nominal -20%'),
                                       (plus, .18, '#217c86', 'Nominal +20%')]:
        ax.barh(y + shift, frame.Delta_RCPI, height=.34, color=color, label=label)
        ax.errorbar(frame.Delta_RCPI, y + shift,
                    xerr=1.96 * frame.Delta_RCPI_MCSE.fillna(0),
                    fmt='none', ecolor='#333333', linewidth=.7, capsize=2)
    ax.set_yticks(y, names)
    ax.axvline(0, color='#444444', linewidth=.8)
    ax.set_xlabel('Paired change in mean RCPI')
    ax.set_title(f'One-at-a-time sensitivity of RCPI\n{sector}')
    ax.legend(loc='best', frameon=False)
    fig.text(.5, .012, 'Whiskers: ±1.96 Monte Carlo SE of the paired mean change; not outcome intervals.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    stem = ('figure7_rcpi_sensitivity_computer_hardware' if sector == 'Computer hardware refurbishment'
            else 'figure_sensitivity_' + sector.lower().replace(' ', '_'))
    save_figure(fig, output_dir, stem)


def plot_results(summary, raw, sensitivity, output_dir):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    groups = [('forward_and_recovery', 'Forward and recovery costs'),
              ('recovery_operations_only', 'Recovery costs only')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, (boundary, title) in zip(axes, groups):
        sectors = summary.loc[summary.Cost_Boundary == boundary, 'Sector'].tolist()
        values = [raw.loc[raw.Sector == s, 'RCPI'].dropna().to_numpy() for s in sectors]
        ax.boxplot(values, tick_labels=[SHORT_NAMES.get(s, s) for s in sectors], showmeans=True,
                   flierprops={'markersize': 2, 'alpha': .3})
        ax.axhline(0, linestyle='--', color='#555555', linewidth=.8)
        ax.set_title(title)
        ax.set_ylabel('RCPI')
        ax.tick_params(axis='x', labelrotation=15)
    fig.suptitle('Distribution of RCPI across simulated archetypes')
    fig.tight_layout()
    save_figure(fig, output_dir, 'figure_rcpi_boxplot')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, (boundary, title) in zip(axes, groups):
        part = summary[summary.Cost_Boundary == boundary]
        x = np.arange(len(part))
        ax.bar(x, part.Mean_RCPI, color='#217c86', width=.6)
        ax.vlines(x, part['RCPI_PI_2.5'], part['RCPI_PI_97.5'], color='#333333')
        ax.scatter(x, part['RCPI_PI_2.5'], marker='_', color='#333333')
        ax.scatter(x, part['RCPI_PI_97.5'], marker='_', color='#333333')
        ax.set_xticks(x, [SHORT_NAMES.get(s, s) for s in part.Sector], rotation=15)
        ax.axhline(0, linestyle='--', color='#555555', linewidth=.8)
        ax.set_title(title)
        ax.set_ylabel('Mean RCPI')
    fig.suptitle('Mean RCPI and 95% simulation percentile intervals')
    fig.tight_layout()
    save_figure(fig, output_dir, 'figure_mean_rcpi_pi')

    x = np.arange(len(summary))
    names = [SHORT_NAMES.get(s, s) for s in summary.Sector]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - .18, summary.Mean_MRR, width=.35, label='MRR', color='#217c86')
    ax.bar(x + .18, summary.Mean_VMD, width=.35, label='VMD', color='#c46842')
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction')
    ax.set_title('Material recovery and virgin material displacement: high-return scenario')
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, 'figure_mrr_vmd')

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, summary.Mean_CO2_Avoided, color='#217c86', width=.6)
    ax.vlines(x, summary['CO2_Avoided_PI_2.5'], summary['CO2_Avoided_PI_97.5'], color='#333333')
    ax.scatter(x, summary['CO2_Avoided_PI_2.5'], marker='_', color='#333333')
    ax.scatter(x, summary['CO2_Avoided_PI_97.5'], marker='_', color='#333333')
    ax.axhline(0, linestyle='--', color='#555555', linewidth=.8)
    ax.set_xticks(x, names)
    ax.set_ylabel('kg CO₂e avoided per modeled horizon')
    ax.set_title('High-return carbon outcome and 95% simulation percentile intervals')
    fig.tight_layout()
    save_figure(fig, output_dir, 'figure_co2_avoided')
    for sector in summary.Sector:
        plot_sensitivity_tornado(sensitivity, sector, output_dir)


def write_outputs(summary, raw, sensitivity, draws, output_dir, settings, legacy_summary=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / 'monte_carlo_summary.csv', index=False)
    raw.to_csv(out / 'monte_carlo_raw_draws.csv', index=False)
    sensitivity.to_csv(out / 'sensitivity_results.csv', index=False)
    audit, cost_rows = [], []
    for sector, results in draws.items():
        for scenario in ('High', 'Low'):
            data = results[scenario]
            audit.append({'Sector': sector, 'Scenario': scenario,
                          'Max_Material_Balance_Error_units': float(data['Material_Balance_Error'].max()),
                          'Max_Demand_Balance_Error_units': float(data['Demand_Balance_Error'].max()),
                          'Mean_Terminal_Unsorted_Units': float(data['Terminal_Unsorted_Units'].mean()),
                          'Mean_Terminal_Sorted_Units': float(data['Terminal_Sorted_Units'].mean()),
                          'Min_TC': float(data['TC'].min()),
                          'No_Return_Replications': int(np.sum(data['Returned_Units'] == 0))})
            cost_rows.append({'Sector': sector, 'Scenario': scenario,
                              **{key: float(data[f'Cost_{key}'].mean()) for key in COST_COMPONENTS},
                              'TC': float(data['TC'].mean())})
    pd.DataFrame(audit).to_csv(out / 'model_audit.csv', index=False)
    pd.DataFrame(cost_rows).to_csv(out / 'cost_components_summary.csv', index=False)
    (out / 'sector_parameters.json').write_text(json.dumps(SECTOR_PARAMS, indent=2) + '\n')
    units = {key: 'dimensionless' for key in next(iter(SECTOR_PARAMS.values()))}
    for key in ('demand_min', 'demand_max', 'sort_capacity', 'process_capacity'):
        units[key] = 'units/period'
    units.update(mass_per_unit='kg/unit', collection_cost='currency/returned unit',
                 holding_cost='currency/queued unit/period', sort_test_cost='currency/sorted unit',
                 transport_cost='currency/counted activity movement',
                 processing_cost='currency/processed unit', disposal_cost='currency/rejected unit',
                 new_mfg_cost='currency/new unit', virgin_material_cost_per_unit='currency/new unit',
                 secondary_market_credit='currency/externally sold recovered unit',
                 ef_virgin='kg CO2e/kg virgin material displaced',
                 ef_recovery='kg CO2e/kg INPUT processed',
                 ef_reverse_per_return='kg CO2e/returned unit',
                 can_substitute_new_production='boolean: forward COSTS included (legacy name)')
    pd.DataFrame([{'Parameter': key, 'Unit': unit, 'Source_Status': 'illustrative archetype assumption'}
                  for key, unit in units.items()]).to_csv(out / 'parameter_units.csv', index=False)
    metadata = dict(settings, model_version=MODEL_VERSION,
                    python_version=platform.python_version(), numpy_version=np.__version__,
                    pandas_version=pd.__version__, scipy_version=scipy.__version__,
                    matplotlib_version=matplotlib.__version__,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    rng='PCG64; independent named-sector demand/return/yield uniform streams',
                    pairing='common base uniforms across high/low and all sensitivity comparisons',
                    interval='central empirical 2.5th and 97.5th outcome percentiles; not mean CIs',
                    terminal_policy='carry unsorted and sorted stocks; zero salvage; no post-horizon costs',
                    recovered_surplus_policy='sell all non-internally-used recovered output in the same period',
                    emission_units=units['ef_virgin'] + '; recovery: ' + units['ef_recovery'],
                    calibration_status='illustrative input values; emission units and sale assumptions require empirical validation',
                    cost_boundary='forward costs excluded for telecommunications and retail; no imputed prices',
                    typology='manuscript Table 5; Not assessable marks undefined statistics outside the typology',
                    sensitivity_reference='recomputed at sensitivity_iterations with shared random inputs',
                    baseline_summary_argument='not used as a sensitivity subtraction reference',
                    probability_clipping='[rho_low, 1]; actual values exported',
                    zero_returns='MRR undefined (NaN); valid counts reported',
                    nonpositive_benchmark='RCPI undefined (NaN); invalid counts reported; no classification')
    (out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if legacy_summary and Path(legacy_summary).is_file():
        old = pd.read_csv(legacy_summary)
        cols = ['Sector', 'Mean_RCPI', 'Mean_MRR', 'Mean_VMD', 'Mean_CO2_Avoided', 'Typology']
        comparison = old[cols].merge(summary[cols], on='Sector', suffixes=('_v1', '_v2'))
        comparison.to_csv(out / 'comparison_v1_v2.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=N_ITER)
    parser.add_argument('--sensitivity-iterations', type=int, default=None,
                        help='Defaults to baseline N; each sensitivity uses its own matched reference.')
    parser.add_argument('--periods', type=int, default=PERIODS)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--legacy-summary', type=Path,
                        default=Path(__file__).resolve().parent / 'reference_v1' / 'monte_carlo_summary.csv')
    parser.add_argument('--skip-plots', action='store_true')
    args = parser.parse_args()
    sensitivity_n = args.sensitivity_iterations or args.iterations
    print(f'{MODEL_VERSION}: baseline N={args.iterations}, sensitivity N={sensitivity_n}, T={args.periods}', flush=True)
    summary, raw, draws = run_monte_carlo(SECTOR_PARAMS, args.iterations, args.periods, args.seed)
    print('Baseline completed; running paired sensitivity scenarios.', flush=True)
    sensitivity = run_sensitivity_analysis(SECTOR_PARAMS, summary, sensitivity_n, args.periods, args.seed)
    settings = {'baseline_iterations': args.iterations, 'sensitivity_iterations': sensitivity_n,
                'periods': args.periods, 'seed': args.seed, 'nominal_perturbation': .20}
    write_outputs(summary, raw, sensitivity, draws, args.output_dir, settings, args.legacy_summary)
    if not args.skip_plots:
        plot_results(summary, raw, sensitivity, args.output_dir)
    print(summary[['Sector', 'Mean_RCPI', 'Mean_MRR', 'Mean_VMD', 'Mean_CO2_Avoided', 'Typology']].to_string(index=False))
    print(f'Outputs: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
