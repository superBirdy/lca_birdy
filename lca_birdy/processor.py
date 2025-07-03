import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class Flow:
    def __init__(self, name, amount, unit, flow_type, r,
                 description="", displacement_product="", displacement_ratio=0.0):
        self.name = name
        self.amount = amount
        self.unit = unit
        self.flow_type = flow_type
        self.r = r
        self.description = description.strip().lower()
        self.displacement_product = displacement_product.strip()
        self.displacement_ratio = float(displacement_ratio) if displacement_ratio else 0.0

class Process:
    """
    Represents a process with input and output flows.
    """
    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.outputs = []

    def add_flow(self, flow):
        if flow.flow_type == 'input':
            self.inputs.append(flow)
        elif flow.flow_type == 'output':
            self.outputs.append(flow)
        else:
            raise ValueError(f"Unknown flow type: {flow.flow_type}")

def read_process_data(filepath):
    """
    Reads process flow data from an Excel file.
    Expected columns: Scenario, Process Name, Flow Name, Type, Amount, Unit, r
    Converts 'gram' units to 'kg'.
    """
    df = pd.read_excel(filepath)
    required_cols = ['Process Name','Description', 'Flow Name', 'Type', 'Amount', 'Unit', 'r']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Excel file must contain columns: {required_cols}")

    processes = {}

    for _, row in df.iterrows():
        process_name = str(row['Process Name']).strip()

        if process_name not in processes:
            processes[process_name] = Process(process_name)

        flow_name = str(row['Flow Name']).strip()
        flow_type = str(row['Type']).strip().lower()
        amount = float(row['Amount'])
        raw_unit = str(row['Unit']).strip().lower()
        r_value = float(row['r'])
        description = str(row.get("Description", "")).strip().lower()
        displacement_product = str(row.get("Displacement Product", "")).strip()
        displacement_ratio = row.get("Displacement Ratio", 0)

        # ✅ Normalize units
        if raw_unit in ['g', 'gram', 'grams']:
            amount /= 1000
            unit = 'kg'
        else:
            unit = raw_unit

        flow = Flow(
            name=flow_name,
            amount=amount,
            unit=unit,
            flow_type=flow_type,
            r=r_value,
            description=description,
            displacement_product=displacement_product,
            displacement_ratio=displacement_ratio
        )

        processes[process_name].add_flow(flow)

    return list(processes.values())


# --- Read Emission Factors from Excel ---
def read_emission_inventory(filepath):
    df = pd.read_excel(filepath)
    impact_factors = {}
    for _, row in df.iterrows():
        flow_name = row['Flow Name']
        impact_factors[flow_name] = {
            'ghg': row['GHG'],
            'factor': row['Emission Factor'],
            'unit': row['Unit']
        }
    return impact_factors

def detailed_lcia_breakdown(processes, impact_factors):
    rows = []

    for p in processes:
        for f in p.inputs + p.outputs:
            factor_info = impact_factors.get(f.name, {'ghg': 'N/A', 'factor': 0, 'unit': f.unit,'r':f.r,"Description": f.description})
            ghg = factor_info['ghg']
            factor = factor_info['factor']
            impact = f.amount * factor*f.r
            rows.append({
                "Process Name": p.name,
                "Description": f.description,
                "Flow Name": f.name,
                "Flow Type": f.flow_type,
                "Amount": f.amount,
                "Unit": f.unit,
                "GHG": ghg,
                "Emission Factor": factor,
                "Recycle Factor": f.r,
                "Impact (kg CO2e)": impact,
                "Displacement Product": getattr(f, 'displacement_product', ''),
                "Displacement Ratio": getattr(f, 'displacement_ratio', 0.0)
            })

    return pd.DataFrame(rows)

def apply_displacement(df_base, impact_factors):
    """
    Applies displacement impact to an LCIA base table.
    Adds columns: Displacement Impact, Adjusted Impact.
    """
    displacement_impacts = []
    adjusted_impacts = []

    for idx, row in df_base.iterrows():
        dp_name = row.get("Displacement Product", "").strip()
        dp_ratio = row.get("Displacement Ratio", 0.0)
        amount = row["Amount"]

        # Compute displacement only if fields are set
        if dp_name and dp_ratio > 0:
            dp_info = impact_factors.get(dp_name)
            if dp_info:
                dp_factor = dp_info["factor"]
                displacement_impact = -1*amount * dp_ratio * dp_factor
            else:
                print(f"⚠️ Displacement product '{dp_name}' not found in emission factors.")
                displacement_impact = 0.0
        else:
            displacement_impact = 0.0

        base_impact = row["Impact (kg CO2e)"]
        adjusted_impact = base_impact + displacement_impact

        displacement_impacts.append(displacement_impact)
        adjusted_impacts.append(adjusted_impact)

    # Add columns to original DataFrame
    df_base["Displacement Impact"] = displacement_impacts
    df_base["Adjusted Impact (kg CO2e)"] = adjusted_impacts

    return df_base

def normalize_impact_per_product(df_lcia, processes, impact_column="Impact (kg CO2e)"):
    """
    Normalize total LCIA impact per process by product output amount (if available).

    Parameters:
    - df_lcia: DataFrame with LCIA results (must include 'Process Name' and impact column)
    - processes: list of Process objects with output flows
    - impact_column: column name to use for summing impact (default is base impact)

    Returns:
    - Dictionary: {process_name: normalized_impact_per_unit}
    """
    # Step 1: Normalize and group total impact by process name
    df_lcia["Process Name"] = df_lcia["Process Name"].str.strip().str.lower()
    impact_by_process = df_lcia.groupby("Process Name")[impact_column].sum()

    # Step 2: Normalize by product output amount if applicable
    normalized_impact = {}

    for process in processes:
        name = process.name
        name_norm = name.strip().lower()
        total_impact = impact_by_process.get(name_norm, 0)

        # Find product flows in outputs
        product_flows = [
            f for f in process.outputs
            if getattr(f, 'description', '').strip().lower() == "product"
        ]

        if product_flows:
            product_amount = sum(f.amount for f in product_flows)
            if product_amount > 0:
                normalized_impact[name] = total_impact / product_amount
            else:
                normalized_impact[name] = None  # Avoid divide by zero
        else:
            normalized_impact[name] = total_impact

    return normalized_impact

import pandas as pd
import matplotlib.pyplot as plt


def plot_normalized_stacked_impact_by_process_multi(
    df_dict,
    processes,
    selected_processes,
    save_path=None
):
    selected_normalized = [p.strip().lower() for p in selected_processes]

    # Step 1: Map product output per process
    product_amount_map = {}
    for proc in processes:
        proc_name = proc.name.strip()
        proc_norm = proc_name.lower()
        product_flows = [f for f in proc.outputs if getattr(f, 'description', '').strip().lower() == "product"]
        total_product = sum(f.amount for f in product_flows)
        product_amount_map[proc_norm] = total_product if total_product > 0 else None

    all_data = []

    # Step 2: Collect normalized impacts
    for scenario_name, df in df_dict.items():
        df = df.copy()
        df["Process Name"] = df["Process Name"].str.strip()
        df["Scenario"] = scenario_name

        # Smart impact column
        impact_col = "Adjusted Impact (kg CO2e)" if any(x in scenario_name.lower() for x in ["displacement"]) else "Impact (kg CO2e)"

        for _, row in df.iterrows():
            proc_name = row["Process Name"]
            proc_norm = proc_name.lower()

            if proc_norm not in selected_normalized:
                continue

            product_amount = product_amount_map.get(proc_norm)
            impact_value = row.get(impact_col, 0)

            if product_amount and product_amount > 0:
                norm_impact = impact_value / product_amount
            else:
                norm_impact = impact_value

            all_data.append({
                "Scenario": scenario_name,
                "Process Name": proc_name,
                "Flow Name": row["Flow Name"],
                "Normalized Impact": norm_impact
            })

    if not all_data:
        print("⚠️ No matching processes or flows found.")
        return

    df_combined = pd.DataFrame(all_data)

    # Pivot for plotting
    pivot_df = df_combined.pivot_table(
        index=["Scenario", "Process Name"],
        columns="Flow Name",
        values="Normalized Impact",
        aggfunc="sum",
        fill_value=0
    )

    # ✅ Drop flows with zero impact across all processes
    pivot_df = pivot_df.loc[:, (pivot_df != 0).any(axis=0)]

    # Plot
    ax = pivot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        edgecolor='black'
    )

    # Total labels on bars
    for i, (idx, row) in enumerate(pivot_df.iterrows()):
        total = row.sum()
        ax.text(i, total + 0.01 * pivot_df.values.max(), f"{total:.4f}", ha='center', va='bottom', fontsize=9)

    process_label = ", ".join(selected_processes)
    plt.title(f"Processes: {process_label}")
    plt.ylabel("Impact (kg CO2e per kg product)")
    #plt.xlabel(selected_processes[0])
    # Custom x-axis labels: only scenario name
    xtick_labels = [idx[0] for idx in pivot_df.index]  # extract only scenario part
    plt.xticks(ticks=range(len(xtick_labels)), labels=xtick_labels, rotation=0, ha='center')


    # ✅ Only show legend for active flows
    if pivot_df.shape[1] > 0:
        plt.legend(title="Flow Name", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend().remove()

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # ✅ Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")

    plt.show()



