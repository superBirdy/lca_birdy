import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class Flow:
    """
    Represents a material or energy flow in a process.
    """
    def __init__(self, name, amount, unit, flow_type):
        self.name = name.strip()
        self.amount = amount
        self.unit = unit.strip()
        self.flow_type = flow_type.strip().lower()

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
    Expected columns: Process Name, Flow Name, Type, Amount, Unit
    """
    df = pd.read_excel(filepath)
    required_cols = ['Process Name', 'Flow Name', 'Type', 'Amount', 'Unit']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Excel file must contain columns: {required_cols}")

    processes = {}

    for _, row in df.iterrows():
        process_name = str(row['Process Name']).strip()

        # ✅ Only create once
        if process_name not in processes:
            processes[process_name] = Process(process_name)

        # Create the flow object
        flow = Flow(
            name=str(row['Flow Name']).strip(),
            amount=float(row['Amount']),
            unit=str(row['Unit']).strip(),
            flow_type=str(row['Type']).strip()
        )

        # ✅ This will now always work
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
            factor_info = impact_factors.get(f.name, {'ghg': 'N/A', 'factor': 0, 'unit': f.unit})
            ghg = factor_info['ghg']
            factor = factor_info['factor']
            impact = f.amount * factor
            rows.append({
                "Process Name": p.name,
                "Flow Name": f.name,
                "Flow Type": f.flow_type,
                "Amount": f.amount,
                "Unit": f.unit,
                "GHG": ghg,
                "Emission Factor": factor,
                "Impact (kg CO2e)": impact
            })

    return pd.DataFrame(rows)

