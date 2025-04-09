import pandas as pd
from collections import defaultdict
from .utils import normalize_flow_name

class Flow:
    def __init__(self, name, amount, unit, flow_type):
        self.name = name
        self.amount = amount
        self.unit = unit
        self.flow_type = flow_type

class Process:
    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.outputs = []

def add_flow(self, flow):
        if flow.flow_type.lower() == 'input':
            self.inputs.append(flow)
        elif flow.flow_type.lower() == 'output':
            self.outputs.append(flow)

def read_process_data(filepath):
    df = pd.read_excel(filepath)
    processes = defaultdict(lambda: Process(None))
    for _, row in df.iterrows():
        process_name = row['Process Name']
        flow = Flow(row['Flow Name'], row['Amount'], row['Unit'], row['Type'])
        if processes[process_name].name is None:
            processes[process_name].name = process_name
        processes[process_name].add_flow(flow)
    return list(processes.values())

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