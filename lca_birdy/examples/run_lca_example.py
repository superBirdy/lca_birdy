from lca_birdy.processor import read_process_data, load_emission_inventory, detailed_lcia

processes = read_process_data("lca_tool/sample_data/process_data.xlsx")
factors = load_emission_inventory("lca_tool/sample_data/ipcc2021_emission_factors.xlsx")

df = detailed_lcia(processes, factors)
print(df)
print(f"\n🌍 Total Impact: {df['Impact (kg CO2e)'].sum():.2f} kg CO2e")