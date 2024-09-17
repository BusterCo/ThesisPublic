from src.settings import Settings # SRC is simulation environment
from src.processor import Processor
from src.compiler import recompile
from custom.DataCreator import Scenario
from custom.PC_Class import PC
import pandas as pd
import os
import time

PC_obj = PC()

def add_to_totals(df_add, scenario_name):
    scenario_totals_path =  # Path to totals result here
    if not os.path.exists(scenario_totals_path):
        df_add.to_csv(scenario_totals_path, index=False)
        return
    df = pd.read_csv(scenario_totals_path)
    df = pd.concat([df, df_add], ignore_index=True)
    df.to_csv(scenario_totals_path, index=False)
    
def aquire_data(outcome_id, instance_name, scenario):
    path1 =  # Path to result simulation environment here
    df = pd.read_csv(path1)
    df = df[df["type"] != -1]
    orders = scenario.return_orderid_dict()
    start_hubs = scenario.return_order_closest_depot()
    df['SHIPMENTNUMBER'] = df['idNode'].map(orders)
    df["originHub"] = df["idNode"].map(start_hubs)
    return df

def solve_scenario(instance_name):
    set_instance_name(instance_name)
    settings = Settings()
    processor = Processor(settings)
    if settings.processing["recompile"]:
        recompile(settings.paths["build"])
    
    processor.process()
    run_id = processor.solver.run_id
    return run_id

def set_instance_name(instance_name):
    import json
    with open("user_settings.json", 'r') as f:
        settings_user = json.load(f)
    settings_user["metaheuristic"]["instName"] = instance_name
    with open("user_settings.json", 'w') as f:
        json.dump(settings_user, f, indent=4)


outcome_ids = []
path = # Path to scenarios here
files = os.listdir(path)
files = [files.replace(".csv", "") for files in files]
output_files = [f for f in os.listdir(# Path to scenario results here ) if f.endswith(".csv")]
run_subscenarios = []
for file in output_files:
    df = pd.read_csv("# Path to scenario results here" + file)
    run_subscenarios += list(df["instance_name"].unique())
files_to_do = [file for file in files if file not in run_subscenarios]
print(files_to_do)

for instance_name in files_to_do:
    print(instance_name)
    start_time = time.time()
    
    scenario_date = instance_name.split("_")[0]
    scenario_time = instance_name.split("_")[1]
    scenario_number = int(instance_name.split("_")[3])
    if scenario_number == 0:
        scenario_name = "Base_" + scenario_date + "_" + scenario_time
    else:
        scenario_name = "Scenario_" + scenario_date + "_" + scenario_time
    if "Dumb" in instance_name:
        scenario_name = "DumbScenario_" + scenario_date + "_" + scenario_time

    scenario = Scenario(path, instance_name, PC_obj)
    scenario.write_instance()
    outcome_id = solve_scenario(instance_name)
    print(outcome_id)
    outcome_ids.append(outcome_id)
    df = aquire_data(outcome_id, instance_name, scenario)
    df["instance_name"] = instance_name

    add_to_totals(df, scenario_name)
    end_time = time.time()
    print(f"Duration for {instance_name}: {end_time - start_time:.2f} seconds")

print(outcome_ids)