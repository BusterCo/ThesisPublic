from src.settings import Settings
from src.processor import Processor
from src.compiler import recompile
from custom.DataCreator import Scenario
from custom.PC_Class import PC
import pandas as pd
import os

PC_obj = PC()

def add_to_totals(df_add):
    path = # Path to total results here
    df = pd.read_csv(path)
    df = pd.concat([df, df_add], ignore_index=True)
    df.to_csv(path, index=False)
    
def aquire_data(outcome_id, instance_name, scenario):
    path1 = # Path to result simulation environment here
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
path = #path to scenarios here
files = os.listdir(path)
files = [files.replace(".csv", "") for files in files]

for instance_name in files:
    print(instance_name)
    scenario = Scenario(path, instance_name, PC_obj)
    scenario.write_instance()
    
    outcome_id = solve_scenario(instance_name)
    print(outcome_id)
    outcome_ids.append(outcome_id)
    df = aquire_data(outcome_id, instance_name, scenario)
    add_to_totals(df)

print(outcome_ids)