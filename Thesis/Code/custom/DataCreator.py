
import csv
import os
import time
class InstanceFileWriter:
    def __init__(self, filename, instance_name, directory=None):
        self.filename = filename
        self.instance_name = instance_name
        self.directory = directory
        self.vehicle_types = []
        self.vehicle_types_info = []
        self.scheduled_lines = []
        self.nodes = []
        self.depots = []


    def add_vehicle_type(self, idVehicleType, costFixed, costVariable, speed, capacity, alpha, beta, timeMax, timeDriveMax):
        self.vehicle_types.append({
            'idVehicleType': idVehicleType,
            'costFixed': costFixed,
            'costVariable': costVariable,
            'speed': speed,
            'capacity': capacity,
            'alpha': alpha,
            'beta': beta,
            'timeMax': timeMax,
            'timeDriveMax': timeDriveMax
        })

    def add_vehicle_type_info(self, idVehicleType, idDepot, nVehicles):
        self.vehicle_types_info.append({
            'idVehicleType': idVehicleType,
            'idDepot': idDepot,
            'nVehicles': nVehicles
        })

    def add_scheduled_line(self, idFrom, idTo, costRequest, capacity, timeDep, timeArr):
        self.scheduled_lines.append({
            'idFrom': idFrom,
            'idTo': idTo,
            'costRequest': costRequest,
            'capacity': capacity,
            'timeDep': timeDep,
            'timeArr': timeArr
        })

    def add_node(self, idNode, idDepot, latitude, longitude, type, timeServ, qDel, qCol, twDepotStart, twDepotEnd, twCustStart, twCustEnd, pc=None):
        self.nodes.append({
            'idNode': idNode,
            'idDepot': idDepot,
            'latitude': latitude,
            'longitude': longitude,
            'type': type,
            'timeServ': timeServ,
            'qDel': int(qDel),
            'qCol': int(qCol),
            'twDepotStart': twDepotStart,
            'twDepotEnd': twDepotEnd,
            'twCustStart': twCustStart,
            'twCustEnd': twCustEnd,
            'pc': pc
        })

    def add_depot(self, idNode, idDepot, latitude, longitude, pc=None):
        self.depots.append({
            'idNode': idNode,
            'idDepot': idDepot,
            'latitude': latitude,
            'longitude': longitude,
            'type': -1,
            'timeServ': 0,
            'qDel': 0,
            'qCol': 0,
            'twDepotStart': 0,
            'twDepotEnd': 86400,
            'twCustStart': 0,
            'twCustEnd': 86400,
            'pc': pc
        })

    def write_file(self):
        self.nVehTypes = len(self.vehicle_types)
        self.nVehTypesInfo = len(self.vehicle_types_info)
        self.nSL = len(self.scheduled_lines)
        self.nNodes = len(self.nodes) + len(self.depots)

        file_path = os.path.join(self.directory, self.filename) if self.directory else self.filename

        with open(file_path, 'w', newline='\n') as file:
            writer = csv.writer(file, delimiter='\t')
            # Write header
            writer.writerow([self.instance_name, self.nVehTypes, self.nVehTypesInfo, self.nSL, self.nNodes])
            
            # Write vehicle types
            writer.writerow([])  # Empty line as separator
            writer.writerow(['[Vehicle Types] - ID_veh_type, fixed_cost, variable_cost, speed, capacity, alpha, beta, max_time, max_drive_time'])
            for vt in self.vehicle_types:
                writer.writerow([
                    vt['idVehicleType'], vt['costFixed'], "{:.2f}".format(vt['costVariable']), vt['speed'],
                    vt['capacity'], vt['alpha'], vt['beta'], vt['timeMax'], vt['timeDriveMax']
                ])
            
            # Write vehicle types info
            writer.writerow([])  # Empty line as separator
            writer.writerow(['[Number of Vehicle Types] - ID_veh_type, ID_depot, number_of_vehicles'])
            for vti in self.vehicle_types_info:
                writer.writerow([
                    vti['idVehicleType'], vti['idDepot'], vti['nVehicles']
                ])
            
            # Write scheduled lines
            writer.writerow([])  # Empty line as separator
            writer.writerow(['[Scheduled Lines] - ID_from, ID_to, cost_per_req, capacity, dep_time, arr_time'])
            for sl in self.scheduled_lines:
                writer.writerow([
                    sl['idFrom'], sl['idTo'], "{:.3f}".format(sl['costRequest']), sl['capacity'],
                    sl['timeDep'], sl['timeArr']
                ])
            
            # Write nodes
            writer.writerow([])  # Empty line as separator
            writer.writerow(['[Nodes/requests] ID_node, ID_depot, lat, long, type, service_time, q_del, q_col, tw_depot_start, tw_depot_end, tw_cust_start, tw_cust_end, PC'])
            for depot in self.depots:
                depot_data = [
                    depot['idNode'], depot['idDepot'], 
                    "{:.5f}".format(depot['latitude']), "{:.5f}".format(depot['longitude']),
                    depot['type'], depot['timeServ'], depot['qDel'], depot['qCol'],
                    depot['twDepotStart'], depot['twDepotEnd'], depot['twCustStart'], depot['twCustEnd']
                ]
                if depot['pc'] is not None:
                    depot_data.append(depot['pc'])
                writer.writerow(depot_data)

            for node in self.nodes:
                node_data = [
                    node['idNode'], node['idDepot'], 
                    "{:.5f}".format(node['latitude']), "{:.5f}".format(node['longitude']),
                    node['type'], node['timeServ'], node['qDel'], node['qCol'],
                    node['twDepotStart'], node['twDepotEnd'], node['twCustStart'], node['twCustEnd']
                ]
                if node['pc'] is not None:
                    node_data.append(node['pc'])
                writer.writerow(node_data)

def seconds_since_midnight(ts):
    """ Returns the number of seconds since midnight for a given timestamp. """
    return ts.hour * 3600 + ts.minute * 60 + ts.second

import json
import pandas as pd
import os
class Scenario:
    def __init__(self, path, instance_name, PC_obj):
        self.depots_file = "depots.json"
        self.variables = "variables.json"
        self.instance_name = instance_name
        self.PC_obj = PC_obj
        self.load_depots()
        self.load_orders(path)
        self.load_variables()
        self.create_orders_dict()

    def return_orderid_dict(self):
        export_dict = {}
        for order in self.orders:
            export_dict[self.orders[order]["idNode"]] = order
        return export_dict
    
    def return_order_closest_depot(self):
        export_dict = {}
        for order in self.orders:
            export_dict[self.orders[order]["idNode"]] = int(self.orders[order]["idDepot"])
        return export_dict
        

    def load_depots(self):
        direct = os.getcwd()
        folders = "\\data\\vos-instances\\"
        filename = direct +folders +self.depots_file
        if not os.path.exists(filename):
            raise FileNotFoundError("There is no depots file in the current directory.")
        with open(filename, 'r') as file:
            self.depots = json.load(file)
        for depot in self.depots:
            lat, lon = self.PC_obj.get_coordinates(self.depots[depot]["PC"])
            self.depots[depot]["lat"] = lat
            self.depots[depot]["lon"] = lon
      

    def load_orders(self, path):
        filename = path + "//" +self.instance_name + ".csv"
        if not os.path.exists(filename):
            raise FileNotFoundError("There is no orders file in the current directory.")
        # Loads the orders from the csv file
        self.df_orders = pd.read_csv(filename)
        for column in ['CREATIONDATETIME', 'LAAD_DATETIME_VAN', 'LAAD_DATETIME_TOT', 'LOS_DATETIME_VAN', 'LOS_DATETIME_TOT', '15CREATIONDATETIME']:
            self.df_orders[column] = pd.to_datetime(self.df_orders[column], errors='coerce')
        
        
    def load_variables(self):
        direct = os.getcwd()
        folders = "\\data\\vos-instances\\"
        filename = direct + folders + self.variables
        if not os.path.exists(filename):
            raise FileNotFoundError("There is no variables file in the current directory.")
        with open(filename, 'r') as file:
            self.variables = json.load(file)

    def create_orders_dict(self):
        node_num = len(self.depots)
        self.orders = {}
        for _, row in self.df_orders.iterrows():
            lat, lon = self.PC_obj.get_coordinates(row["LOS_CPC"])
            closest_depot_pc = self.PC_obj.get_closest_hub(row["LAAD_CPC"], [x["PC"] for x in self.depots.values()])
            closest_depot = [x["depotId"] for x in self.depots.values() if x["PC"] == closest_depot_pc][0]
            self.orders[row["SHIPMENTNUMBER"]] = {
                'idNode': node_num,
                'idDepot': closest_depot,
                'latitude': lat,
                'longitude': lon,
                "type": 1,
                "timeServ": int(row["PALLETPLAATSEN"]* self.variables["serv_time_per_pallet"] + self.variables["serv_time_per_location"]),
                'qDel': int(row["PALLETPLAATSEN"] * self.variables["units_per_pallet"]),
                'qCol': 0,
                'twDepotStart': 0,
                'twDepotEnd': 24*60*60, # 24 hours
                'twCustStart': seconds_since_midnight(row["LOS_DATETIME_VAN"]),
                'twCustEnd': seconds_since_midnight(row["LOS_DATETIME_TOT"]),
        }
            node_num += 1
    
    def write_instance(self, include_linehauls=True):
        directory = os.getcwd() + "\\data\\vos-instances\\"
        writer = InstanceFileWriter(self.instance_name+".txt",self.instance_name, directory= directory)
        #Create vehicle types
        writer.add_vehicle_type(0, 
                                self.variables["fixed_vehicle_cost"], 
                                self.variables["variable_vehicle_cost"],
                                self.variables["speed"], 
                                self.variables["capacity"]*self.variables["units_per_pallet"], 
                                self.variables["alpha"], 
                                self.variables["beta"], 
                                self.variables["timeMax"], 
                                self.variables["timeDriveMax"])
        
        #Create depots + vehicles at depot
        for depot in self.depots:
            writer.add_depot(self.depots[depot]["depotId"], self.depots[depot]["depotId"], self.depots[depot]["lat"], self.depots[depot]["lon"])
            writer.add_vehicle_type_info(0, self.depots[depot]["depotId"], self.variables["num_vehicles_per_depot"])

        #Create linehauls
        for origin in self.depots:
            for destination in self.depots:
                if origin != destination:
                    for i in range(self.variables["linehauls_per_route"]):
                        distance = self.PC_obj.get_distance(self.depots[origin]["PC"], self.depots[destination]["PC"]) # in km
                        linehaul_cost = (distance * ((self.variables["linehaul_km"]/
                                                    self.variables["units_per_pallet"])/ self.variables["linehaul_capacity"])) + (4.20/self.variables["units_per_pallet"])
                        start_time_lh = self.variables["linehaul_times"][i % len(self.variables["linehaul_times"])] + i
                        travel_time_sec = int(distance / self.variables["lh_speed"] * 60 * 60) # in seconds
                        end_time_lh = start_time_lh + travel_time_sec
                        linehaul_capacity = int(self.variables["linehaul_capacity"]* self.variables["units_per_pallet"])
                        if not include_linehauls:
                            linehaul_capacity = 0
                        writer.add_scheduled_line(self.depots[origin]["depotId"], self.depots[destination]["depotId"], 
                                                linehaul_cost, 
                                                linehaul_capacity, 
                                                start_time_lh, end_time_lh)

        #Create Orders
        for order in self.orders:
            writer.add_node(self.orders[order]["idNode"], self.orders[order]["idDepot"], 
                            self.orders[order]["latitude"], self.orders[order]["longitude"], 
                            self.orders[order]["type"], self.orders[order]["timeServ"],
                            self.orders[order]["qDel"], self.orders[order]["qCol"], 
                            self.orders[order]["twDepotStart"], self.orders[order]["twDepotEnd"], 
                            self.orders[order]["twCustStart"], self.orders[order]["twCustEnd"], 
                            )
        writer.write_file()