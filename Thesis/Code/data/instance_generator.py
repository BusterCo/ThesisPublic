"""
Random instance generator for the MDDCVRP-SSL problem. 
Parameters are set to generate realistic instances for the MDDCVRP-SSL problem.
The generator creates up to four depots and an arbitrary number of customers.

Written by: Siem ter Braake, 2024
"""


import os
import random
import math

class InstanceGenerator:
    def __init__(self, id, n_cust, n_depot) -> None:
        self.id = id                    # instance id
        self.n_customers = n_cust       # number of customers 
        self.n_depots = n_depot         # number of depots (max 4)
        self.quantity_min = 50          # minimum request quantity
        self.quantity_max = 1500        # maximum request quantity
        self.f_diff_cust = 0.2          # factor to difficult customers, with narrow time windows
        self.diff_cust_tw_len = 7200    # time window length of difficult customers
        self.f_sl_cust = 0.4            # factor to scheduled line customers
        self.tw_min = 28800             # minimum time window   
        self.tw_max = 64800             # maximum time window
        self.t_service_min = 600        # minimum service time
        self.t_service_max = 1200       # maximum service time
        self.lat_min = 51.00            # minimum latitude
        self.lat_max = 52.25            # maximum latitude
        self.lon_min = 4.75             # minimum longitude
        self.lon_max = 6.75             # maximum longitude
        self.vehicle_types = [
            [0, 20, 1.50, 80, 3200, 2400, 2800, 54000, 22000],
            [1, 20, 2.20, 50, 7000, 6000, 6200, 54000, 32400]
        ]
        self.t_sl_depot = [
            [4386,6334],
            [24681,26629],
            [54527,56475],
            [75274,77222]
        ]
        self.n_cust_per_route = 5      # number of customers per route
        self.sl_cost = 0.1              # scheduled line cost per kg
        self.sl_capacity = 13000         # scheduled line capacity

    def distance(self, lat1, lon1, lat2, lon2):
        # Calculate Haversine distance between two coordinates
        R = 6371e3
        phi1 = lat1 * math.pi / 180
        phi2 = lat2 * math.pi / 180
        delta_phi = (lat2 - lat1) * math.pi / 180
        delta_lambda = (lon2 - lon1) * math.pi / 180
        a = math.sin(delta_phi / 2) * math.sin(delta_phi / 2) + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) * math.sin(delta_lambda / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = R * c
        return d

    def create_instance(self):
        # create random depot locations, but make sure they are not too close to each other 
        self.depots = []
        limits = [[self.lat_min, self.lat_min + (self.lat_max - self.lat_min) / 2, self.lon_min, self.lon_min + (self.lon_max - self.lon_min) / 2],
                  [self.lat_min + (self.lat_max - self.lat_min) / 2, self.lat_max, self.lon_min + (self.lon_max - self.lon_min) / 2, self.lon_max],
                  [self.lat_min, self.lat_min + (self.lat_max - self.lat_min) / 2, self.lon_min, self.lon_min + (self.lon_max - self.lon_min) / 2],
                  [self.lat_min + (self.lat_max - self.lat_min) / 2, self.lat_max, self.lon_min + (self.lon_max - self.lon_min) / 2, self.lon_max]]
                  
        for i in range(self.n_depots):
            limit = limits[i]
            lat = random.uniform(limit[0], limit[1])
            lon = random.uniform(limit[2], limit[3])
            self.depots.append([i, lat, lon])
        
        # create random customer requests
        self.customers = []
        for i in range(self.n_customers):
            # Set random coordinate, quantity, service time and shipment type
            lat = random.uniform(self.lat_min, self.lat_max)
            lon = random.uniform(self.lon_min, self.lon_max)
            quantity = random.randint(self.quantity_min, self.quantity_max)
            t_service = random.randint(self.t_service_min, self.t_service_max)
            shipment_type = random.randint(0, 1)
            if shipment_type == 0:
                q_collect = quantity
                q_deliver = 0
            else:
                q_collect = 0
                q_deliver = quantity
            # Set time windows, if difficult customer, set narrow time window with random start time
            sl = random.random() < self.f_sl_cust
            # Set international departs and arrivals, last arrival is at 9AM, first departure is at 4:30PM
            if sl and shipment_type == 1:
                tw_depot_start = random.randint(0, 9*60*60)
                tw_depot_end = 24*60*60
            elif sl and shipment_type == 0:
                tw_depot_start = 0
                tw_depot_end = random.randint(17*60*60, 24*60*60)
            else:
                tw_depot_start = 0
                tw_depot_end = 24*60*60
            difficult = random.random() < self.f_diff_cust
            if difficult:
                tw_start = random.randint(self.tw_min, self.tw_max - self.diff_cust_tw_len)
                tw_end = tw_start + self.diff_cust_tw_len
                # Make sure the end is more than 3h after the depot start
                if tw_end - tw_depot_start < 3*60*60:
                    tw_end = tw_depot_start + 3*60*60
            else:
                tw_start = self.tw_min
                tw_end = self.tw_max
            # Set depot randomly, but weighted by distance to depot
            distances = []
            for depot in self.depots:
                distances.append(self.distance(lat, lon, depot[1], depot[2]))
            total_distance = sum(distances)
            weights = [1 - distance / total_distance for distance in distances]
            depot_id = random.choices(self.depots, weights)[0][0]
            # Add customer to list
            self.customers.append([i+self.n_depots, depot_id, lat, lon, shipment_type, t_service, q_deliver, q_collect, 
                               tw_depot_start, tw_depot_end, tw_start, tw_end])

        # Set the number of vehicles per type
        self.n_vehicles = []
        n_vehicles = self.n_customers // (self.n_cust_per_route*len(self.vehicle_types)*self.n_depots)
        for i_depot in range(self.n_depots):
            for i_vehicle in range(len(self.vehicle_types)):
                self.n_vehicles.append([i_vehicle, i_depot, max(n_vehicles,2)])
        
        # Create schedules lines
        self.sls = []
        for i in range(self.n_depots):
            for j in range(self.n_depots):
                if i != j:
                    for i_line in range(len(self.t_sl_depot)):
                        self.sls.append([i, j, self.sl_cost, self.sl_capacity, self.t_sl_depot[i_line][0], self.t_sl_depot[i_line][1]])

    def write_instance(self):
        dir = "data/mddcvrp-ssl-instances"
        name = f"T_{self.n_depots}_{len(self.t_sl_depot)}_{self.n_customers}_[{self.id}]"
        # Write instance file
        with open(dir+"/"+name+".txt", 'w') as f:
            # write name
            f.write(f"{name} {len(self.vehicle_types)} {len(self.n_vehicles)} {len(self.sls)} {self.n_depots+self.n_customers} \n")
            # write vehicle types
            f.write("\n[Vehicle Types] - ID_veh_type, fixed_cost, variable_cost, Speed, Capacity, alpha, beta, max_time, max_drive_time\n")
            for vehicle_type in self.vehicle_types:
                f.write(f"{vehicle_type[0]}\t{vehicle_type[1]}\t{vehicle_type[2]}\t{vehicle_type[3]}\t{vehicle_type[4]}\t{vehicle_type[5]}\t{vehicle_type[6]}\t{vehicle_type[7]}\t{vehicle_type[8]}\n")
            # write vehicles
            f.write("\n[Number of Vehicle Types] - ID_veh_type, ID_depot, number_of_vehicles\n")
            for n_vehicle in self.n_vehicles:
                f.write(f"{n_vehicle[0]}\t{n_vehicle[1]}\t{n_vehicle[2]}\n")
            # write scheduled lines
            f.write("\n[Scheduled Lines] - ID_from, ID_to, cost_per_req, capacity, dep_time, arr_time\n")
            for sl in self.sls:
                f.write(f"{sl[0]}\t{sl[1]}\t{sl[2]}\t{sl[3]}\t{sl[4]}\t{sl[5]}\n")
            # Write nodes
            n_space = 12
            f.write("\n[Nodes/requests]: ID_node, ID_depot, lat, long, type, service_time, q_del, q_col, tw_depot_start, tw_depot_end, tw_cust_start, tw_cust_end\n")
            for depot in self.depots:
                f.write(f"{depot[0]:<10}{depot[0]:<10}{format(depot[1], '.5f'):<10}{format(depot[2], '.5f'):<10}-1        0         0         0         0         86400     0         86400\n")
            for customer in self.customers:
                f.write(f"{customer[0]:<10}{customer[1]:<10}{format(customer[2],'.5f'):<10}{format(customer[3],'.5f'):<10}{customer[4]:<10}{customer[5]:<10}{customer[6]:<10}{customer[7]:<10}{customer[8]:<10}{customer[9]:<10}{customer[10]:<10}{customer[11]:<10}\n")

def create_single():
    id = 1
    n_cust = 100
    n_depot = 2
    instance_generator = InstanceGenerator(id, n_cust, n_depot)
    instance_generator.create_instance()
    instance_generator.write_instance()

def create_multiple():
    id_list = list(range(0, 10))
    cust_size_list = [2000]
    depot_size_list = [2]
    for cust_size in cust_size_list:
        for depot_size in depot_size_list:
            for id in id_list:
                instance_generator = InstanceGenerator(id, cust_size, depot_size)
                instance_generator.create_instance()
                instance_generator.write_instance()

if __name__ == "__main__":
    create_multiple()


            
