import os

DIR_BENCHMARK = "data/gehring-homberger-raw/"

def main():
    filenames = os.listdir(DIR_BENCHMARK)
    for filename in filenames:
        transform_instance(filename)

def transform_instance(filename: str) -> None:
    with open(f"data/gehring-homberger-raw/{filename}", "r") as f:
        lines = f.readlines()
    
    # Extract vehicle data
    name = lines[0].strip()
    vehicle_data_line = next(i for i, line in enumerate(lines) if 'CAPACITY' in line) + 1
    vehicle_data = lines[vehicle_data_line].split()
    n_veh = int(vehicle_data[0])
    veh_cap = int(vehicle_data[1])

    # Extract node data
    start = next(i for i, line in enumerate(lines) if 'CUST NO.' in line) + 2
    end = len(lines)
    node_data = [line.split() for line in lines[start:end]]

    
    with open(f"data/mddcvrp-ssl-instances/"+name+".txt", 'w') as f:
        f.write(f"{name} 1 1 0 {end-start} \n\n")
        f.write("[Vehicle Types] - ID_veh_type, fixed_cost, variable_cost, Speed, Capacity, alpha, beta, max_time, max_drive_time\n")
        f.write(f"0\t0\t1\t3600\t{veh_cap}\t{veh_cap}\t{veh_cap}\t1000000\t1000000\n\n")
        f.write("[Number of Vehicle Types] - ID_veh_type, ID_depot, number_of_vehicles\n")
        f.write(f"0\t0\t{n_veh}\n\n")
        f.write("[Scheduled Lines] - ID_from, ID_to, cost_per_req, capacity, dep_time, arr_time\n\n")
        f.write("[Nodes/requests]: ID_node, ID_depot, lat, long, type, service_time, q_del, q_col, tw_depot_start, tw_depot_end, tw_cust_start, tw_cust_end\n")
        # Add the depot first, and then the nodes in the order they appear in the file
        f.write(f"0\t0\t{node_data[0][1]}\t{node_data[0][2]}\t-1\t0\t0\t0\t{node_data[0][4]}\t{node_data[0][5]}\t0\t0\n")
        for node in node_data[1:]:
            f.write(f"{node[0]}\t0\t{node[1]}\t{node[2]}\t0\t{node[6]}\t0\t{node[3]}\t{node_data[0][4]}\t{node_data[0][5]}\t{node[4]}\t{node[5]}\n")
            
if __name__ == '__main__':
    main()