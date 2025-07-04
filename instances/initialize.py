import vrplib

all_cvrp = vrplib.list_names(vrp_type="cvrp")
print(all_cvrp)

for name in all_cvrp:
    try: 
        vrplib.download_instance(name, "C:/Users/lucca/dev/large-cvrp/instances")
        vrplib.download_solution(name, "C:/Users/lucca/dev/large-cvrp/solutions")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
