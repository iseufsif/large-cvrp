import copy
import numpy as np

def two_opt_move(routes, route_idx):
    # Copy to avoid side effects
    new_routes = copy.deepcopy(routes)

    # Randomly choose a route to apply 2-opt
    if not new_routes:
        return new_routes

    route = new_routes[route_idx]
    i = np.random.randint(0, len(route) - 2)
    j = np.random.randint(i + 2, len(route))

    if len(route) < 4:
        return new_routes  # too short to improve

    # Apply 2-opt (reverse segment)
    new_route = route[:i] + list(reversed(route[i:j])) + route[j:]

    new_routes[route_idx] = new_route

    return new_routes