import random
from utils.utils import compute_total_cost

def random_removal(solution, num_remove):
    """
    Randomly selects and removes #num_remove nodes from a VRP solution

    Return:
    List[List[int]]: destroyed VRP solution
    List[int]: removed nodes
    """
    # Create a set of all unique visited customers
    visited_customers = sorted(set(cust for route in solution for cust in route))

    # Safety check
    assert num_remove <= len(visited_customers), \
        f"Cannot remove {num_remove} from only {len(visited_customers)} customers"

    to_remove = set(random.sample(visited_customers, num_remove))

    partial_solution = []
    removed_customers = []

    for route in solution:
        new_route = [cust for cust in route if cust not in to_remove]
        removed = [cust for cust in route if cust in to_remove]
        if new_route:
            partial_solution.append(new_route)
        removed_customers.extend(removed)

    # Final integrity check
    assert set(removed_customers) == to_remove, \
        f"Mismatch: intended to remove {to_remove}, but removed {set(removed_customers)}"

    return partial_solution, removed_customers


def worst_removal(instance, solution, num_remove):
    """
    Selects the #num_remove worst (most expensive) nodes and removes them from a route

    Return:
    List[List[int]]: destroyed VRP solution
    List[int]: removed nodes
    """
    # Save all customers in a flat list
    flat_customers = [cust for route in solution for cust in route]

    # Calculate the increase in cost for each customer removal
    customer_cost_impact = {}

    for cust in flat_customers:
        # Try removing the customer and compute the cost change
        partial_solution = []
        for route in solution:
            new_route = [c for c in route if c != cust]
            if new_route:  # Only add non-empty routes
                partial_solution.append(new_route)

        # Compute the cost of the modified solution
        new_cost = compute_total_cost(partial_solution, instance["edge_weight"])
        old_cost = compute_total_cost(solution, instance["edge_weight"])
        cost_increase = new_cost - old_cost
        customer_cost_impact[cust] = cost_increase

    # Sort customers by the largest cost increase (worst removal first)
    worst_customers = sorted(customer_cost_impact, key=customer_cost_impact.get, reverse=True)

    # Select the top `num_remove` worst customers
    to_remove = set(worst_customers[:num_remove])

    # Rebuild the solution after removing the worst customers
    partial_solution = []
    removed_customers = []

    for route in solution:
        new_route = [cust for cust in route if cust not in to_remove]
        removed = [cust for cust in route if cust in to_remove]
        if new_route:
            partial_solution.append(new_route)
        removed_customers.extend(removed)

    return partial_solution, removed_customers
