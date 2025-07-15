import random
from utils.utils import compute_total_cost

def random_removal(solution, num_remove):
    # save all customers
    flat_customers = [cust for route in solution for cust in route]

    # randomly select customers to remove -- num_remove = number of customers you want to remove
    to_remove = set(random.sample(flat_customers, num_remove))
    
    partial_solution = []
    removed_customers = []

    for route in solution:
        # rebuild partial routes
        new_route = [cust for cust in route if cust not in to_remove]
        removed = [cust for cust in route if cust in to_remove]
        if new_route:
            partial_solution.append(new_route)
        removed_customers.extend(removed)

    return partial_solution, removed_customers


def worst_removal(instance, solution, num_remove):
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
